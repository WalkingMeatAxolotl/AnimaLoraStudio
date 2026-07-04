"""PP0 — presets_io 等价于 PP0 之前的 configs_io（更名 + 异常类换名）。

复用原 test_studio_configs.py 的 IO 用例集，把字眼从 config 切换到 preset，
确保所有原有行为（命名校验、roundtrip、duplicate 冲突等）继续保持。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.services.presets import io as presets_io
from studio.schema import TrainingConfig


@pytest.fixture
def presets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pdir = tmp_path / "presets"
    pdir.mkdir()
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", pdir)
    return pdir


def _payload() -> dict:
    return TrainingConfig().model_dump(mode="python")


def test_write_then_read_roundtrip(presets_dir: Path) -> None:
    payload = _payload()
    payload["lora_rank"] = 64
    presets_io.write_preset("alpha", payload)
    assert (presets_dir / "alpha.yaml").exists()
    got = presets_io.read_preset("alpha")
    assert got["lora_rank"] == 64


def test_write_prunes_inactive_fields_read_reinflates(presets_dir: Path) -> None:
    """落盘 yaml 只带 show_when 生效的字段；读回时缺失字段补 schema 默认值。"""
    import yaml

    payload = _payload()  # optimizer_type 默认 adamw
    payload["came_beta1"] = 0.5  # 隐藏字段的残留值，不应写盘
    presets_io.write_preset("pruned", payload)

    raw = yaml.safe_load((presets_dir / "pruned.yaml").read_text(encoding="utf-8"))
    assert "came_beta1" not in raw
    assert "infonoise_K" not in raw
    assert "lr_scheduler" in raw  # disable_when 字段保留

    got = presets_io.read_preset("pruned")
    assert got["came_beta1"] == TrainingConfig().came_beta1  # 补回默认值


def test_write_invalid_rejected(presets_dir: Path) -> None:
    with pytest.raises(presets_io.PresetError):
        presets_io.write_preset("bad", {"lora_rank": "not-an-int"})
    assert not list(presets_dir.glob("*.yaml"))


def test_name_validation(presets_dir: Path) -> None:
    for bad in ("../escape", "name with space", "name/sub", "name.dot"):
        with pytest.raises(presets_io.PresetError, match="Invalid preset name"):
            presets_io.write_preset(bad, _payload())


def test_list_sorted_by_mtime(presets_dir: Path) -> None:
    import time
    presets_io.write_preset("first", _payload())
    time.sleep(0.05)
    presets_io.write_preset("second", _payload())
    items = presets_io.list_presets()
    assert [x["name"] for x in items[:2]] == ["second", "first"]


def test_delete(presets_dir: Path) -> None:
    presets_io.write_preset("to_delete", _payload())
    presets_io.delete_preset("to_delete")
    assert not (presets_dir / "to_delete.yaml").exists()


def test_delete_missing_raises(presets_dir: Path) -> None:
    with pytest.raises(presets_io.PresetError, match="not found"):
        presets_io.delete_preset("ghost")


def test_duplicate(presets_dir: Path) -> None:
    payload = _payload()
    payload["lora_rank"] = 16
    presets_io.write_preset("src", payload)
    presets_io.duplicate_preset("src", "src_copy")
    assert (presets_dir / "src_copy.yaml").exists()
    assert presets_io.read_preset("src_copy")["lora_rank"] == 16


def test_duplicate_conflict(presets_dir: Path) -> None:
    presets_io.write_preset("a", _payload())
    presets_io.write_preset("b", _payload())
    with pytest.raises(presets_io.PresetError, match="already exists"):
        presets_io.duplicate_preset("a", "b")


# ---------------------------------------------------------------------------
# parse_preset_bytes（端到端文件导入用）
# ---------------------------------------------------------------------------


def test_parse_yaml_returns_config_and_suggested_name() -> None:
    import yaml
    payload = _payload()
    payload["epochs"] = 12
    raw = yaml.safe_dump(payload, allow_unicode=True).encode("utf-8")
    config, suggested = presets_io.parse_preset_bytes(raw, "my-run.yaml")
    assert config["epochs"] == 12
    assert suggested == "my-run"


def test_parse_json_works_via_yaml_superset() -> None:
    """yaml.safe_load 同时吃 JSON —— 旧 .json 导出能直接导入回来。"""
    import json
    raw = json.dumps(_payload()).encode("utf-8")
    config, suggested = presets_io.parse_preset_bytes(raw, "old.json")
    assert config["lora_type"] == "lokr"
    assert suggested == "old"


def test_parse_drops_unknown_field() -> None:
    import yaml
    bad = _payload()
    bad["nonexistent_field"] = 123
    raw = yaml.safe_dump(bad).encode("utf-8")
    cfg, suggested = presets_io.parse_preset_bytes(raw, "bad.yaml")
    assert suggested == "bad"
    assert "nonexistent_field" not in cfg


def test_parse_migrates_legacy_attention_fields() -> None:
    import yaml
    legacy = _payload()
    legacy.pop("attention_backend", None)
    legacy["flash_attn"] = False
    legacy["xformers"] = True
    raw = yaml.safe_dump(legacy).encode("utf-8")
    cfg, _ = presets_io.parse_preset_bytes(raw, "legacy.yaml")
    assert cfg["attention_backend"] == "xformers"
    assert "flash_attn" not in cfg
    assert "xformers" not in cfg


def test_parse_rejects_non_mapping() -> None:
    # 顶层是 list 不是 dict
    raw = b"- foo\n- bar\n"
    with pytest.raises(presets_io.PresetError, match="Preset file format is invalid"):
        presets_io.parse_preset_bytes(raw, "list.yaml")


def test_parse_rejects_invalid_utf8() -> None:
    raw = b"\xff\xfe\x00\x00bogus"
    with pytest.raises(presets_io.PresetError, match="UTF-8"):
        presets_io.parse_preset_bytes(raw, "binary.yaml")


def test_parse_sanitizes_suggested_name() -> None:
    """文件名带空格 / 中文 / 特殊字符 → suggested 走 [A-Za-z0-9_-] 白名单。"""
    import yaml
    raw = yaml.safe_dump(_payload()).encode("utf-8")
    _, suggested = presets_io.parse_preset_bytes(raw, "我的 preset (v2).yaml")
    assert all(c.isalnum() or c in "_-" for c in suggested)
    assert "v2" in suggested


def test_parse_empty_filename_fallback() -> None:
    import yaml
    raw = yaml.safe_dump(_payload()).encode("utf-8")
    _, suggested = presets_io.parse_preset_bytes(raw, "")
    assert suggested == "imported"


def test_preset_path_is_public_alias() -> None:
    assert presets_io.preset_path("foo").name == "foo.yaml"
    with pytest.raises(presets_io.PresetError, match="Invalid preset name"):
        presets_io.preset_path("bad/name")


# ---------------------------------------------------------------------------
# InfoNoise 老 config 互斥兼容：_tolerant_validate 自动关 InfoNoise 保住用户原值
# ---------------------------------------------------------------------------


def test_tolerant_validate_infonoise_mutex_disables_infonoise() -> None:
    """老 yaml 同时 infonoise=on + 4 互斥字段任一非默认 → 自动关 InfoNoise
    保留用户原投入字段，"infonoise_enabled" 进 defaulted_fields 让前端 banner。
    """
    cases = [
        {"infonoise_enabled": True, "noise_enhancement_type": "offset"},
        {"infonoise_enabled": True, "loss_weighting": "detail_inv_t"},
        {"infonoise_enabled": True, "loss_type": "huber"},
        {"infonoise_enabled": True, "timestep_schedule_shift": 2.0},
    ]
    for overrides in cases:
        cfg, _, defaulted = presets_io._tolerant_validate(overrides)
        assert cfg.infonoise_enabled is False, overrides
        assert "infonoise_enabled" in defaulted, overrides
        # 用户原投入字段全保留
        for k, v in overrides.items():
            if k == "infonoise_enabled":
                continue
            assert getattr(cfg, k) == v, overrides


def test_tolerant_validate_infonoise_mutex_multi_conflict_one_pass() -> None:
    """同时 4 条互斥全冲突 → 一刀关 InfoNoise，4 字段全保留，defaulted 只有
    "infonoise_enabled"（一次性处理，不重复 append）。"""
    cfg, _, defaulted = presets_io._tolerant_validate({
        "infonoise_enabled": True,
        "noise_enhancement_type": "offset",
        "loss_weighting": "detail_inv_t",
        "loss_type": "huber",
        "timestep_schedule_shift": 2.0,
    })
    assert cfg.infonoise_enabled is False
    assert cfg.noise_enhancement_type == "offset"
    assert cfg.loss_weighting == "detail_inv_t"
    assert cfg.loss_type == "huber"
    assert cfg.timestep_schedule_shift == 2.0
    assert defaulted == ["infonoise_enabled"]


def test_write_preset_strict_path_still_rejects_infonoise_mutex(
    presets_dir: Path,
) -> None:
    """write_preset 走严格 model_validate（非 tolerant）—— 互斥配置直接拒，
    防 UI 绕过 disable_when 锁直接发 PUT 提交 silent 改值。"""
    with pytest.raises(presets_io.PresetError):
        presets_io.write_preset("conflict", {
            **_payload(),
            "infonoise_enabled": True,
            "loss_type": "huber",
        })


# ---------------------------------------------------------------------------
# 路径规范化（PP10.5）：yaml 写盘 / 读取统一绝对路径
# ---------------------------------------------------------------------------


def test_read_preset_absolutizes_relative_paths(presets_dir: Path) -> None:
    """老 yaml 里 4 模型字段是相对路径 → 读取时转为基于 REPO_ROOT 的绝对 POSIX 路径。"""
    import yaml
    from studio.paths import REPO_ROOT
    # 直接写老格式 yaml（绕过 write_preset 的 normalize），模拟老用户的预设
    raw = TrainingConfig().model_dump()
    raw["transformer_path"] = "models/diffusion_models/anima-base-v1.0.safetensors"
    raw["vae_path"] = "models/vae/qwen_image_vae.safetensors"
    (presets_dir / "legacy.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )
    got = presets_io.read_preset("legacy")
    assert Path(got["transformer_path"]).is_absolute()
    # 路径规范化：分隔符统一 `/`，无反斜杠
    assert "\\" not in got["transformer_path"]
    assert got["transformer_path"] == (
        REPO_ROOT / "models/diffusion_models/anima-base-v1.0.safetensors"
    ).resolve().as_posix()
    assert got["vae_path"] == (
        REPO_ROOT / "models/vae/qwen_image_vae.safetensors"
    ).resolve().as_posix()


def test_read_preset_normalizes_to_posix(presets_dir: Path) -> None:
    """绝对路径字段读取时规范化为 POSIX 分隔符（Windows 反斜杠 → `/`）。"""
    import yaml
    raw = TrainingConfig().model_dump()
    # 用 Path 构造再 str，在 Windows 上会得到反斜杠；其他平台保持 `/`
    abs_path = str(Path("/data/anima/custom.safetensors").resolve())
    raw["transformer_path"] = abs_path
    (presets_dir / "modern.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )
    got = presets_io.read_preset("modern")
    # 读出来一律 POSIX
    assert "\\" not in got["transformer_path"]
    assert got["transformer_path"] == Path(abs_path).as_posix()


def test_write_preset_absolutizes_relative_paths(presets_dir: Path) -> None:
    """写预设时相对路径会被规范化为绝对 POSIX 落盘 → yaml 文件本身统一格式。"""
    import yaml
    payload = _payload()
    payload["transformer_path"] = "models/foo.safetensors"
    presets_io.write_preset("alpha", payload)
    raw = yaml.safe_load((presets_dir / "alpha.yaml").read_text(encoding="utf-8"))
    assert Path(raw["transformer_path"]).is_absolute()
    assert "\\" not in raw["transformer_path"]


def test_absolutize_preserves_windows_drive_letter_on_posix(monkeypatch) -> None:
    """跨平台 bundle import：Windows 盘符 (`G:/...`) 在 POSIX 上
    `Path.is_absolute()` 返回 False，会被误拼到 REPO_ROOT 下变成
    `<repo>/G:/...`。盘符前缀必须被识别为绝对，原样保留（含反斜杠归一为 `/`）。

    monkeypatch 强制 Path.is_absolute 返回 False，模拟 Linux 处理 Windows 盘符的
    行为 —— 否则在 Windows 上跑 `Path("G:/foo").is_absolute()` 本来就是 True，
    没法 catch 修前的 bug。"""
    from studio.services.presets import io as presets_io
    from studio.services.presets.io import _absolutize_model_paths

    monkeypatch.setattr(
        presets_io.Path, "is_absolute", lambda self: False
    )

    data = {
        "transformer_path": "G:/models/diffusion_models/anima.safetensors",
        "vae_path": "D:\\anima\\vae.safetensors",
    }
    out = _absolutize_model_paths(data)
    assert out["transformer_path"] == "G:/models/diffusion_models/anima.safetensors"
    assert out["vae_path"] == "D:/anima/vae.safetensors"
