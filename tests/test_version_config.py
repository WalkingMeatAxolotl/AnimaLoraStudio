"""PP6.2 — version 私有 config + preset fork/save_as 流。"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from studio import db
from studio.services.projects import projects, versions
from studio.services import presets as preset_flow, version_config


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    # 全局 preset 池
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    from studio.services.presets import io as presets_io
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", presets_dir)
    return {"db": dbfile, "presets": presets_dir}


def _make_pv(env) -> tuple[dict, dict]:
    with db.connection_for(env["db"]) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
    return p, v


# ---------------------------------------------------------------------------
# project_specific_overrides
# ---------------------------------------------------------------------------


def test_project_specific_overrides_uses_version_dir(env) -> None:
    p, v = _make_pv(env)
    ov = version_config.project_specific_overrides(p, v)
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    assert ov["data_dir"] == str(vdir / "train")
    assert ov["output_dir"] == str(vdir / "output")
    assert ov["output_name"] == f"{p['slug']}_baseline"
    assert ov["reg_data_dir"] is None  # 没 reg meta
    assert ov["resume_lora"] is None
    assert ov["resume_state"] is None


def test_project_specific_overrides_includes_reg_when_meta_exists(env) -> None:
    p, v = _make_pv(env)
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    (vdir / "reg").mkdir(parents=True, exist_ok=True)
    (vdir / "reg" / "meta.json").write_text("{}", encoding="utf-8")
    ov = version_config.project_specific_overrides(p, v)
    assert ov["reg_data_dir"] == str(vdir / "reg")


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------


def _minimal_config(**overrides) -> dict:
    """合法的最小 TrainingConfig dict（schema defaults 已够用，只覆盖几个字段）。"""
    from studio.schema import TrainingConfig
    return {**TrainingConfig().model_dump(), **overrides}


def test_has_version_config_false_initially(env) -> None:
    p, v = _make_pv(env)
    assert version_config.has_version_config(p, v) is False


def test_write_then_read(env) -> None:
    p, v = _make_pv(env)
    cfg_in = _minimal_config(lora_rank=64)
    version_config.write_version_config(p, v, cfg_in)
    assert version_config.has_version_config(p, v) is True
    cfg_out = version_config.read_version_config(p, v)
    assert cfg_out["lora_rank"] == 64


def test_write_forces_project_overrides(env) -> None:
    """用户传错的 data_dir / output_dir 都会被服务端覆盖回项目路径。"""
    p, v = _make_pv(env)
    cfg = _minimal_config(
        data_dir="/some/wrong/path",
        output_dir="/another/wrong/path",
        output_name="hacker",
    )
    version_config.write_version_config(p, v, cfg)
    out = version_config.read_version_config(p, v)
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    assert out["data_dir"] == str(vdir / "train")
    assert out["output_dir"] == str(vdir / "output")
    assert out["output_name"] == f"{p['slug']}_baseline"


def test_read_missing_raises(env) -> None:
    p, v = _make_pv(env)
    with pytest.raises(version_config.VersionConfigError):
        version_config.read_version_config(p, v)


def test_delete_version_config(env) -> None:
    p, v = _make_pv(env)
    version_config.write_version_config(p, v, _minimal_config())
    assert version_config.delete_version_config(p, v) is True
    assert version_config.delete_version_config(p, v) is False
    assert version_config.has_version_config(p, v) is False


# ---------------------------------------------------------------------------
# fork / save_as 流
# ---------------------------------------------------------------------------


def _seed_preset(env, name: str, **overrides) -> None:
    from studio.services.presets import io as presets_io
    presets_io.write_preset(name, _minimal_config(**overrides))


def test_fork_preset_for_version_applies_overrides(env) -> None:
    p, v = _make_pv(env)
    _seed_preset(env, "tpl", lora_rank=128, data_dir="/wrong")
    cfg = preset_flow.fork_preset_for_version("tpl", p, v)
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    # 项目特定字段被强制覆盖
    assert cfg["data_dir"] == str(vdir / "train")
    assert cfg["output_name"] == f"{p['slug']}_baseline"
    # 其他字段沿用 preset
    assert cfg["lora_rank"] == 128


def test_fork_then_modify_does_not_change_preset(env) -> None:
    """version 私有 config 改动不应该回流到全局 preset。"""
    p, v = _make_pv(env)
    _seed_preset(env, "tpl", lora_rank=32)
    preset_flow.fork_preset_for_version("tpl", p, v)
    # 改 version 私有
    cfg = version_config.read_version_config(p, v)
    cfg["lora_rank"] = 128
    version_config.write_version_config(p, v, cfg)
    # 全局 preset 不受影响
    from studio.services.presets import io as presets_io
    preset_now = presets_io.read_preset("tpl")
    assert preset_now["lora_rank"] == 32


def test_save_version_config_as_preset_clears_project_fields(env) -> None:
    p, v = _make_pv(env)
    _seed_preset(env, "tpl", lora_rank=64)
    preset_flow.fork_preset_for_version("tpl", p, v)

    saved = preset_flow.save_version_config_as_preset(p, v, "my-tuned")
    # 项目特定字段被清回 schema 默认（不带项目数据外流）
    assert saved["data_dir"] == "./dataset"
    assert saved["output_dir"] == "./output"
    assert saved["output_name"] == "anima_lora"
    assert saved["reg_data_dir"] is None
    # 其他字段保留
    assert saved["lora_rank"] == 64

    # 新预设确实落到了 preset 池
    yaml_path = env["presets"] / "my-tuned.yaml"
    assert yaml_path.exists()
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert raw["lora_rank"] == 64


def test_save_as_preset_rejects_existing_without_overwrite(env) -> None:
    from studio.services.presets import io as presets_io
    p, v = _make_pv(env)
    _seed_preset(env, "tpl", lora_rank=64)
    preset_flow.fork_preset_for_version("tpl", p, v)
    # tpl 已存在 → 不带 overwrite 应该 raise
    with pytest.raises(presets_io.PresetError):
        preset_flow.save_version_config_as_preset(p, v, "tpl", overwrite=False)
    # overwrite=True 可以
    preset_flow.save_version_config_as_preset(p, v, "tpl", overwrite=True)


def test_save_as_preset_rejects_invalid_name(env) -> None:
    from studio.services.presets import io as presets_io
    p, v = _make_pv(env)
    _seed_preset(env, "tpl")
    preset_flow.fork_preset_for_version("tpl", p, v)
    with pytest.raises(presets_io.PresetError):
        preset_flow.save_version_config_as_preset(p, v, "../etc/passwd")


# ---------------------------------------------------------------------------
# auto_sync_paths toggle (PP10.5)
# ---------------------------------------------------------------------------


def _custom_path() -> str:
    """跨平台合法的绝对路径（POSIX 形式），用作"用户自定义模型路径"。

    yaml 落盘 + read 都会规范化成 POSIX，所以这里直接用 POSIX 写法做期望值。
    """
    return Path("/tmp/anima-custom/foo.safetensors").resolve().as_posix()


def _normalize_default(path_str: str) -> str:
    """default_paths_for_new_version 用 str(Path)，Windows 上是反斜杠；
    跟预设流的 normalize 一致比较时统一转 POSIX。"""
    return Path(path_str).as_posix()


def test_fork_with_toggle_on_overrides_model_paths(env, monkeypatch) -> None:
    """toggle ON：预设里的 4 模型字段 fork 时被 default_paths_for_new_version 覆盖。"""
    from studio.services import model_downloader
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: True)
    p, v = _make_pv(env)
    custom = _custom_path()
    _seed_preset(env, "tpl", transformer_path=custom)
    cfg = preset_flow.fork_preset_for_version("tpl", p, v)
    expected = _normalize_default(model_downloader.default_paths_for_new_version()["transformer_path"])
    assert cfg["transformer_path"] == expected
    assert cfg["transformer_path"] != custom


def test_fork_with_toggle_off_respects_preset(env, monkeypatch) -> None:
    """toggle OFF：fork 时尊重预设里的绝对路径，不覆盖。"""
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: False)
    p, v = _make_pv(env)
    custom = _custom_path()
    _seed_preset(env, "tpl", transformer_path=custom)
    cfg = preset_flow.fork_preset_for_version("tpl", p, v)
    assert cfg["transformer_path"] == custom


def test_save_as_preset_toggle_on_clears_model_paths(env, monkeypatch) -> None:
    """toggle ON：保存预设时 4 模型字段清回 default_paths（不带本机自定义出去）。"""
    from studio.services import model_downloader
    # fork 时用 toggle OFF 保留用户自定义路径
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: False)
    p, v = _make_pv(env)
    custom = _custom_path()
    _seed_preset(env, "tpl", transformer_path=custom)
    preset_flow.fork_preset_for_version("tpl", p, v)
    # 此时 version yaml 里 transformer_path = custom
    assert version_config.read_version_config(p, v)["transformer_path"] == custom
    # 切到 toggle ON 保存预设
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: True)
    saved = preset_flow.save_version_config_as_preset(p, v, "saved")
    expected = _normalize_default(model_downloader.default_paths_for_new_version()["transformer_path"])
    assert saved["transformer_path"] == expected
    assert saved["transformer_path"] != custom


def test_save_as_preset_toggle_off_keeps_model_paths(env, monkeypatch) -> None:
    """toggle OFF：保存预设时保留 version yaml 里的绝对路径（独立模型用户主动设置）。"""
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: False)
    p, v = _make_pv(env)
    custom = _custom_path()
    _seed_preset(env, "tpl", transformer_path=custom)
    preset_flow.fork_preset_for_version("tpl", p, v)
    saved = preset_flow.save_version_config_as_preset(p, v, "saved")
    assert saved["transformer_path"] == custom
