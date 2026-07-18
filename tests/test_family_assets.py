"""studio 侧 FAMILY_ASSETS registry + secrets selected 泛化（多模型 PR-4）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from studio.infrastructure.secrets import ModelsConfig
from studio.services.models.families import FAMILY_ASSETS, get_assets


def test_family_assets_cover_runtime_families():
    """下载资产可先于训练实现落地，但已支持训练的族绝不能缺下载清单。"""
    from studio.domain.common import FAMILY_CAPABILITIES
    from training.families import SPECS

    assert set(FAMILY_CAPABILITIES) == set(SPECS)
    assert set(FAMILY_ASSETS) >= set(SPECS)


def test_get_assets_unknown_lists_registered():
    with pytest.raises(ValueError, match="anima"):
        get_assets("no-such-family")


def test_anima_assets_surface():
    assets = get_assets("anima")
    assert assets.family_id == "anima"
    paths = assets.default_paths_for_new_version()
    assert set(paths) == {
        "transformer_path", "vae_path", "text_encoder_path", "t5_tokenizer_path",
    }


def test_krea2_assets_surface_uses_shared_vae_and_isolated_text_dir(monkeypatch):
    # 隔离真实 secrets：断言硬编码官方文件名，开发机 selected 切 raw_fp8 会假红
    from studio import secrets

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets(models={
        "selected": {"anima": "1.0", "krea2": "raw"},
    }))
    assets = get_assets("krea2")
    assert assets.family_id == "krea2"
    paths = assets.default_paths_for_new_version()
    anima_paths = get_assets("anima").default_paths_for_new_version()
    assert paths["vae_path"] == anima_paths["vae_path"]
    assert paths["transformer_path"].endswith("krea2-raw-bf16.safetensors")
    assert paths["text_encoder_path"].endswith("Qwen_Qwen3-VL-4B-Instruct")
    assert paths["t5_tokenizer_path"] == ""


# ── default_paths_for_new_version 按族派发（多模型 P4-1）───────────────────


def test_default_paths_dispatches_by_family():
    """门面函数经 registry 派发；无 family 参数 = anima（老调用方零迁移）。"""
    from studio.services import models as model_downloader

    anima = model_downloader.default_paths_for_new_version()
    assert anima == get_assets("anima").default_paths_for_new_version()
    krea2 = model_downloader.default_paths_for_new_version(family="krea2")
    assert krea2["t5_tokenizer_path"] == ""
    assert krea2["vae_path"] == anima["vae_path"]
    with pytest.raises(ValueError, match="anima"):
        model_downloader.default_paths_for_new_version(family="no-such-family")


def test_krea2_default_paths_ignore_inference_selected(tmp_path, monkeypatch):
    """Settings 选中 turbo（purpose=inference，为 Generate 页选的推理底模）时，
    新训练 version 的默认主权重不静默跟随，落回 training variant（Raw）。"""
    from studio import secrets
    from studio.services import models as model_downloader

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets(models={
        "root": str(tmp_path), "selected": {"krea2": "turbo"},
    }))
    paths = model_downloader.default_paths_for_new_version(family="krea2")
    assert paths["transformer_path"].endswith("krea2-raw-bf16.safetensors")
    # 显式 base_model 传 turbo = 用户显式选择，尊重（A1：不加底模白名单）
    explicit = model_downloader.default_paths_for_new_version(
        "turbo", family="krea2")
    assert explicit["transformer_path"].endswith("krea2-turbo-bf16.safetensors")


def test_krea2_default_paths_follow_selected_te(tmp_path, monkeypatch):
    """selected_te=fp8 → 训练/出图默认 text_encoder_path 指向 fp8 单文件目录；
    缺失/非法回退 bf16 目录。"""
    from studio import secrets
    from studio.services import models as model_downloader

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets(models={
        "root": str(tmp_path), "selected_te": {"krea2": "fp8"},
    }))
    paths = model_downloader.default_paths_for_new_version(family="krea2")
    assert paths["text_encoder_path"].endswith("qwen3vl-4b-fp8")

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets(models={
        "root": str(tmp_path), "selected_te": {"krea2": "bogus"},
    }))
    paths = model_downloader.default_paths_for_new_version(family="krea2")
    assert paths["text_encoder_path"].endswith("Qwen_Qwen3-VL-4B-Instruct")


def test_krea2_default_paths_respect_custom_selected(tmp_path, monkeypatch):
    """selected 是注册的本地 custom 权重（社区微调等，无 purpose 元数据）→ 尊重。"""
    from studio import secrets
    from studio.services import models as model_downloader

    custom = tmp_path / "krea2-community-ft.safetensors"
    custom.write_bytes(b"weights")
    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets(models={
        "root": str(tmp_path),
        "selected": {"krea2": str(custom)},
        "custom": {"krea2": [str(custom)]},
    }))
    paths = model_downloader.default_paths_for_new_version(family="krea2")
    assert paths["transformer_path"] == str(custom)


def test_catalog_sections_shape(tmp_path):
    """输出键与旧 build_catalog 内联实现一致（前端零改动的契约）。"""
    sections = get_assets("anima").catalog_sections(tmp_path, ModelsConfig())
    assert set(sections) == {"anima_main", "anima_vae", "qwen3", "t5_tokenizer"}
    assert sections["anima_main"]["selected"] == "1.0"
    assert sections["anima_main"]["variants"][0]["is_latest"] is True


def test_krea2_catalog_sections_report_raw_turbo_and_text_encoder(tmp_path):
    custom = tmp_path / "custom-krea2.safetensors"
    custom.write_bytes(b"weights")
    cfg = ModelsConfig(
        selected={"anima": "1.0", "krea2": str(custom)},
        custom={"krea2": [str(custom)]},
    )
    sections = get_assets("krea2").catalog_sections(tmp_path, cfg)
    assert set(sections) == {
        "krea2_main", "krea2_text_encoder", "krea2_text_encoder_fp8",
    }
    fp8_te = sections["krea2_text_encoder_fp8"]
    assert fp8_te["repo"] == "Comfy-Org/Krea-2"
    assert any(
        f["name"].endswith(".safetensors") for f in fp8_te["files"]
    )
    assert any(f["name"] == "config.json" for f in fp8_te["files"])
    main = sections["krea2_main"]
    assert [v["variant"] for v in main["variants"]] == [
        "raw", "raw_fp8", "turbo", "turbo_fp8",
    ]
    assert [v["purpose"] for v in main["variants"]] == [
        "training", "training", "inference", "inference",
    ]
    assert main["selected"] == str(custom)
    assert main["custom"] == [{
        "path": str(custom),
        "name": custom.name,
        "exists": True,
        "size": len(b"weights"),
        "mtime": custom.stat().st_mtime,
    }]
    assert main["license"] == "Krea 2 Community License"
    text = sections["krea2_text_encoder"]
    assert text["target_dir"].endswith("Qwen_Qwen3-VL-4B-Instruct")
    assert {f["name"] for f in text["files"]} >= {
        "config.json", "model.safetensors.index.json", "tokenizer.json",
    }


# ── secrets selected 泛化（迁移语义三向）─────────────────────────────────

def test_secrets_legacy_key_migrates():
    cfg = ModelsConfig.model_validate({"selected_anima": "preview2"})
    assert cfg.selected == {"anima": "preview2"}
    assert cfg.selected_anima == "preview2"


def test_secrets_incoming_legacy_key_overrides_merged_dict():
    # settings PUT：merged dict 同时带旧 selected（来自当前 dump）与入站
    # selected_anima（来自前端）→ 入站键必须赢
    cfg = ModelsConfig.model_validate({
        "selected": {"anima": "1.0"},
        "selected_anima": "preview3-base",
    })
    assert cfg.selected["anima"] == "preview3-base"


def test_secrets_dump_keeps_read_compat():
    # 前端读 sec.models.selected_anima —— computed_field 保证 dump 里有此键
    dumped = ModelsConfig().model_dump()
    assert dumped["selected_anima"] == "1.0"
    assert dumped["selected"] == {"anima": "1.0"}
    assert dumped["custom_anima_paths"] == []


def test_secrets_legacy_custom_anima_paths_migrate_to_family_map(tmp_path: Path):
    path = str(tmp_path / "anima.safetensors")
    cfg = ModelsConfig.model_validate({"custom_anima_paths": [path]})
    assert cfg.custom == {"anima": [path]}
    assert cfg.custom_anima_paths == [path]
