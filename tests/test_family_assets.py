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
    assert set(FAMILY_ASSETS) - set(SPECS) == {"krea2"}


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


def test_krea2_assets_surface_uses_shared_vae_and_isolated_text_dir():
    assets = get_assets("krea2")
    assert assets.family_id == "krea2"
    paths = assets.default_paths_for_new_version()
    anima_paths = get_assets("anima").default_paths_for_new_version()
    assert paths["vae_path"] == anima_paths["vae_path"]
    assert paths["transformer_path"].endswith("krea2-raw-bf16.safetensors")
    assert paths["text_encoder_path"].endswith("Qwen_Qwen3-VL-4B-Instruct")
    assert paths["t5_tokenizer_path"] == ""


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
    assert set(sections) == {"krea2_main", "krea2_text_encoder"}
    main = sections["krea2_main"]
    assert [v["variant"] for v in main["variants"]] == ["raw", "turbo"]
    assert [v["purpose"] for v in main["variants"]] == ["training", "inference"]
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
