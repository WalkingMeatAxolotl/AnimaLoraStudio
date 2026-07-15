"""studio 侧 FAMILY_ASSETS registry + secrets selected 泛化（多模型 PR-4）。"""

from __future__ import annotations

import pytest

from studio.infrastructure.secrets import ModelsConfig
from studio.services.models.families import FAMILY_ASSETS, get_assets


def test_family_assets_keys_sync_with_capability_matrix_and_runtime():
    """三居所 join key 同步：studio assets ↔ studio 能力矩阵 ↔ runtime SPECS。"""
    from studio.domain.common import FAMILY_CAPABILITIES
    from training.families import SPECS

    assert set(FAMILY_ASSETS) == set(FAMILY_CAPABILITIES) == set(SPECS)


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


def test_catalog_sections_shape(tmp_path):
    """输出键与旧 build_catalog 内联实现一致（前端零改动的契约）。"""
    sections = get_assets("anima").catalog_sections(tmp_path, ModelsConfig())
    assert set(sections) == {"anima_main", "anima_vae", "qwen3", "t5_tokenizer"}
    assert sections["anima_main"]["selected"] == "1.0"
    assert sections["anima_main"]["variants"][0]["is_latest"] is True


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
