"""model_family schema 门控三层防线 + studio↔runtime 能力矩阵镜像同步（多模型 PR-3）。

studio/domain 刻意不 import runtime（trainer cli 反向依赖 + server sys.path），
FAMILY_CAPABILITIES 是 runtime SPECS 的镜像——本文件是锁死两者同步的唯一机制，
改任何一侧必须同时改另一侧。
"""

from __future__ import annotations

from typing import get_args

import pytest

from studio.domain.common import (
    FAMILY_CAPABILITIES,
    FIELD_CAPABILITY_REQUIREMENTS,
    MODEL_FAMILIES,
    cap_gate,
    capability_violations,
)
from studio.domain.config_prune import eval_show_when
from studio.schema import TrainingConfig


def test_capability_matrix_mirrors_runtime_specs():
    from training.families import SPECS

    assert set(FAMILY_CAPABILITIES) == set(SPECS)
    for fid, spec in SPECS.items():
        assert FAMILY_CAPABILITIES[fid] == spec.capabilities, (
            f"studio 镜像与 runtime SPECS['{fid}'].capabilities 失同步"
        )


def test_schema_literal_matches_families():
    literal_values = get_args(TrainingConfig.model_fields["model_family"].annotation)
    assert tuple(literal_values) == MODEL_FAMILIES
    assert TrainingConfig.model_fields["model_family"].default == "anima"  # D7 零迁移


def test_cap_gate_renders_field_comparison():
    assert cap_gate("navit") == "model_family==anima"
    with pytest.raises(ValueError):
        cap_gate("warp_drive")


def test_gated_fields_show_when_true_for_anima():
    """当前只有 anima → 所有门控表达式恒真，对存量行为零变化。"""
    values = {"model_family": "anima"}
    for field in ("t5_tokenizer_path", "navit_packing", "masked_loss",
                  "leap_enabled", "sra_enabled", "shuffle_caption"):
        extra = TrainingConfig.model_fields[field].json_schema_extra
        assert eval_show_when(extra.get("show_when"), values) is True, field


def test_default_config_valid_and_yaml_roundtrip():
    cfg = TrainingConfig()
    assert cfg.model_family == "anima"
    # 显式开启 anima 支持的能力 → 合法
    TrainingConfig(navit_packing=True, masked_loss=True)


def test_capability_violations_flags_unsupported(monkeypatch):
    # 模拟一个无 tag 语义 / 无 navit 的族（K2 形态），三层防线共用这个判定
    monkeypatch.setitem(FAMILY_CAPABILITIES, "fakefam", frozenset({"masked_loss", "text_cache"}))
    bad = capability_violations("fakefam", {
        "navit_packing": True, "masked_loss": True,
        "shuffle_caption": True, "keep_tokens": 0, "tag_dropout": 0.0,
    })
    assert bad == ["navit_packing", "shuffle_caption"]
    # 全部关闭 → 无违规（默认关闭字段对任何族合法）
    assert capability_violations("fakefam", {k: False for k in FIELD_CAPABILITY_REQUIREMENTS}) == []
    # 未知族不在此层重复报错（Literal / resolve_family 负责）
    assert capability_violations("no-such", {"navit_packing": True}) == []


def test_requirements_fields_exist_in_schema():
    for field in FIELD_CAPABILITY_REQUIREMENTS:
        assert field in TrainingConfig.model_fields, field


def test_prune_keeps_gated_fields_for_default_dump():
    """落盘裁剪对 model_dump（默认已填充 model_family）求值 → 门控字段不被误裁。"""
    from studio.domain.config_prune import prune_inactive_fields

    dumped = prune_inactive_fields(TrainingConfig().model_dump(mode="python"))
    assert dumped.get("model_family") == "anima"
    for field in ("t5_tokenizer_path", "shuffle_caption", "masked_loss",
                  "navit_packing", "leap_enabled", "sra_enabled"):
        assert field in dumped, field
