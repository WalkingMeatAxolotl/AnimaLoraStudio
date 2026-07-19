"""model_family schema 门控防线 + studio↔runtime 族数据单源（多模型 PR-3 / 刀 1 R3）。

studio/domain/common.py 是能力矩阵 / 族默认 / 采样白名单的单一权威源，
runtime SPECS 直接引用（依赖方向 runtime → studio）。本文件锁同一性——
防止将来有人把引用又复制回字面量、退化成镜像。
"""

from __future__ import annotations

from typing import get_args

import pytest

from studio.domain.common import (
    FAMILY_CAPABILITIES,
    FAMILY_CONFIG_DEFAULTS,
    FAMILY_SAMPLING,
    FIELD_CAPABILITY_REQUIREMENTS,
    MODEL_FAMILIES,
    TIMESTEP_SAMPLING_OPTION_FAMILIES,
    cap_gate,
    capability_violations,
    option_gates,
    sampling_option_gates,
)
from studio.domain.config_prune import eval_show_when
from studio.schema import TrainingConfig


def test_capability_matrix_single_source():
    """SPECS 的 capabilities / config_defaults 必须是 studio 表的同一对象引用。"""
    from training.families import SPECS

    assert set(FAMILY_CAPABILITIES) == set(SPECS)
    for fid, spec in SPECS.items():
        assert spec.capabilities is FAMILY_CAPABILITIES[fid], (
            f"SPECS['{fid}'].capabilities 不再引用 studio 单源——退化成镜像了"
        )
        assert spec.config_defaults is FAMILY_CONFIG_DEFAULTS[fid], fid


def test_sampling_whitelist_single_source():
    """SPECS 的 samplers/schedulers 引用 FAMILY_SAMPLING 单源；首项 = 族默认值。
    default_sampler/default_scheduler 与 config_defaults 的 sample_* 仍是族内
    常量（如 KREA2_SAMPLER），锁与单源的一致性。"""
    from training.families import SPECS

    assert set(FAMILY_SAMPLING) == set(SPECS)
    for fid, spec in SPECS.items():
        source = FAMILY_SAMPLING[fid]
        assert spec.sampling.samplers is source["samplers"], fid
        assert spec.sampling.schedulers is source["schedulers"], fid
        assert source["samplers"][0] == spec.sampling.default_sampler, fid
        assert source["schedulers"][0] == spec.sampling.default_scheduler, fid
        # 族 config_defaults 声明的采样默认值（如 krea2）与 SamplingDefaults 一致
        cd = spec.config_defaults
        if "sample_sampler_name" in cd:
            assert cd["sample_sampler_name"] == spec.sampling.default_sampler, fid
        if "sample_scheduler" in cd:
            assert cd["sample_scheduler"] == spec.sampling.default_scheduler, fid
        if "sample_infer_steps" in cd:
            assert cd["sample_infer_steps"] == spec.sampling.default_steps, fid
        if "sample_cfg_scale" in cd:
            assert cd["sample_cfg_scale"] == spec.sampling.default_cfg, fid


def test_timestep_option_gate_targets_registered_strategy():
    """门控的 timestep 选项必须真实存在于 Literal 与 runtime BUILDERS。"""
    from training.timestep_samplers import BUILDERS

    literal = set(get_args(TrainingConfig.model_fields["timestep_sampling"].annotation))
    for option, fams in TIMESTEP_SAMPLING_OPTION_FAMILIES.items():
        assert option in literal, option
        assert option in BUILDERS, option
        assert set(fams) <= set(MODEL_FAMILIES), option


def test_schema_literal_matches_families():
    literal_values = get_args(TrainingConfig.model_fields["model_family"].annotation)
    assert tuple(literal_values) == MODEL_FAMILIES
    assert TrainingConfig.model_fields["model_family"].default == "anima"  # D7 零迁移


def test_cap_gate_renders_field_comparison():
    assert cap_gate("navit") == "model_family==anima"
    with pytest.raises(ValueError):
        cap_gate("warp_drive")


def test_gated_fields_follow_family_capabilities():
    values = {"model_family": "anima"}
    for field in ("t5_tokenizer_path", "navit_packing", "masked_loss",
                  "leap_enabled", "sra_enabled", "shuffle_caption"):
        extra = TrainingConfig.model_fields[field].json_schema_extra
        assert eval_show_when(extra.get("show_when"), values) is True, field
    krea2 = {"model_family": "krea2"}
    assert eval_show_when(
        TrainingConfig.model_fields["masked_loss"].json_schema_extra["show_when"],
        krea2,
    ) is True
    assert eval_show_when(
        TrainingConfig.model_fields["text_encoder_cache"].json_schema_extra["show_when"],
        krea2,
    ) is True
    for field in ("t5_tokenizer_path", "navit_packing", "leap_enabled",
                  "sra_enabled", "shuffle_caption"):
        extra = TrainingConfig.model_fields[field].json_schema_extra
        assert eval_show_when(extra.get("show_when"), krea2) is False, field


def test_default_config_valid_and_yaml_roundtrip():
    cfg = TrainingConfig()
    assert cfg.model_family == "anima"
    # 显式开启 anima 支持的能力 → 合法（分开开：navit×masked_loss 是
    # disable_when 声明的互斥，刀 2 起后端也强制）
    TrainingConfig(navit_packing=True)
    TrainingConfig(masked_loss=True)
    krea2 = TrainingConfig(model_family="krea2")
    assert krea2.shuffle_caption is False
    assert krea2.text_encoder_cache is True
    assert krea2.attention_backend == "none"
    assert krea2.timestep_sampling == "krea2_shift"
    assert (krea2.sample_sampler_name, krea2.sample_scheduler) == (
        "euler", "simple",
    )
    assert (krea2.sample_infer_steps, krea2.sample_cfg_scale) == (28, 4.5)


# ─── option 级门控 + sampler 越族值报错（多模型 P4-2，A13/C1）────────────────


def test_option_gates_render_expressions():
    assert sampling_option_gates("samplers") == {
        "er_sde": "model_family==anima",
        "dpmpp_3m_sde": "model_family==anima",
        "euler": "model_family==krea2",
    }
    assert option_gates(TIMESTEP_SAMPLING_OPTION_FAMILIES) == {
        "krea2_shift": "model_family==krea2",
    }


def test_option_show_when_wired_into_schema_fields():
    """三个 enum 字段带 option_show_when，且键都在各自 Literal 值域内。"""
    for field, kind in (
        ("sample_sampler_name", "samplers"),
        ("sample_scheduler", "schedulers"),
    ):
        extra = TrainingConfig.model_fields[field].json_schema_extra
        gates = extra.get("option_show_when")
        assert gates == sampling_option_gates(kind), field
        literal = set(get_args(TrainingConfig.model_fields[field].annotation))
        assert set(gates) <= literal, field
    ts_extra = TrainingConfig.model_fields["timestep_sampling"].json_schema_extra
    assert ts_extra.get("option_show_when") == option_gates(
        TIMESTEP_SAMPLING_OPTION_FAMILIES
    )


def test_cross_family_sampler_asymmetric_grandfather():
    """白名单外值的按族处理（#419 债 C1 的修法）：

    - anima（Literal 收紧前就存在）：euler 等历史存量静默归并——#256 迁移
      契约保留（test_comfy_parity_contract 另有原契约测试锁死）
    - krea2（Literal 时代出生，无 legacy 语料）：union 内跨族值报错，
      不静默改写显式配置
    """
    legacy = TrainingConfig(
        model_family="anima",
        sample_sampler_name="euler", sample_scheduler="krea2_shift",
    )
    assert (legacy.sample_sampler_name, legacy.sample_scheduler) == (
        "er_sde", "simple",
    )
    with pytest.raises(ValueError, match="er_sde"):
        TrainingConfig(model_family="krea2", sample_sampler_name="er_sde")
    with pytest.raises(ValueError, match="sgm_uniform"):
        TrainingConfig(model_family="krea2", sample_scheduler="sgm_uniform")


def test_krea2_shift_scheduler_value_renamed_to_simple():
    """scheduler 值 krea2_shift → simple（命名对齐 ComfyUI）：改名前落盘的
    krea2 config 经 Literal-外归并落回族默认 simple——语义等价（同一条 sigma
    时刻表），不整份报废。"""
    cfg = TrainingConfig(model_family="krea2", sample_scheduler="krea2_shift")
    assert cfg.sample_scheduler == "simple"


def test_legacy_garbage_sampler_still_migrates_to_family_default():
    """Literal 外的垃圾值任何族都归并族默认（加载健壮性），config 不整体失败。"""
    cfg = TrainingConfig(
        sample_sampler_name="ancient_free_text", sample_scheduler="karras",
    )
    assert (cfg.sample_sampler_name, cfg.sample_scheduler) == ("er_sde", "simple")
    krea2 = TrainingConfig(
        model_family="krea2",
        sample_sampler_name="ancient_free_text", sample_scheduler="karras",
    )
    assert (krea2.sample_sampler_name, krea2.sample_scheduler) == (
        "euler", "simple",
    )


def test_krea2_shift_timestep_allowed_for_any_family():
    """timestep_sampling 不设后端硬闸（A1：共享循环同代码）——仅 UI 门控。"""
    cfg = TrainingConfig(model_family="anima", timestep_sampling="krea2_shift")
    assert cfg.timestep_sampling == "krea2_shift"


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

    krea2 = prune_inactive_fields(
        TrainingConfig(model_family="krea2").model_dump(mode="python")
    )
    assert krea2["text_encoder_cache"] is True
    assert krea2["masked_loss"] is False
    for field in ("t5_tokenizer_path", "shuffle_caption", "keep_tokens",
                  "tag_dropout", "navit_packing", "leap_enabled", "sra_enabled"):
        assert field not in krea2, field
