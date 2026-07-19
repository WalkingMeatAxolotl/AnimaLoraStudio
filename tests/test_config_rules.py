"""disable_when 规则引擎(刀 2 / R2 v2)—— 声明式钉值/禁值的双端强制。

设计:docs/design/config-pipeline-refactor.md §6。字段元数据
disable_when/disable_value/option_disable_when 是单源规则声明,派生后端校验
(_enforce_disable_rules)、构造期 setdefault(_pin_setdefaults)、tolerant 修复
(apply_disable_rule_fixes)。历史 8 个手写互斥 validator 已退役,本文件锁
「退役后拦截面不缩水 + 新语义(缺省跟随/显式报错/gate-first)」。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from studio.domain.config_rules import (
    ADVISORY_DISABLE_FIELDS,
    apply_disable_rule_fixes,
    disable_rule_violations,
    iter_forbid_rules,
    iter_pin_rules,
)
from studio.schema import TrainingConfig


# ---------------------------------------------------------------------------
# 构造期 setdefault:缺省跟随钉值
# ---------------------------------------------------------------------------


def test_pin_setdefault_fills_missing_targets() -> None:
    cfg = TrainingConfig(navit_packing=True)
    assert cfg.attention_backend == "xformers"
    assert cfg.cache_latents is True


def test_pin_setdefault_does_not_touch_when_gate_off() -> None:
    cfg = TrainingConfig()
    assert cfg.attention_backend == "flash_attn"  # schema 默认,不被 setdefault


def test_explicit_violation_fails_fast_not_coerced() -> None:
    """显式提供且违反 → 报错(取代历史 _coerce_navit_attention_backend 的
    无差别静默改写;用户显式配置绝不静默改)。"""
    with pytest.raises(ValidationError, match="attention_backend"):
        TrainingConfig(navit_packing=True, attention_backend="flash_attn")
    with pytest.raises(ValidationError, match="cache_latents"):
        TrainingConfig(navit_packing=True, cache_latents=False)


# ---------------------------------------------------------------------------
# 退役 validator 的拦截面不缩水(逐对锁死)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overrides", [
    {"optimizer_type": "prodigy", "lr_scheduler": "cosine"},
    {"optimizer_type": "soap_sf", "lr_scheduler": "cosine"},
    {"infonoise_enabled": True, "loss_weighting": "min_snr"},
    {"infonoise_enabled": True, "timestep_schedule_shift": 2.0},
    {"infonoise_enabled": True, "loss_type": "huber"},
    {"infonoise_enabled": True, "noise_enhancement_type": "offset"},
    {"leap_enabled": True, "infonoise_enabled": True},
    {"leap_enabled": True, "loss_weighting": "min_snr"},
    {"leap_enabled": True, "loss_type": "huber"},
    {"navit_packing": True, "leap_enabled": True},
    {"navit_packing": True, "infonoise_enabled": True},
    {"navit_packing": True, "sra_enabled": True},
    {"navit_packing": True, "masked_loss": True},
    {"leap_enabled": True, "masked_loss": True},
])
def test_mutex_pairs_still_rejected(overrides) -> None:
    with pytest.raises(ValidationError, match="字段联动约束"):
        TrainingConfig(**overrides)


def test_forbid_navit_tlora_rejected() -> None:
    """option_disable_when(禁值)经同一 validator 强制。"""
    with pytest.raises(ValidationError, match="lora_type"):
        TrainingConfig(navit_packing=True, lora_type="tlora")


# ---------------------------------------------------------------------------
# 审计补录规则(设计文档 §10.1 #1 / #6)
# ---------------------------------------------------------------------------


def test_leap_sra_now_rejected() -> None:
    """审计 #1:leap 步整段跳过 SRA(leap_ratio=1.0 时 100% 静默零生效),
    刀 2 前配置层完全没拦。"""
    with pytest.raises(ValidationError, match="sra_enabled"):
        TrainingConfig(leap_enabled=True, sra_enabled=True)


def test_automagic_v2_grad_clip_pinned_zero() -> None:
    """审计 #6:automagic v2 fused backward 下 grad_clip 静默失效且默认 1.0
    人人踩中 —— 缺省自动落 0,显式非零报错。"""
    cfg = TrainingConfig(optimizer_type="automagic", automagic_variant="v2")
    assert cfg.grad_clip_max_norm == 0.0
    with pytest.raises(ValidationError, match="grad_clip_max_norm"):
        TrainingConfig(
            optimizer_type="automagic", automagic_variant="v2",
            grad_clip_max_norm=1.0,
        )
    # v1 不受影响
    assert TrainingConfig(optimizer_type="automagic").grad_clip_max_norm == 1.0


# ---------------------------------------------------------------------------
# ADVISORY:软钉值不强制
# ---------------------------------------------------------------------------


def test_advisory_learning_rate_not_enforced() -> None:
    """learning_rate 对 Prodigy 是真实生效的缩放因子,UI 钉 1.0 只是推荐 ——
    显式偏离不报错(ADVISORY_DISABLE_FIELDS)。"""
    assert "learning_rate" in ADVISORY_DISABLE_FIELDS
    cfg = TrainingConfig(optimizer_type="prodigy", learning_rate=0.5)
    assert cfg.learning_rate == 0.5
    assert all(name != "learning_rate" for name, *_ in iter_pin_rules(TrainingConfig))


# ---------------------------------------------------------------------------
# tolerant 修复:gate-first + 单步收敛
# ---------------------------------------------------------------------------


def test_fix_gate_first_preserves_user_investment() -> None:
    """InfoNoise 冲突优先关 InfoNoise,保住用户在对侧字段的投入
    (历史 InfoNoise 专用垫片的泛化)。"""
    data = {"infonoise_enabled": True, "loss_type": "huber", "loss_weighting": "min_snr"}
    fixed, fields = apply_disable_rule_fixes(data, TrainingConfig)
    assert fixed["infonoise_enabled"] is False
    assert fixed["loss_type"] == "huber"       # 保住
    assert fixed["loss_weighting"] == "min_snr"  # 保住
    assert fields == ["infonoise_enabled"]


def test_fix_pins_target_when_no_gate() -> None:
    """无 gate-first 开关参与时钉目标字段(用户显式选了 optimizer)。"""
    data = {"optimizer_type": "prodigy", "lr_scheduler": "cosine"}
    fixed, fields = apply_disable_rule_fixes(data, TrainingConfig)
    assert fixed["optimizer_type"] == "prodigy"
    assert fixed["lr_scheduler"] == "none"
    assert fields == ["lr_scheduler"]


def test_fix_noop_on_valid_data() -> None:
    fixed, fields = apply_disable_rule_fixes({"epochs": 5}, TrainingConfig)
    assert fixed == {"epochs": 5}
    assert fields == []


def test_tolerant_validate_uses_rule_fixes() -> None:
    """_tolerant_validate 全链:冲突 preset 修复后可用,修复字段进 defaulted。"""
    from studio.services.presets.io import _tolerant_validate

    cfg, dropped, defaulted = _tolerant_validate({
        "infonoise_enabled": True, "loss_weighting": "detail_inv_t",
    })
    assert cfg.infonoise_enabled is False
    assert cfg.loss_weighting == "detail_inv_t"
    assert "infonoise_enabled" in defaulted
    assert dropped == []


# ---------------------------------------------------------------------------
# 声明完备性
# ---------------------------------------------------------------------------


def test_all_rules_have_hints() -> None:
    """每条强制规则必须带 disable_hint —— 报错文案与 UI 徽章的领域理由来源。"""
    for name, _expr, _pin, hint in iter_pin_rules(TrainingConfig):
        assert hint, f"{name} 的 disable_when 缺 disable_hint"
    for name, value, _expr, hint in iter_forbid_rules(TrainingConfig):
        assert hint, f"{name} 的 option_disable_when[{value}] 缺 disable_hint"


def test_default_config_has_no_violations() -> None:
    """默认组合恒合法(tolerant 缺键不参与判定的前提,见 apply_disable_rule_fixes)。"""
    dumped = TrainingConfig().model_dump(mode="python")
    assert disable_rule_violations(dumped, TrainingConfig) == []
