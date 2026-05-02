"""Stage 6 — schema 扩展字段验证：新算法选项 + DoRA/dropout/rs_lora 字段。"""
from __future__ import annotations

from studio.schema import TrainingConfig


def test_lora_type_accepts_loha():
    """新增 'loha' 算法支持"""
    cfg = TrainingConfig(lora_type="loha")
    assert cfg.lora_type == "loha"


def test_lora_type_still_accepts_legacy_values():
    """旧 yaml 配置（lora/lokr）加载零回归"""
    assert TrainingConfig(lora_type="lora").lora_type == "lora"
    assert TrainingConfig(lora_type="lokr").lora_type == "lokr"


def test_lora_type_rejects_unknown():
    """非法算法名仍然拒绝"""
    import pytest
    with pytest.raises(Exception):
        TrainingConfig(lora_type="random_string")


def test_dora_field_default_off():
    """DoRA 默认关闭，避免给老用户行为漂移"""
    assert TrainingConfig().lora_dora is False


def test_rs_lora_field_default_off():
    assert TrainingConfig().lora_rs is False


def test_dropout_fields_default_zero():
    cfg = TrainingConfig()
    assert cfg.lora_dropout == 0.0
    assert cfg.lora_rank_dropout == 0.0
    assert cfg.lora_module_dropout == 0.0


def test_dropout_fields_clamped_to_unit_range():
    """[0, 1] 范围验证"""
    import pytest
    with pytest.raises(Exception):
        TrainingConfig(lora_dropout=1.5)
    with pytest.raises(Exception):
        TrainingConfig(lora_rank_dropout=-0.1)


def test_legacy_yaml_config_loads_with_only_old_fields():
    """旧 yaml 缺新字段，所有 lora_* 新字段走默认值"""
    legacy = {
        "lora_type": "lokr",
        "lora_rank": 32,
        "lora_alpha": 32.0,
        "lokr_factor": 8,
    }
    cfg = TrainingConfig.model_validate(legacy)
    assert cfg.lora_dora is False
    assert cfg.lora_rs is False
    assert cfg.lora_dropout == 0.0


def test_new_fields_in_lora_group():
    """所有新字段归入 'lora' UI 分组"""
    schema = TrainingConfig.model_json_schema()
    props = schema["properties"]
    for f in ("lora_dora", "lora_rs", "lora_dropout", "lora_rank_dropout", "lora_module_dropout"):
        assert f in props, f"字段缺失: {f}"
        assert props[f].get("group") == "lora", f"{f} 不在 lora 分组"
