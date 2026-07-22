"""schema.attention_backend 字段回归。"""
from __future__ import annotations

import pytest

from studio.schema import (
    AttentionBackend,  # noqa: F401  re-export 烟雾测试
    GenerateConfig,
    RegAiConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# pydantic schema
# ---------------------------------------------------------------------------


def test_generate_config_default_is_flash_attn() -> None:
    """无字段 → 默认 flash_attn（与历史 flash_attn=True 默认一致）。"""
    g = GenerateConfig(transformer_path="", vae_path="", text_encoder_path="")
    assert g.attention_backend == "flash_attn"
    assert g.lora_merge_precision == "fp32"


def test_generate_config_accepts_bf16_lora_merge_precision() -> None:
    g = GenerateConfig(
        transformer_path="", vae_path="", text_encoder_path="",
        lora_merge_precision="bf16",
    )
    assert g.lora_merge_precision == "bf16"


def test_generate_config_rejects_invalid_lora_merge_precision() -> None:
    with pytest.raises(ValueError):
        GenerateConfig(
            transformer_path="", vae_path="", text_encoder_path="",
            lora_merge_precision="fp16",
        )


def test_generate_config_new_field() -> None:
    g = GenerateConfig(transformer_path="", vae_path="", text_encoder_path="",
                       attention_backend="xformers")
    assert g.attention_backend == "xformers"


@pytest.mark.parametrize("backend", ["none", "xformers", "flash_attn"])
def test_all_backends_validate(backend: str) -> None:
    """三个枚举值都能 validate。"""
    g = GenerateConfig(transformer_path="", vae_path="", text_encoder_path="",
                       attention_backend=backend)  # type: ignore[arg-type]
    assert g.attention_backend == backend


def test_reg_ai_config_default_backend() -> None:
    r = RegAiConfig()
    assert r.attention_backend == "flash_attn"


def test_training_config_default_backend() -> None:
    t = TrainingConfig()
    assert t.attention_backend == "flash_attn"


def test_invalid_backend_rejected() -> None:
    """无效 backend → ValidationError。"""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GenerateConfig(transformer_path="", vae_path="", text_encoder_path="",
                       attention_backend="sdpa")  # type: ignore[arg-type]
