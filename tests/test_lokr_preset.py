"""utils/lokr_preset.py — 验证 ANIMA_PRESET 内容稳定 + apply() 调用契约"""
from __future__ import annotations

import torch.nn as nn

from utils.lokr_preset import ANIMA_PRESET, apply


class _Spy:
    """收集 apply_preset 调用，避免依赖真实 lycoris 类状态"""
    received: dict | None = None

    @classmethod
    def apply_preset(cls, preset: dict) -> None:
        cls.received = dict(preset)


def test_preset_disables_conv():
    assert ANIMA_PRESET["enable_conv"] is False


def test_preset_targets_attention_and_mlp():
    names = ANIMA_PRESET["target_name"]
    for needle in ("q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"):
        assert any(needle in p for p in names), f"missing target {needle}"


def test_preset_excludes_llm_adapter():
    assert any("llm_adapter" in p for p in ANIMA_PRESET["exclude_name"])


def test_preset_uses_fnmatch():
    assert ANIMA_PRESET["use_fnmatch"] is True


def test_preset_keeps_comfyui_prefix():
    """lora_prefix=lora_unet 是 ComfyUI 现有加载流程兼容的关键，不可改"""
    assert ANIMA_PRESET["lora_prefix"] == "lora_unet"


def test_apply_calls_apply_preset_with_dict():
    _Spy.received = None
    apply(_Spy)
    assert _Spy.received is not None
    assert _Spy.received["lora_prefix"] == "lora_unet"
    assert _Spy.received["enable_conv"] is False
