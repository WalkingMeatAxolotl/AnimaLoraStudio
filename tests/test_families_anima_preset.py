"""families/anima/preset.py — ANIMA_PRESET 内容稳定 + preset 注入契约（多模型 PR-2b）。

自 tests/test_lokr_preset.py 迁入：preset 从 utils/lokr_preset.py 移居
families/anima/，apply() helper 退役改为构造期显式注入。
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from training.families.anima import ANIMA_SPEC
from training.families.anima.preset import ANIMA_PRESET


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


def test_preset_prefix_matches_spec():
    """lora_prefix=lora_unet 是 ComfyUI 加载流程兼容的关键，且必须与 ModelSpec 一致"""
    assert ANIMA_PRESET["lora_prefix"] == "lora_unet"
    assert ANIMA_PRESET["lora_prefix"] == ANIMA_SPEC.lora.prefix


def test_lycoris_adapter_requires_explicit_preset():
    """族知识不再内置：不传 preset 的 inject 必须 fail-fast，防静默打错层"""
    from utils.lycoris_adapter import LycorisAdapter

    adapter = LycorisAdapter(algo="lokr", rank=4, alpha=4.0, factor=8)
    with pytest.raises(ValueError):
        adapter.inject(nn.Linear(4, 4))


def test_lycoris_adapter_alias_kept():
    from utils.lycoris_adapter import AnimaLycorisAdapter, LycorisAdapter

    assert AnimaLycorisAdapter is LycorisAdapter
