"""Krea2 LoRA preset target and Comfy/kohya key contract."""

from __future__ import annotations

from fnmatch import fnmatch
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from modeling.krea2 import SingleStreamDiT
from training.families.krea2.preset import KREA2_PRESET


def _targeted_linear_names() -> list[str]:
    with torch.device("meta"):
        model = SingleStreamDiT()
    patterns = KREA2_PRESET["target_name"]
    excluded = KREA2_PRESET["exclude_name"]
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and any(fnmatch(name, pattern) for pattern in patterns)
        and not any(fnmatch(name, pattern) for pattern in excluded)
    ]


def test_krea2_preset_targets_all_264_linear_modules() -> None:
    targets = _targeted_linear_names()

    assert len(targets) == 264
    assert len(targets) == len(set(targets))
    assert {
        "first",
        "blocks.0.attn.wq",
        "blocks.0.attn.gate",
        "blocks.0.mlp.up",
        "txtfusion.refiner_blocks.0.attn.wo",
        "txtmlp.1",
        "last.linear",
        "tproj.1",
    } <= set(targets)


def test_krea2_preset_matches_musubi_and_comfy_key_contract() -> None:
    assert KREA2_PRESET["enable_conv"] is False
    assert KREA2_PRESET["target_module"] == []
    assert KREA2_PRESET["target_name"] == ["*"]
    assert KREA2_PRESET["exclude_name"] == []
    assert KREA2_PRESET["use_fnmatch"] is True
    assert KREA2_PRESET["lora_prefix"] == "lora_unet"

    flattened = {
        f'{KREA2_PRESET["lora_prefix"]}_{name.replace(".", "_")}'
        for name in _targeted_linear_names()
    }
    assert "lora_unet_blocks_0_attn_wq" in flattened
    assert "lora_unet_blocks_0_attn_gate" in flattened
    assert "lora_unet_txtfusion_refiner_blocks_0_attn_wo" in flattened
    assert "lora_unet_txtmlp_1" in flattened


def test_krea2_preset_injects_264_lycoris_modules_on_meta() -> None:
    pytest.importorskip("lycoris")
    from utils.lycoris_adapter import LycorisAdapter

    with torch.device("meta"):
        model = SingleStreamDiT()
        adapter = LycorisAdapter(
            preset=KREA2_PRESET,
            algo="lora",
            rank=2,
            alpha=2,
        )
        injected = adapter.inject(model)

    assert len(injected) == 264
    assert {
        "lora_unet_blocks_0_attn_wq",
        "lora_unet_blocks_0_attn_gate",
        "lora_unet_txtfusion_refiner_blocks_0_attn_wo",
        "lora_unet_txtmlp_1",
    } <= set(injected)
    assert all(parameter.device.type == "meta" for parameter in adapter.get_params())


def test_lycoris_save_uses_family_metadata(tmp_path: Path) -> None:
    pytest.importorskip("lycoris")
    from utils.lycoris_adapter import LycorisAdapter

    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    adapter = LycorisAdapter(
        preset=KREA2_PRESET,
        algo="lora",
        rank=2,
        alpha=2,
    )
    adapter.metadata_extra = {
        "model_family": "krea2",
        "preset": "krea2_full",
    }
    adapter.inject(model)
    checkpoint = tmp_path / "krea2-lora.safetensors"
    adapter.save(checkpoint)

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        args = json.loads(handle.metadata()["ss_network_args"])
    assert args["model_family"] == "krea2"
    assert args["preset"] == "krea2_full"
