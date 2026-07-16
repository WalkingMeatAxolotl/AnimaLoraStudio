from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from training.families.anima.preset import ANIMA_PRESET


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = REPO_ROOT / "runtime"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_DIR))


class MockDiT(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.output_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.v_proj(self.k_proj(self.q_proj(x))))


def test_ortho_adapter_injects_and_saves_plain_lora(tmp_path: Path) -> None:
    from utils.ortho_adapter import OrthoLoRAAdapter

    model = MockDiT()
    adapter = OrthoLoRAAdapter(preset=ANIMA_PRESET, rank=4, alpha=4)
    injected = adapter.inject(model)

    assert len(injected) == 4
    assert len(adapter.get_params()) == 12  # S_p/S_q/lambda for each layer

    x = torch.randn(2, 3, 16)
    out = model(x)
    assert out.shape == x.shape

    path = tmp_path / "ortho.safetensors"
    adapter.metadata_extra = {
        "model_family": "krea2",
        "preset": "krea2_full",
    }
    adapter.save(path)
    sd = load_file(str(path))

    assert "lora_unet_q_proj.lora_down.weight" in sd
    assert "lora_unet_q_proj.lora_up.weight" in sd
    assert not any(key.endswith(".S_p") or key.endswith(".S_q") for key in sd)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        args = json.loads(handle.metadata()["ss_network_args"])
    assert args["model_family"] == "krea2"
    assert args["preset"] == "krea2_full"


def test_ortho_distilled_lora_matches_runtime_delta() -> None:
    from utils.ortho_adapter import OrthoLoRALinear

    torch.manual_seed(7)
    base = nn.Linear(8, 8, bias=False)
    layer = OrthoLoRALinear("lora_unet_test", base, rank=4, alpha=2.0)
    layer.eval()

    with torch.no_grad():
        layer.S_p.normal_(std=0.05)
        layer.S_q.normal_(std=0.05)
        layer.lambda_layer.normal_(std=0.1)

    x = torch.randn(2, 3, 8)
    runtime_delta = layer(x) - layer.org_module(x)

    sd = layer.distilled_lora_state(dtype=torch.float32)
    down = sd["lora_unet_test.lora_down.weight"]
    up = sd["lora_unet_test.lora_up.weight"]
    alpha = float(sd["lora_unet_test.alpha"])
    baked_delta = F.linear(F.linear(x, down), up) * (alpha / down.shape[0])

    assert torch.allclose(runtime_delta, baked_delta, atol=1e-5, rtol=1e-5)


def test_ortho_tlora_mask_matches_source_formula() -> None:
    from utils.ortho_adapter import OrthoLoRAAdapter

    model = MockDiT(d=8)
    adapter = OrthoLoRAAdapter(preset=ANIMA_PRESET, 
        rank=8,
        alpha=8,
        use_timestep_mask=True,
        tlora_min_rank=2,
        tlora_alpha_rank_scale=1.0,
    )
    adapter.inject(model)
    first_layer = next(iter(adapter.loras))

    adapter._set_timestep_mask(torch.tensor([0.0]))
    assert first_layer._timestep_mask.tolist() == [[1.0] * 8]

    adapter._set_timestep_mask(torch.tensor([0.5]))
    assert first_layer._timestep_mask.tolist() == [[1.0] * 5 + [0.0] * 3]

    adapter._set_timestep_mask(torch.tensor([1.0]))
    assert first_layer._timestep_mask.tolist() == [[1.0, 1.0] + [0.0] * 6]


def test_tlora_builder_defaults_to_plain_lycoris_tlora() -> None:
    pytest.importorskip("lycoris")
    from training.adapters import build_adapter
    from utils.lycoris_adapter import AnimaLycorisAdapter

    args = argparse.Namespace(
        lora_type="tlora",
        lora_rank=4,
        lora_alpha=4,
        lokr_factor=8,
        lora_dropout=0.0,
        lora_rank_dropout=0.0,
        lora_module_dropout=0.0,
        lora_dora=False,
        lora_rs=False,
        lora_reg_dims=None,
        tlora_min_rank=1,
        tlora_alpha_rank_scale=1.0,
        tlora_use_ortho=False,
    )
    adapter = build_adapter(args, preset=ANIMA_PRESET)

    assert isinstance(adapter, AnimaLycorisAdapter)
    assert adapter.algo == "tlora"


def test_tlora_builder_uses_ortho_timestep_mask_when_enabled() -> None:
    from training.adapters import build_adapter
    from training.adapters.protocol import StepContext

    args = argparse.Namespace(
        lora_type="tlora",
        lora_rank=4,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_rank_dropout=0.0,
        lora_module_dropout=0.0,
        tlora_min_rank=1,
        tlora_alpha_rank_scale=1.0,
        tlora_use_ortho=True,
    )
    model = MockDiT()
    adapter = build_adapter(args, preset=ANIMA_PRESET)
    adapter.inject(model)

    adapter.on_step_begin(StepContext(0, 10, 0, torch.tensor([1.0]), args))
    first_layer = next(iter(adapter.loras))
    assert first_layer._timestep_mask.tolist() == [[1.0, 0.0, 0.0, 0.0]]
