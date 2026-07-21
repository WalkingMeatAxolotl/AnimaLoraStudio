"""Krea2 modeling structure, forward contract, and ComfyUI key compatibility."""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

import modeling.krea2.krea2_modeling as krea2_modeling
from modeling.krea2 import KREA2_CONFIG, Attention, Krea2Config, SingleStreamDiT


def _tiny_config() -> Krea2Config:
    return Krea2Config(
        features=64,
        tdim=16,
        txtdim=32,
        heads=4,
        kvheads=2,
        multiplier=2,
        layers=2,
        patch=2,
        channels=4,
        txtlayers=3,
        txtheads=4,
        txtkvheads=2,
    )


def _tiny_inputs():
    x = torch.randn(2, 4, 1, 5, 7)
    timesteps = torch.tensor([0.2, 0.8])
    context = torch.randn(2, 5, 3, 32)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    return x, timesteps, context, mask


def test_public_config_matches_krea2_checkpoint_architecture() -> None:
    assert KREA2_CONFIG == Krea2Config()
    assert KREA2_CONFIG.features == 6144
    assert KREA2_CONFIG.heads == 48
    assert KREA2_CONFIG.kvheads == 12
    assert KREA2_CONFIG.layers == 28
    assert KREA2_CONFIG.txtlayers == 12
    assert KREA2_CONFIG.patch == 2
    assert KREA2_CONFIG.channels == 16


def test_full_meta_model_has_expected_size_linear_count_and_gqa_shapes() -> None:
    with torch.device("meta"):
        model = SingleStreamDiT()

    assert sum(parameter.numel() for parameter in model.parameters()) == 12_820_073_036
    assert sum(isinstance(module, torch.nn.Linear) for module in model.modules()) == 264
    state = model.state_dict()
    assert len(state) == 430
    assert state["blocks.0.attn.wq.weight"].shape == (6144, 6144)
    assert state["blocks.0.attn.wk.weight"].shape == (1536, 6144)
    assert state["blocks.0.attn.wv.weight"].shape == (1536, 6144)
    assert state["blocks.0.attn.gate.weight"].shape == (6144, 6144)


def test_parameter_paths_flatten_to_comfyui_kohya_lora_keys() -> None:
    with torch.device("meta"):
        state_keys = set(SingleStreamDiT().state_dict())

    parameter_paths = [
        "blocks.0.attn.wq.weight",
        "blocks.0.attn.gate.weight",
        "blocks.0.mlp.up.weight",
        "txtfusion.refiner_blocks.0.attn.wo.weight",
        "txtmlp.1.weight",
    ]
    for path in parameter_paths:
        assert path in state_keys

    flattened = {
        "lora_unet_" + path.removesuffix(".weight").replace(".", "_")
        for path in parameter_paths
    }
    assert flattened == {
        "lora_unet_blocks_0_attn_wq",
        "lora_unet_blocks_0_attn_gate",
        "lora_unet_blocks_0_mlp_up",
        "lora_unet_txtfusion_refiner_blocks_0_attn_wo",
        "lora_unet_txtmlp_1",
    }


def test_tiny_forward_preserves_5d_shape_and_crops_patch_padding() -> None:
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, mask = _tiny_inputs()
    with torch.no_grad():
        output = model(x, timesteps, context, mask)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_batched_forward_matches_per_sample_forward() -> None:
    """Modulation is broadcast per sample, never across the token axis.

    ``tproj`` emits (B, 1, 6*features); a missing token axis would broadcast
    (B, features) against (B, L, features) and silently pass only when B == 1.
    """
    torch.manual_seed(11)
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, mask = _tiny_inputs()
    with torch.no_grad():
        batched = model(x, timesteps, context, mask)
        per_sample = torch.cat(
            [
                model(
                    x[i : i + 1],
                    timesteps[i : i + 1],
                    context[i : i + 1],
                    mask[i : i + 1],
                )
                for i in range(x.shape[0])
            ]
        )
    torch.testing.assert_close(batched, per_sample, rtol=1e-5, atol=1e-5)


def test_flattened_and_layered_text_context_are_equivalent() -> None:
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, mask = _tiny_inputs()
    flattened = context.flatten(2)
    with torch.no_grad():
        layered_out = model(x, timesteps, context, mask)
        flattened_out = model(x, timesteps, flattened, mask)
    torch.testing.assert_close(layered_out, flattened_out)


def test_padding_mask_prevents_padded_text_from_affecting_image_output() -> None:
    torch.manual_seed(7)
    model = SingleStreamDiT(_tiny_config()).eval()
    for block in model.blocks:
        block.mod.lin.data.fill_(0.1)

    x, timesteps, context, mask = _tiny_inputs()
    changed = context.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1000
    with torch.no_grad():
        original_out = model(x, timesteps, context, mask)
        changed_out = model(x, timesteps, changed, mask)
    torch.testing.assert_close(original_out, changed_out, rtol=1e-5, atol=1e-5)


def test_gradient_checkpointing_path_backpropagates() -> None:
    model = SingleStreamDiT(_tiny_config()).train()
    model.enable_gradient_checkpointing()
    x, timesteps, context, mask = _tiny_inputs()
    x.requires_grad_(True)
    output = model(x, timesteps, context, mask)
    output.square().mean().backward()
    assert x.grad is not None
    assert model.first.weight.grad is not None
    assert model.last.linear.weight.grad is not None
    model.disable_gradient_checkpointing()
    assert model.gradient_checkpointing is False


def test_4d_latent_path_is_supported() -> None:
    model = SingleStreamDiT(_tiny_config()).eval()
    x, timesteps, context, mask = _tiny_inputs()
    with torch.no_grad():
        output = model(x.squeeze(2), timesteps, context, mask)
    assert output.shape == (2, 4, 5, 7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"features": 63, "heads": 4},
        {"heads": 6, "kvheads": 4},
        {"txtdim": 31, "txtheads": 4},
        {"txtheads": 6, "txtkvheads": 4},
        {"tdim": 15},
        {"theta": 0},
    ],
)
def test_config_rejects_invalid_divisibility(kwargs) -> None:
    with pytest.raises(ValueError):
        Krea2Config(**kwargs)


def test_forward_rejects_invalid_temporal_and_conditioning_shapes() -> None:
    model = SingleStreamDiT(_tiny_config())
    _, timesteps, context, mask = _tiny_inputs()
    with pytest.raises(ValueError, match="T==1"):
        model(torch.randn(2, 4, 2, 4, 4), timesteps, context, mask)
    with pytest.raises(ValueError, match="context"):
        model(torch.randn(2, 4, 4, 4), timesteps, torch.randn(2, 5, 95), mask)
    with pytest.raises(ValueError, match="attention_mask"):
        model(
            torch.randn(2, 4, 4, 4),
            timesteps,
            context,
            torch.ones(2, 4, dtype=torch.bool),
        )


def test_attention_gqa_preserves_shape() -> None:
    attention = Attention(dim=64, heads=4, kvheads=2)
    x = torch.randn(2, 7, 64)
    mask = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    assert attention(x, mask=mask).shape == x.shape


def test_modeling_layer_only_imports_torch_einops_and_stdlib() -> None:
    tree = ast.parse(inspect.getsource(krea2_modeling))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert roots <= {"__future__", "dataclasses", "einops", "math", "torch"}
