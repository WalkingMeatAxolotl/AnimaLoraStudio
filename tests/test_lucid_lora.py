from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

import torch
import torch.nn as nn
from safetensors import safe_open

from utils.lucid_lora import AnimaLucidLoRAAdapter, LucidLoKrLinear, LucidLoRALinear


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.output_proj = nn.Linear(8, 8, bias=False)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(8, 8, bias=False)
        self.layer2 = nn.Linear(8, 8, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.cross_attn = _Attention()
        self.mlp = _Mlp()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block()])


def test_lucid_role_split_and_param_groups() -> None:
    model = _Model()
    adapter = AnimaLucidLoRAAdapter(rank=8, qk_rank_ratio=0.25, min_rank_ratio=0.25, sig_type="random")
    adapter.inject(model)

    assert isinstance(model.blocks[0].self_attn.q_proj, LucidLoRALinear)
    assert model.blocks[0].self_attn.q_proj.rank == 2
    assert model.blocks[0].self_attn.k_proj.rank == 2
    assert model.blocks[0].self_attn.v_proj.rank == 8
    assert model.blocks[0].mlp.layer1.rank == 8

    groups = adapter.get_param_groups(weight_decay=0.01)
    assert len(groups) == 2
    assert len(groups[0]["params"]) == 10
    assert len(groups[1]["params"]) == 10
    assert groups[1]["_lucid_lora_plus_group"] is True


def test_lucid_optional_ffn_lokr_replaces_mlp_layers() -> None:
    model = _Model()
    adapter = AnimaLucidLoRAAdapter(
        rank=8,
        qk_rank_ratio=0.25,
        use_lokr_ffn=True,
        lokr_factor=2,
        sig_type="random",
    )
    adapter.inject(model)

    assert isinstance(model.blocks[0].self_attn.q_proj, LucidLoRALinear)
    assert isinstance(model.blocks[0].mlp.layer1, LucidLoKrLinear)
    assert isinstance(model.blocks[0].mlp.layer2, LucidLoKrLinear)

    groups = adapter.get_param_groups(weight_decay=0.01)
    assert len(groups) == 3
    assert len(groups[0]["params"]) == 8
    assert len(groups[1]["params"]) == 8
    assert len(groups[2]["params"]) == 4

    sd = adapter.state_dict()
    assert any(key.endswith("mlp_layer1.lokr_w1") for key in sd)
    assert any(key.endswith("mlp_layer1.lokr_w2_b") for key in sd)


def test_lucid_mask_scales_to_layer_rank() -> None:
    model = _Model()
    adapter = AnimaLucidLoRAAdapter(rank=8, qk_rank_ratio=0.25, min_rank_ratio=0.25, sig_type="random")
    adapter.inject(model)

    adapter.set_mask(torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32))

    q_mask = model.blocks[0].self_attn.q_proj.current_mask
    v_mask = model.blocks[0].self_attn.v_proj.current_mask
    assert q_mask is not None
    assert v_mask is not None
    assert q_mask.shape == (1, 2)
    assert v_mask.shape == (1, 8)
    assert int(q_mask.sum().item()) == 1
    assert int(v_mask.sum().item()) == 4


def test_lucid_multiplier_scales_delta() -> None:
    model = _Model()
    adapter = AnimaLucidLoRAAdapter(rank=8, qk_rank_ratio=0.25, sig_type="random")
    adapter.inject(model)
    layer = model.blocks[0].self_attn.v_proj
    assert isinstance(layer, LucidLoRALinear)
    layer.down.weight.data.fill_(1.0)
    layer.up.weight.data.fill_(1.0)

    x = torch.ones(1, 8)
    original = layer.original(x)
    adapter.multiplier = 0.5
    adapter.set_mask(torch.ones(1, 8))

    assert torch.allclose(layer(x) - original, torch.full_like(original, 16.0))


def test_lucid_load_crops_legacy_full_rank_qk_weights() -> None:
    model = _Model()
    adapter = AnimaLucidLoRAAdapter(rank=8, qk_rank_ratio=0.25, sig_type="random")
    adapter.inject(model)

    sd = {}
    for key, layer in adapter._layer_keys.items():
        if isinstance(layer, LucidLoRALinear):
            sd[f"{key}.lora_down.weight"] = torch.ones(8, 8)
            sd[f"{key}.lora_up.weight"] = torch.ones(8, 8)
    result = adapter.load_state_dict(sd, strict=False)

    q_layer = model.blocks[0].self_attn.q_proj
    assert isinstance(q_layer, LucidLoRALinear)
    assert q_layer.down.weight.shape == (2, 8)
    assert q_layer.up.weight.shape == (8, 2)
    assert torch.all(q_layer.down.weight == 1)
    assert torch.all(q_layer.up.weight == 1)
    assert result.unexpected_keys == []


def test_lucid_save_writes_compat_metadata(tmp_path) -> None:
    model = _Model()
    adapter = AnimaLucidLoRAAdapter(rank=8, qk_rank_ratio=0.25, sig_type="random")
    adapter.inject(model)

    path = tmp_path / "lucid.safetensors"
    adapter.save(path)

    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata()
    assert meta["ss_network_module"] == "lycoris.kohya"
    assert meta["ss_network_alpha"] == "8.0"
    args = json.loads(meta["ss_network_args"])
    assert args["algo"] == "lora"
    assert args["training_alpha"] == 16.0
    assert args["lucid_algo"] == "lucid"
    assert "base" not in args
    assert args["compat"] == "lycoris_compat"


def test_lucid_native_state_keeps_role_split_but_compat_export_pads_qk(tmp_path) -> None:
    model = _Model()
    adapter = AnimaLucidLoRAAdapter(rank=8, qk_rank_ratio=0.25, sig_type="random")
    adapter.inject(model)

    native = adapter.state_dict()
    assert native["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"].shape == (2, 8)
    assert native["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"].shape == (8, 2)
    assert native["lora_unet_blocks_0_self_attn_v_proj.lora_down.weight"].shape == (8, 8)
    assert native["lora_unet_blocks_0_self_attn_v_proj.lora_up.weight"].shape == (8, 8)

    compat = adapter.export_state_dict()
    assert compat["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"].shape == (8, 8)
    assert compat["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"].shape == (8, 8)
    assert torch.all(compat["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"][2:] == 0)
    assert torch.all(compat["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"][:, 2:] == 0)

    path = tmp_path / "lucid_compat.safetensors"
    adapter.save(path)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        assert f.get_tensor("lora_unet_blocks_0_self_attn_q_proj.lora_down.weight").shape == (8, 8)
        assert f.get_tensor("lora_unet_blocks_0_self_attn_q_proj.lora_up.weight").shape == (8, 8)


def test_lucid_native_export_keeps_actual_qk_rank() -> None:
    model = _Model()
    adapter = AnimaLucidLoRAAdapter(rank=8, qk_rank_ratio=0.25, sig_type="random", export_mode="native")
    adapter.inject(model)

    exported = adapter.export_state_dict()
    assert exported["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"].shape == (2, 8)
    assert exported["lora_unet_blocks_0_self_attn_q_proj.lora_up.weight"].shape == (8, 2)
