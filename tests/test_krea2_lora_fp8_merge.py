"""Krea2 fp8 底模 LoRA merge（ComfyUI merge 语义逐位复刻）。

CPU 上验证公式与生命周期；CUDA generator 的 RNG 序列 parity 属真卡验收。
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from training.families.krea2.lora_fp8_merge import (
    Fp8LoraMergeAdapter,
    _requantize_scaled,
    merge_loras_into_fp8_model,
    stochastic_round_to_fp8,
    string_to_seed,
)
from training.families.krea2.quant_fp8 import patch_fp8_linears


E4M3 = torch.float8_e4m3fn


def _comfy_string_to_seed(data: str) -> int:
    """comfy.utils.string_to_seed 手写参考实现（对拍 zlib 等价性）。"""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= ord(byte)
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF


def test_string_to_seed_matches_comfy_reference():
    for key in (
        "diffusion_model.blocks.0.attn.gate.weight",
        "diffusion_model.final_layer.linear.weight",
        "a",
        "",
    ):
        assert string_to_seed(key) == _comfy_string_to_seed(key)


def test_string_to_seed_rejects_non_ascii():
    with pytest.raises(ValueError, match="ASCII"):
        string_to_seed("层名")


# ---------------------------------------------------------------------------
# stochastic rounding
# ---------------------------------------------------------------------------


def test_stochastic_round_deterministic_and_seed_sensitive():
    value = torch.linspace(-5.0, 5.0, 64, dtype=torch.float16)
    a = stochastic_round_to_fp8(value, E4M3, seed=1234)
    b = stochastic_round_to_fp8(value, E4M3, seed=1234)
    c = stochastic_round_to_fp8(value, E4M3, seed=4321)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert not torch.equal(a.view(torch.uint8), c.view(torch.uint8))


def test_stochastic_round_zero_clamp_and_error_bound():
    value = torch.tensor([0.0, 448.0, 500.0, -500.0, 1.0, -3.14], dtype=torch.float16)
    out = stochastic_round_to_fp8(value, E4M3, seed=7).to(torch.float32)
    assert out[0].item() == 0.0
    assert out[1].item() == 448.0     # e4m3 max 保持
    assert out[2].item() == 448.0     # 超界 clamp
    assert out[3].item() == -448.0
    # normal 域随机舍入误差 ≤ 一个 mantissa 步长（相对 2^-3）
    ref = value[4:].to(torch.float32)
    got = out[4:]
    assert torch.all((got - ref).abs() / ref.abs() <= 2 ** -3 + 1e-3)


def test_requantize_scaled_recalculates_amax_over_448():
    w16 = torch.tensor([[2.0, -8.0], [1.0, 4.0]], dtype=torch.float16)
    qdata, scale = _requantize_scaled(w16.clone(), E4M3, seed=99)
    assert scale.dtype == torch.float32
    assert scale.item() == pytest.approx(8.0 / 448.0, rel=1e-6)
    # 归一后逐元素 |x| ≤ 448，dequant 回来近似原值
    back = qdata.to(torch.float32) * scale
    assert torch.all((back - w16.float()).abs() / w16.float().abs().clamp(min=1e-6) <= 0.13)


def test_requantize_scaled_fp16_underflow_clamp():
    # amax 极小 → 1/scale 超出 fp16 max → comfy 防下溢 clamp 生效
    w16 = torch.full((2, 2), 1e-7, dtype=torch.float16)
    _, scale = _requantize_scaled(w16.clone(), E4M3, seed=1)
    # clamp 后 1/scale 顶在 fp16 max → scale 精确落在 1/65504（float32 域）
    assert scale.item() == pytest.approx(1.0 / torch.finfo(torch.float16).max, rel=1e-6)


# ---------------------------------------------------------------------------
# merge 端到端（scaled fp8 + 纯 cast fp8 + 非量化 bf16 三形态）
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4, bias=False)
        self.k = nn.Linear(4, 4, bias=False)
        self.m = nn.Linear(4, 4, bias=False)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block()])


def _make_fp8_model() -> _Tiny:
    torch.manual_seed(0)
    model = _Tiny()
    with torch.no_grad():
        block = model.blocks[0]
        block.q.weight.data = (torch.randn(4, 4) * 4).to(E4M3)
        block.k.weight.data = (torch.randn(4, 4)).to(E4M3)
        block.m.weight.data = (torch.randn(4, 4)).to(torch.bfloat16)
    patch_fp8_linears(model, {
        "blocks.0.q": torch.tensor(0.02, dtype=torch.float32),
        "blocks.0.k": None,
    })
    return model


def _plain_lora_sd(layers: list[str], rank: int = 2, seed: int = 5) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for layer in layers:
        prefix = f"lora_unet_{layer.replace('.', '_')}"
        sd[f"{prefix}.lora_up.weight"] = torch.randn(4, rank, dtype=torch.float16)
        sd[f"{prefix}.lora_down.weight"] = torch.randn(rank, 4, dtype=torch.float16)
        sd[f"{prefix}.alpha"] = torch.tensor(float(rank))
    return sd


def test_merge_three_layer_forms_and_detach_restores():
    model = _make_fp8_model()
    layers = ["blocks.0.q", "blocks.0.k", "blocks.0.m"]
    sd = _plain_lora_sd(layers)
    block = model.blocks[0]

    orig = {
        "q": block.q.weight.detach().clone(),
        "k": block.k.weight.detach().clone(),
        "m": block.m.weight.detach().clone(),
        "q_scale": block.q.weight_scale.detach().clone(),
    }
    grouped = {
        layer: {
            suffix: sd[f"lora_unet_{layer.replace('.', '_')}.{suffix}"]
            for suffix in ("lora_up.weight", "lora_down.weight", "alpha")
        }
        for layer in layers
    }

    adapter = merge_loras_into_fp8_model(model, [(sd, 0.8, "test.safetensors")])
    assert isinstance(adapter, Fp8LoraMergeAdapter)
    assert adapter.network is None
    assert adapter.supports_hot_reload is False

    # scaled 层：recalculate scale + 同 seed SR 重放逐位一致
    w16_q = _expected_w16_from(orig["q"], orig["q_scale"], grouped["blocks.0.q"], 0.8)
    qdata_ref, scale_ref = _requantize_scaled(
        w16_q.clone(), E4M3, string_to_seed("diffusion_model.blocks.0.q.weight"),
    )
    assert torch.equal(block.q.weight.view(torch.uint8), qdata_ref.view(torch.uint8))
    assert torch.equal(block.q.weight_scale, scale_ref)

    # 纯 cast 层：无 scale 重算，直接 SR
    w16_k = _expected_w16_from(orig["k"], None, grouped["blocks.0.k"], 0.8)
    k_ref = stochastic_round_to_fp8(
        w16_k, E4M3, string_to_seed("diffusion_model.blocks.0.k.weight"),
    )
    assert torch.equal(block.k.weight.view(torch.uint8), k_ref.view(torch.uint8))

    # 非量化层：cast 回 bf16，无 SR
    w16_m = _expected_w16_from(orig["m"], None, grouped["blocks.0.m"], 0.8)
    assert torch.equal(block.m.weight.detach(), w16_m.to(torch.bfloat16))

    # detach 逐位还原三层 + scale
    assert adapter.detach() is True
    assert torch.equal(block.q.weight.view(torch.uint8), orig["q"].view(torch.uint8))
    assert torch.equal(block.k.weight.view(torch.uint8), orig["k"].view(torch.uint8))
    assert torch.equal(block.m.weight.detach(), orig["m"])
    assert torch.equal(block.q.weight_scale, orig["q_scale"])


def _expected_w16_from(weight, scale, tensors, strength) -> torch.Tensor:
    w16 = weight.to(torch.float16)
    if scale is not None:
        w16 = w16 * scale.to(torch.float16)
    up = tensors["lora_up.weight"].float()
    down = tensors["lora_down.weight"].float()
    alpha = float(tensors["alpha"]) / down.shape[0]
    w16 += ((strength * alpha) * torch.mm(up, down)).type(torch.float16)
    return w16


def test_merge_is_deterministic_across_remerge():
    model = _make_fp8_model()
    sd = _plain_lora_sd(["blocks.0.q"])
    adapter = merge_loras_into_fp8_model(model, [(sd, 1.0, "a")])
    first = model.blocks[0].q.weight.detach().view(torch.uint8).clone()
    first_scale = model.blocks[0].q.weight_scale.detach().clone()
    adapter.detach()
    merge_loras_into_fp8_model(model, [(sd, 1.0, "a")])
    assert torch.equal(model.blocks[0].q.weight.view(torch.uint8), first)
    assert torch.equal(model.blocks[0].q.weight_scale, first_scale)


def test_merge_multiple_loras_apply_in_mount_order():
    model = _make_fp8_model()
    sd_a = _plain_lora_sd(["blocks.0.m"], seed=11)
    sd_b = _plain_lora_sd(["blocks.0.m"], seed=22)
    block = model.blocks[0]
    orig_m = block.m.weight.detach().clone()

    merge_loras_into_fp8_model(model, [(sd_a, 0.5, "a"), (sd_b, -0.25, "b")])

    w16 = orig_m.to(torch.float16)
    for sd, strength in ((sd_a, 0.5), (sd_b, -0.25)):
        up = sd["lora_unet_blocks_0_m.lora_up.weight"].float()
        down = sd["lora_unet_blocks_0_m.lora_down.weight"].float()
        alpha = float(sd["lora_unet_blocks_0_m.alpha"]) / down.shape[0]
        w16 += ((strength * alpha) * torch.mm(up, down)).type(torch.float16)
    assert torch.equal(block.m.weight.detach(), w16.to(torch.bfloat16))


def test_merge_lokr_matches_comfy_kron_order():
    model = _make_fp8_model()
    block = model.blocks[0]
    orig_m = block.m.weight.detach().clone()
    torch.manual_seed(3)
    w1 = torch.randn(2, 2, dtype=torch.float16)
    w2_a = torch.randn(2, 2, dtype=torch.float16)
    w2_b = torch.randn(2, 2, dtype=torch.float16)
    sd = {
        "lora_unet_blocks_0_m.lokr_w1": w1,
        "lora_unet_blocks_0_m.lokr_w2_a": w2_a,
        "lora_unet_blocks_0_m.lokr_w2_b": w2_b,
        "lora_unet_blocks_0_m.alpha": torch.tensor(2.0),
    }

    merge_loras_into_fp8_model(model, [(sd, 1.0, "lokr")])

    w2 = torch.mm(w2_a.float(), w2_b.float())
    diff = torch.kron(w1.float(), w2).reshape(4, 4)
    alpha = 2.0 / w2_b.shape[0]  # dim = w2_b.shape[0]（comfy lokr.py 覆盖语义）
    expected = (orig_m.to(torch.float16) + ((1.0 * alpha) * diff).type(torch.float16))
    assert torch.equal(block.m.weight.detach(), expected.to(torch.bfloat16))


def test_merge_rejects_dora_t2_unknown_and_all_miss():
    model = _make_fp8_model()
    dora = _plain_lora_sd(["blocks.0.q"])
    dora["lora_unet_blocks_0_q.dora_scale"] = torch.ones(4, 1)
    with pytest.raises(ValueError, match="DoRA"):
        merge_loras_into_fp8_model(model, [(dora, 1.0, "d")])

    t2 = {
        "lora_unet_blocks_0_q.lokr_w1": torch.randn(2, 2),
        "lora_unet_blocks_0_q.lokr_w2_a": torch.randn(2, 2),
        "lora_unet_blocks_0_q.lokr_w2_b": torch.randn(2, 2),
        "lora_unet_blocks_0_q.lokr_t2": torch.randn(2, 2, 1, 1),
    }
    with pytest.raises(ValueError, match="t2"):
        merge_loras_into_fp8_model(model, [(t2, 1.0, "t")])

    unknown = {"lora_unet_blocks_0_q.mystery": torch.randn(2)}
    with pytest.raises(ValueError, match="无法识别"):
        merge_loras_into_fp8_model(model, [(unknown, 1.0, "u")])

    all_miss = _plain_lora_sd(["no.such.layer"])
    with pytest.raises(ValueError, match="无法对应"):
        merge_loras_into_fp8_model(model, [(all_miss, 1.0, "m")])


# ---------------------------------------------------------------------------
# apply_loras 集成（fp8 检测 → merge 路径）
# ---------------------------------------------------------------------------


def _write_lora_file(tmp_path, sd, *, network_args: dict) -> str:
    from safetensors.torch import save_file

    path = tmp_path / "krea2_lora.safetensors"
    metadata = {
        "ss_network_dim": "2",
        "ss_network_alpha": "2.0",
        "ss_network_args": json.dumps({"model_family": "krea2", **network_args}),
    }
    save_file(sd, str(path), metadata=metadata)
    return str(path)


def test_apply_loras_routes_fp8_model_to_merge(tmp_path):
    from studio.services.inference.core import LoRASpec, apply_loras

    model = _make_fp8_model()
    path = _write_lora_file(
        tmp_path, _plain_lora_sd(["blocks.0.q"]), network_args={"algo": "lora"},
    )
    before = model.blocks[0].q.weight.detach().view(torch.uint8).clone()

    adapters = apply_loras(
        model, [LoRASpec(path=path, scale=0.7)], "cpu", torch.float32,
        family_id="krea2",
    )

    assert len(adapters) == 1
    assert isinstance(adapters[0], Fp8LoraMergeAdapter)
    assert not torch.equal(model.blocks[0].q.weight.view(torch.uint8), before)
    assert adapters[0].detach() is True
    assert torch.equal(model.blocks[0].q.weight.view(torch.uint8), before)


def test_apply_loras_fp8_rejects_rs_lora_and_dora_meta(tmp_path):
    from studio.services.inference.core import LoRASpec, apply_loras

    model = _make_fp8_model()
    path = _write_lora_file(
        tmp_path, _plain_lora_sd(["blocks.0.q"]),
        network_args={"algo": "lora", "rs_lora": True},
    )
    with pytest.raises(ValueError, match="rs_lora / DoRA"):
        apply_loras(
            model, [LoRASpec(path=path, scale=1.0)], "cpu", torch.float32,
            family_id="krea2",
        )


# ---------------------------------------------------------------------------
# 外部生态文件（civitai / PEFT / comfy 键格式，无 ss_* metadata）
# ---------------------------------------------------------------------------


def _peft_lora_sd(layers: list[str], rank: int = 2, seed: int = 9,
                  with_alpha: bool = False) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for layer in layers:
        prefix = f"diffusion_model.{layer}"
        sd[f"{prefix}.lora_A.weight"] = torch.randn(rank, 4, dtype=torch.float16)
        sd[f"{prefix}.lora_B.weight"] = torch.randn(4, rank, dtype=torch.float16)
        if with_alpha:
            sd[f"{prefix}.alpha"] = torch.tensor(1.0)
    return sd


def test_merge_accepts_peft_comfy_key_format_no_alpha_means_scale_one():
    """civitai 形态：diffusion_model.{层}.lora_A/B、无 alpha → comfy 缩放 1.0。"""
    model = _make_fp8_model()
    block = model.blocks[0]
    orig_m = block.m.weight.detach().clone()
    sd = _peft_lora_sd(["blocks.0.m"])

    merge_loras_into_fp8_model(model, [(sd, 0.5, "civit.safetensors")])

    down = sd["diffusion_model.blocks.0.m.lora_A.weight"].float()
    up = sd["diffusion_model.blocks.0.m.lora_B.weight"].float()
    # 无 alpha 键 → alpha 系数 1.0（非 alpha/rank）
    expected = orig_m.to(torch.float16) + ((0.5 * 1.0) * torch.mm(up, down)).type(torch.float16)
    assert torch.equal(block.m.weight.detach(), expected.to(torch.bfloat16))


def test_merge_peft_with_alpha_key_uses_alpha_over_rank():
    model = _make_fp8_model()
    block = model.blocks[0]
    orig_m = block.m.weight.detach().clone()
    sd = _peft_lora_sd(["blocks.0.m"], with_alpha=True)

    merge_loras_into_fp8_model(model, [(sd, 1.0, "civit")])

    down = sd["diffusion_model.blocks.0.m.lora_A.weight"].float()
    up = sd["diffusion_model.blocks.0.m.lora_B.weight"].float()
    alpha = 1.0 / down.shape[0]
    expected = orig_m.to(torch.float16) + ((1.0 * alpha) * torch.mm(up, down)).type(torch.float16)
    assert torch.equal(block.m.weight.detach(), expected.to(torch.bfloat16))


def test_merge_rejects_unknown_comfy_suffix():
    model = _make_fp8_model()
    sd = {"diffusion_model.blocks.0.m.mystery.weight": torch.randn(2)}
    with pytest.raises(ValueError, match="后缀"):
        merge_loras_into_fp8_model(model, [(sd, 1.0, "u")])


def test_read_lora_meta_family_explicit_semantics(tmp_path):
    from safetensors.torch import save_file

    from studio.services.inference.core import read_lora_meta

    tagged = tmp_path / "tagged.safetensors"
    save_file({"x": torch.zeros(1)}, str(tagged), metadata={
        "ss_network_args": json.dumps({"algo": "lora", "model_family": "krea2"}),
    })
    untagged = tmp_path / "untagged.safetensors"
    save_file({"x": torch.zeros(1)}, str(untagged))

    m1 = read_lora_meta(str(tagged))
    assert m1.model_family == "krea2" and m1.family_explicit is True
    m2 = read_lora_meta(str(untagged))
    assert m2.model_family == "anima" and m2.family_explicit is False


def test_apply_loras_untagged_peft_file_routes_to_merge(tmp_path):
    """用户场景复现：civitai krea2 LoRA（PEFT 键、零 metadata）挂 fp8 krea2
    底模——不再被 grandfather 判成 anima 拒绝，直接走 merge。"""
    from safetensors.torch import save_file

    from studio.services.inference.core import LoRASpec, apply_loras

    model = _make_fp8_model()
    path = tmp_path / "civit_krea2.safetensors"
    save_file(_peft_lora_sd(["blocks.0.q"]), str(path))  # 无任何 metadata
    before = model.blocks[0].q.weight.detach().view(torch.uint8).clone()

    adapters = apply_loras(
        model, [LoRASpec(path=str(path), scale=1.0)], "cpu", torch.float32,
        family_id="krea2",
    )

    assert isinstance(adapters[0], Fp8LoraMergeAdapter)
    assert not torch.equal(model.blocks[0].q.weight.view(torch.uint8), before)


def test_apply_loras_untagged_alien_keys_fail_fast(tmp_path):
    """无标记 + 键全对不上（真异族/坏文件）→ merge 全 miss 报错，不静默。"""
    from safetensors.torch import save_file

    from studio.services.inference.core import LoRASpec, apply_loras

    model = _make_fp8_model()
    path = tmp_path / "alien.safetensors"
    save_file(_plain_lora_sd(["no.such.layer"]), str(path))

    with pytest.raises(ValueError, match="无法对应"):
        apply_loras(
            model, [LoRASpec(path=str(path), scale=1.0)], "cpu", torch.float32,
            family_id="krea2",
        )
