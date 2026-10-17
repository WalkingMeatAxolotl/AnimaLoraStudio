"""MOSS-Image DiT 结构 / 加载 / int8 convrot 运行时单测（MOSS 设计稿 §5.3 / §5.4）。

小配置随机权重，全部 CPU：
- 每样本按真实文本长度前向：右 padding 不改变结果
- 4D RoPE：r 轴恒等、图像 f 坐标 = 文本长度
- σ → ×1000 的 timestep 口径
- loader：融合 / 非融合 qkv、外层前缀、派生键丢弃、strict 报错
- int8 convrot：按 comfy-kitchen 约定量化后反量化误差、W8A16 前向等价、梯度回传
- 梯度检查点与直通前向等价
"""
from __future__ import annotations

import json
import math

import pytest
import torch
from safetensors.torch import save_file

from modeling.moss.moss_modeling import MossConfig, MossImageTransformer, _rope, patchify, unpatchify
from training.families.moss.forward import MossTextCondition, forward_train
from training.families.moss.loader import inspect_moss_checkpoint, load_moss_model
from training.families.moss.quant_int8 import (
    dequantize,
    hadamard,
    parse_comfy_quant,
    patch_int8_linear,
)

_CFG = MossConfig(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=32, freq_dim=16)
_GROUP = 16


def _model(seed: int = 0) -> MossImageTransformer:
    torch.manual_seed(seed)
    model = MossImageTransformer(_CFG).eval()
    with torch.no_grad():
        for p in model.parameters():  # 默认初始化下 gate / modulation 太小，放大让各层都起作用
            p.mul_(3.0)
    return model


def _inputs(batch: int = 1, h: int = 4, w: int = 6, text: int = 5, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 32, 1, h, w, generator=g)
    ctx = torch.randn(batch, text, _CFG.text_dim, generator=g)
    return x, ctx


def _comfy_quant_blob(group: int, fmt: str = "int8_tensorwise", convrot: bool = True) -> torch.Tensor:
    conf = {"format": fmt, "convrot": convrot, "convrot_groupsize": group}
    return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)


def _quantize_convrot(w: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """comfy-kitchen quantize_int8_convrot_weight 的同口径参照：W@Hᵀ 后逐行 int8。"""
    out_f, in_f = w.shape
    h = hadamard(group, dtype=torch.float32)
    rot = (w.float().reshape(out_f, in_f // group, group) @ h.T).reshape(out_f, in_f)
    scale = (rot.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    q = torch.round(rot / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def _comfy_state(model: MossImageTransformer, *, fuse_qkv: bool, int8: bool) -> dict[str, torch.Tensor]:
    """把模型权重组织成 ComfyUI 单文件形态（可选融合 qkv / int8 convrot）。"""
    sd = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    out: dict[str, torch.Tensor] = {"rope.bands": torch.ones(2)}
    for i in range(_CFG.num_layers):
        p = f"layers.{i}.proj"
        if fuse_qkv:
            sd[f"{p}.qkv.weight"] = torch.cat([sd.pop(f"{p}.{n}.weight") for n in "qkv"])
        out[f"layers.{i}.comfy_attention.config"] = torch.tensor(list(b'{"attention": "x"}'), dtype=torch.uint8)
    for k, v in sd.items():
        quant_target = int8 and k.startswith("layers.") and ".proj." in k
        if quant_target:
            layer = k[: -len(".weight")]
            q, s = _quantize_convrot(v, _GROUP)
            out[k] = q
            out[f"{layer}.weight_scale"] = s
            out[f"{layer}.comfy_quant"] = _comfy_quant_blob(_GROUP)
        else:
            out[k] = v
    return out


# ── 结构 ────────────────────────────────────────────────────────────────


def test_forward_shapes_and_patchify_roundtrip() -> None:
    model = _model()
    x, ctx = _inputs(batch=2)
    with torch.no_grad():
        out5 = model(x, torch.tensor([0.3, 0.7]), ctx)
        out4 = model(x[:, :, 0], torch.tensor([0.3, 0.7]), ctx)
    assert out5.shape == x.shape
    torch.testing.assert_close(out5[:, :, 0], out4)
    z = torch.randn(1, 32, 4, 6)
    assert torch.equal(unpatchify(patchify(z)), z)


def test_right_padding_does_not_change_output() -> None:
    model = _model()
    x, _ = _inputs(batch=2)
    g = torch.Generator().manual_seed(7)
    ctx_a = torch.randn(1, 3, _CFG.text_dim, generator=g)
    ctx_b = torch.randn(1, 6, _CFG.text_dim, generator=g)
    padded = torch.zeros(2, 6, _CFG.text_dim)
    padded[0, :3] = ctx_a[0]
    padded[0, 3:] = 99.0  # padding 内容任意，不能泄漏进结果
    padded[1] = ctx_b[0]
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1] * 6], dtype=torch.bool)
    t = torch.tensor([0.4, 0.4])
    with torch.no_grad():
        batched = model(x, t, padded, attention_mask=mask)
        single_a = model(x[:1], t[:1], ctx_a)
        single_b = model(x[1:], t[1:], ctx_b)
    # batch=2 与 batch=1 的 modulation matmul 走不同 BLAS 分块，只差浮点舍入
    torch.testing.assert_close(batched[:1], single_a, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(batched[1:], single_b, rtol=1e-4, atol=1e-3)


def test_rope_reference_axis_identity_and_image_f_coordinate() -> None:
    head_dim, text_len, h, w = 16, 5, 2, 3
    rope = _rope(head_dim, 10000.0, h, w, text_len, "cpu")[0]  # [h*w + text, head_dim]
    bands = head_dim // 8
    sin, cos = rope[:, : rope.shape[1] // 2], rope[:, rope.shape[1] // 2:]
    # r 轴（每个 token 的前 bands 个频率）恒等：sin=0、cos=1
    assert torch.all(sin[:, :bands] == 0) and torch.all(cos[:, :bands] == 1)
    # 图像 token 的 f 坐标 = 文本长度；f 轴第一个频率 = 1
    torch.testing.assert_close(sin[: h * w, bands], torch.full((h * w,), math.sin(text_len)))
    # 文本 token 的 f 坐标 = 0..text_len-1
    torch.testing.assert_close(sin[h * w:, bands], torch.sin(torch.arange(text_len).float()))


def test_timestep_is_sigma_times_1000() -> None:
    model = _model()
    mod, _head = model._time(torch.tensor([0.25]), 1, "cpu")
    half = _CFG.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = 250.0 * freq
    emb = torch.cat((phase.cos(), phase.sin())).float()[None]
    ref = model.shared_block_modulation(model.time_embedding(emb)).view(1, 6, -1)
    torch.testing.assert_close(mod, ref)


def test_skip_layers_validation_and_effect() -> None:
    model = _model()
    x, ctx = _inputs()
    with torch.no_grad():
        full = model(x, torch.tensor([0.5]), ctx)
        skipped = model(x, torch.tensor([0.5]), ctx, skip_layers=[1])
    assert not torch.allclose(full, skipped)
    with pytest.raises(ValueError, match="skip_layers"):
        model(x, torch.tensor([0.5]), ctx, skip_layers=[3])


def test_gradient_checkpoint_matches_plain_forward() -> None:
    model = _model().train()
    model.requires_grad_(False)
    x, ctx = _inputs()
    grads = []
    for use_checkpoint in (False, True):
        xi = x.clone().requires_grad_(True)
        out = forward_train(model, xi, torch.tensor([0.6]), MossTextCondition(ctx, torch.ones(1, ctx.shape[1])),
                            use_checkpoint=use_checkpoint)
        out.square().mean().backward()
        grads.append((out.detach(), xi.grad))
    torch.testing.assert_close(grads[0][0], grads[1][0])
    torch.testing.assert_close(grads[0][1], grads[1][1])


# ── int8 convrot 运行时 ──────────────────────────────────────────────────


def test_hadamard_is_symmetric_orthogonal_involution() -> None:
    for g in (16, 64, 256):
        h = hadamard(g, dtype=torch.float64)
        torch.testing.assert_close(h, h.T)
        torch.testing.assert_close(h @ h, torch.eye(g, dtype=torch.float64))
    with pytest.raises(ValueError):
        hadamard(32)


def test_dequantize_recovers_weight_within_int8_error() -> None:
    torch.manual_seed(0)
    w = torch.randn(48, 64)
    q, s = _quantize_convrot(w, _GROUP)
    rel = ((dequantize(q, s, _GROUP) - w).norm() / w.norm()).item()
    assert rel < 0.02  # 逐行 int8 的正常误差量级（真实权重实测 ~0.9%）


def test_int8_forward_equals_linear_on_dequantized_weight_and_backprops() -> None:
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 48, bias=True)
    q, s = _quantize_convrot(lin.weight.detach(), _GROUP)
    w_deq = dequantize(q, s, _GROUP)
    lin.weight = torch.nn.Parameter(q, requires_grad=False)
    lin.bias.requires_grad_(False)
    patch_int8_linear(lin, s, _GROUP)
    x = torch.randn(3, 7, 64, requires_grad=True)
    lora_a = torch.zeros(64, 48, requires_grad=True)
    out = lin(x) + x @ lora_a  # LoRA 旁路：底模 int8 冻结，增量走全精度
    ref = torch.nn.functional.linear(x.detach(), w_deq, lin.bias)
    torch.testing.assert_close(out.detach(), ref, atol=1e-4, rtol=1e-4)
    out.sum().backward()
    assert lora_a.grad is not None and torch.isfinite(lora_a.grad).all()
    # lora_a 初值为 0，对 x 的梯度只来自底模：ones @ W_deq
    torch.testing.assert_close(x.grad, torch.ones(3, 7, 48) @ w_deq, atol=1e-4, rtol=1e-4)


def test_parse_comfy_quant_rejects_other_formats() -> None:
    assert parse_comfy_quant(_comfy_quant_blob(64))["convrot_groupsize"] == 64
    with pytest.raises(ValueError, match="int8_tensorwise"):
        parse_comfy_quant(_comfy_quant_blob(64, fmt="float8_e4m3fn"))
    with pytest.raises(ValueError, match="int8_tensorwise"):
        parse_comfy_quant(_comfy_quant_blob(64, convrot=False))


# ── loader ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fuse_qkv", [True, False])
def test_loader_bf16_matches_source_model(tmp_path, fuse_qkv) -> None:
    src = _model(seed=2)
    path = tmp_path / "moss.safetensors"
    save_file(_comfy_state(src, fuse_qkv=fuse_qkv, int8=False), str(path))
    info = inspect_moss_checkpoint(path)
    assert info.config == _CFG and info.fused_qkv is fuse_qkv and not info.is_int8
    loaded = load_moss_model(path, "cpu", torch.float32)
    for k, v in src.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v)
    x, ctx = _inputs()
    with torch.no_grad():
        # 权重逐元素相同；只差内存布局带来的 BLAS 舍入
        torch.testing.assert_close(loaded(x, torch.tensor([0.5]), ctx), src(x, torch.tensor([0.5]), ctx),
                                   rtol=1e-4, atol=1e-3)
    assert not any(p.requires_grad for p in loaded.parameters())


def test_loader_strips_known_prefix(tmp_path) -> None:
    src = _model(seed=2)
    path = tmp_path / "moss.safetensors"
    state = _comfy_state(src, fuse_qkv=False, int8=False)
    save_file({f"model.diffusion_model.{k}": v for k, v in state.items()}, str(path))
    assert inspect_moss_checkpoint(path).prefix == "model.diffusion_model."
    load_moss_model(path, "cpu", torch.float32)


def test_loader_int8_convrot_splits_fused_qkv_and_stays_close(tmp_path) -> None:
    src = _model(seed=4)
    path = tmp_path / "moss-int8.safetensors"
    save_file(_comfy_state(src, fuse_qkv=True, int8=True), str(path))
    info = inspect_moss_checkpoint(path)
    assert info.is_int8 and len(info.int8_layers) == _CFG.num_layers * 6
    assert info.int8_layers["layers.0.proj.k"] == _GROUP
    loaded = load_moss_model(path, "cpu", torch.float32)
    k_lin = loaded.layers[0].proj["k"]
    assert k_lin.weight.dtype == torch.int8 and k_lin.weight.shape == (_CFG.dim, _CFG.dim)
    rel_w = ((dequantize(k_lin.weight, k_lin.weight_scale, _GROUP) - src.layers[0].proj["k"].weight).norm()
             / src.layers[0].proj["k"].weight.norm()).item()
    assert rel_w < 0.02
    # text_embedder / embedder 等非量化层原样浮点
    assert loaded.text_embedder.blocks[0].qkv.weight.dtype == torch.float32
    # 参照：把源模型的量化层换成「反量化后的浮点权重」，int8 前向应与之代数等价
    ref_model = _model(seed=4)
    with torch.no_grad():
        for i in range(_CFG.num_layers):
            for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
                lin = loaded.layers[i].proj[name]
                ref_model.layers[i].proj[name].weight.copy_(dequantize(lin.weight, lin.weight_scale, _GROUP))
    x, ctx = _inputs()
    with torch.no_grad():
        out = loaded(x, torch.tensor([0.5]), ctx)
        ref = ref_model(x, torch.tensor([0.5]), ctx)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)


def test_loader_rejects_mismatched_and_unsupported(tmp_path) -> None:
    src = _model()
    state = _comfy_state(src, fuse_qkv=False, int8=False)
    state.pop("final_norm_video.weight")
    bad = tmp_path / "missing.safetensors"
    save_file(state, str(bad))
    with pytest.raises(ValueError, match="缺少"):
        load_moss_model(bad, "cpu", torch.float32)

    state = _comfy_state(src, fuse_qkv=False, int8=True)
    state.pop("layers.0.proj.o.weight_scale")
    no_scale = tmp_path / "no-scale.safetensors"
    save_file(state, str(no_scale))
    with pytest.raises(ValueError, match="weight_scale"):
        inspect_moss_checkpoint(no_scale)

    state = _comfy_state(src, fuse_qkv=False, int8=False)
    state["layers.0.proj.o.weight"] = state["layers.0.proj.o.weight"].to(torch.float8_e4m3fn)
    fp8 = tmp_path / "fp8.safetensors"
    save_file(state, str(fp8))
    with pytest.raises(ValueError, match="F8_E4M3"):
        inspect_moss_checkpoint(fp8)

    with pytest.raises(ValueError, match="不是 MOSS-Image DiT"):
        other = tmp_path / "other.safetensors"
        save_file({"blocks.0.weight": torch.zeros(1)}, str(other))
        inspect_moss_checkpoint(other)
