"""MOSS int8 出图的 ComfyUI parity（PR-6，MOSS 设计稿 §5.9 / M9）。

把 comfy-kitchen 0.2.38 eager 后端 / ComfyUI 0.41.0 的算子配方固化成单测：

- W8A8 线性层：激活旋转（3 维批量）→ 逐行量化（四舍六入五成双）→ ``torch._int_mm``
  → ``(x_scale·w_scale)`` fp32 缩放；``int8_mm`` 的补零规则
- 反量化前向（TE）、merge 后的重新量化（随机舍入，seed ≤ 0 退回四舍五入）
- LoRA merge：融合 qkv 按整块 merge 与取种子，拆回 q/k/v；detach 逐位还原
- Comfy 口径 DiT 前向与上游前向数学等价；percent_to_sigma 的 double 精度与 SLG 边界
- er_sde 与 ComfyUI ``sample_er_sde`` 原式逐位一致

真机（作者本地 ComfyUI，同 torch 2.7 环境）实测：DiT 前向、LoRA merge、端到端
采样 latent 与 ComfyUI 逐位一致（max diff 0），数据见 PR 描述。
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from tests.test_moss_dit import _CFG, _comfy_state, _inputs, _model
from training.families.comfy_lora_merge import QuantizedMergeAdapter, string_to_seed
from training.families.moss import quant_int8 as q8
from training.families.moss.lora_int8_merge import merge_loras_into_int8_model

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="int8 GEMM / 逐位 parity 在 CUDA 上验证")
_GROUP = 16


# ── 算子配方 ────────────────────────────────────────────────────────────────

def test_quantize_rowwise_matches_comfy_recipe() -> None:
    x = torch.randn(5, 64, dtype=torch.bfloat16) * 3
    x[2] = 0  # 全零行：scale 被 clamp 到 1e-30，转 bf16 为 0 → 换成 tiny
    q, s = q8.quantize_rowwise(x)
    ref_s = (x.abs().amax(-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    div = ref_s.to(torch.bfloat16)
    div = torch.where(div == 0, torch.full_like(div, torch.finfo(torch.bfloat16).tiny), div)
    ref_q = (x / div).round_().clamp_(-128, 127).to(torch.int8)
    assert torch.equal(s, ref_s) and s.dtype == torch.float32
    assert torch.equal(q, ref_q)
    assert torch.equal(q[2], torch.zeros_like(q[2]))


def test_round_int8_stochastic_seed_semantics() -> None:
    x = torch.linspace(-3, 3, 64, dtype=torch.float16).reshape(4, 16)
    a = q8._round_int8(x.clone(), seed=123)
    b = q8._round_int8(x.clone(), seed=123)
    assert torch.equal(a, b)  # 同 seed 确定
    assert not torch.equal(a, q8._round_int8(x.clone(), seed=124))
    # seed ≤ 0：四舍六入五成双（comfy _round_int8 的 else 分支）
    assert torch.equal(q8._round_int8(x.clone(), seed=0), x.clone().round_().to(torch.int8))
    # 随机舍入 = floor(x + U[0,1))
    g = torch.Generator().manual_seed(123)
    ref = (x.clone() + torch.rand(x.shape, dtype=x.dtype, generator=g)).floor_().clamp_(-128, 127).to(torch.int8)
    assert torch.equal(a, ref)


def test_dequant_forward_matches_linear_on_dequantized_weight() -> None:
    lin = nn.Linear(64, 32, bias=True)
    w_q, scale = _quantize(lin.weight.detach())
    lin.weight = nn.Parameter(w_q, requires_grad=False)
    q8.patch_int8_linear(lin, scale, _GROUP, forward="dequant")
    x = torch.randn(3, 64, dtype=torch.float32)
    ref = torch.nn.functional.linear(x, q8.dequantize(w_q, scale, _GROUP), lin.bias)
    assert torch.equal(lin(x), ref)


def test_patch_int8_linear_rejects_unknown_forward() -> None:
    lin = nn.Linear(64, 32, bias=False)
    w_q, scale = _quantize(lin.weight.detach())
    lin.weight = nn.Parameter(w_q, requires_grad=False)
    with pytest.raises(ValueError, match="未知 int8 前向"):
        q8.patch_int8_linear(lin, scale, _GROUP, forward="w4a4")


@_cuda
def test_w8a8_linear_matches_comfy_recipe_bitwise() -> None:
    torch.manual_seed(0)
    w = torch.randn(48, 256, device="cuda")
    w_q, scale = _quantize(w.cpu(), group=256)
    w_q, scale = w_q.cuda(), scale.cuda()
    for m in (1, 7, 33, 100):  # M 补零到 32 的倍数且 ≥ 32
        x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
        h = q8.hadamard(256, device="cuda", dtype=torch.bfloat16)
        xr = torch.matmul(x.reshape(-1, 1, 256), h).reshape(-1, 256)
        s = (xr.abs().amax(-1, keepdim=True).float() / 127).clamp(min=1e-30)
        q = (xr / s.to(torch.bfloat16)).round_().clamp_(-128, 127).to(torch.int8)
        pad = max(32, -(-m // 32) * 32) - m
        acc = torch._int_mm(torch.cat([q, q.new_zeros(pad, 256)]), w_q.t().contiguous())[:m]
        ref = (acc.float() * (s * scale.reshape(1, -1))).to(torch.bfloat16)
        assert torch.equal(q8.w8a8_linear(x, w_q, scale, 256), ref), m


# ── LoRA merge ──────────────────────────────────────────────────────────────

def _quantize(w: torch.Tensor, group: int = _GROUP):
    rot = q8._rotate_weight_comfy(w.float(), group)
    s = (rot.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
    return torch.round(rot / s).clamp(-127, 127).to(torch.int8), s


class _Proj(nn.Module):
    def __init__(self, fused: bool):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        torch.manual_seed(3)
        proj = nn.ModuleDict()
        for name, (o, i) in {"q": (32, 64), "k": (32, 64), "v": (32, 64), "o": (64, 32)}.items():
            lin = nn.Linear(i, o, bias=False)
            w_q, s = _quantize(lin.weight.detach())
            lin.weight = nn.Parameter(w_q, requires_grad=False)
            q8.patch_int8_linear(lin, s, _GROUP, forward="w8a8")
            proj[name] = lin
        self.layers[0].proj = proj
        self.moss_fused_qkv = fused


def _lora(layers: dict[str, tuple[int, int]], seed: int = 5) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name, (o, i) in layers.items():
        base = f"lora_unet_layers_0_proj_{name}"
        sd[f"{base}.lora_down.weight"] = torch.randn(4, i, generator=g) * 0.1
        sd[f"{base}.lora_up.weight"] = torch.randn(o, 4, generator=g) * 0.1
        sd[f"{base}.alpha"] = torch.tensor(2.0)
    return sd


def _manual_merge(mods, names, lora, strength, seed_key):
    q = torch.cat([mods[n].weight for n in names])
    s = torch.cat([mods[n].weight_scale.reshape(-1, 1) for n in names])
    w16 = q8.dequantize(q, s, _GROUP).to(torch.float16)
    row = 0
    for n in names:
        rows = mods[n].out_features
        key = f"lora_unet_{n.replace('.', '_')}"
        if f"{key}.lora_up.weight" in lora:
            up, down = lora[f"{key}.lora_up.weight"].float(), lora[f"{key}.lora_down.weight"].float()
            alpha = float(lora[f"{key}.alpha"]) / down.shape[0]
            w16[row:row + rows] += ((strength * alpha) * (up @ down)).to(torch.float16)
        row += rows
    return q8.requantize_convrot(w16, _GROUP, seed=string_to_seed(seed_key))


def test_merge_fused_qkv_uses_comfy_unit_and_seed() -> None:
    model = _Proj(fused=True)
    lora = _lora({"q": (32, 64), "v": (32, 64), "o": (64, 32)})
    mods = dict(model.named_modules())
    names = ["layers.0.proj.q", "layers.0.proj.k", "layers.0.proj.v"]
    q_ref, s_ref = _manual_merge(mods, names, lora, 0.8, "diffusion_model.layers.0.proj.qkv.weight")
    o_ref = _manual_merge(mods, ["layers.0.proj.o"], lora, 0.8, "diffusion_model.layers.0.proj.o.weight")
    orig = {n: (mods[n].weight.clone(), mods[n].weight_scale.clone()) for n in names + ["layers.0.proj.o"]}

    adapter = merge_loras_into_int8_model(model, [(lora, 0.8, "t")])
    assert isinstance(adapter, QuantizedMergeAdapter)
    # k 没有 LoRA，但与 q/v 同属融合块：Comfy 整块重新量化，k 也被重写
    assert torch.equal(torch.cat([mods[n].weight for n in names]), q_ref)
    assert torch.equal(torch.cat([mods[n].weight_scale.reshape(-1, 1) for n in names]), s_ref)
    assert torch.equal(mods["layers.0.proj.o"].weight, o_ref[0])

    assert adapter.detach() is True
    for n, (w, s) in orig.items():
        assert torch.equal(mods[n].weight, w) and torch.equal(mods[n].weight_scale, s)


def test_merge_unfused_checkpoint_keys_each_layer() -> None:
    model = _Proj(fused=False)
    lora = _lora({"q": (32, 64)})
    mods = dict(model.named_modules())
    k_before = mods["layers.0.proj.k"].weight.clone()
    q_ref, _ = _manual_merge(mods, ["layers.0.proj.q"], lora, 1.0, "diffusion_model.layers.0.proj.q.weight")
    merge_loras_into_int8_model(model, [(lora, 1.0, "t")])
    assert torch.equal(mods["layers.0.proj.q"].weight, q_ref)
    assert torch.equal(mods["layers.0.proj.k"].weight, k_before)  # 非融合：k 不动


def test_merge_rejects_dora_and_missing_layers() -> None:
    lora = _lora({"q": (32, 64)})
    lora["lora_unet_layers_0_proj_q.dora_scale"] = torch.ones(32)
    with pytest.raises(ValueError, match="DoRA"):
        merge_loras_into_int8_model(_Proj(fused=True), [(lora, 1.0, "t")])
    with pytest.raises(ValueError, match="无法对应"):
        merge_loras_into_int8_model(_Proj(fused=True), [(
            {"lora_unet_nope.lora_down.weight": torch.zeros(1, 1), "lora_unet_nope.lora_up.weight": torch.zeros(1, 1)},
            1.0, "t")])


def test_apply_loras_routes_int8_base_to_merge(tmp_path) -> None:
    import json

    from safetensors.torch import save_file

    from studio.services.inference.core import LoRASpec, apply_loras, model_uses_lora_merge

    model = _Proj(fused=True)
    assert model_uses_lora_merge(model)
    path = tmp_path / "moss.safetensors"
    save_file(_lora({"q": (32, 64)}), str(path), metadata={
        "ss_network_dim": "4", "ss_network_alpha": "2.0",
        "ss_network_args": json.dumps({"model_family": "moss", "algo": "lora"}),
    })
    before = model.layers[0].proj["q"].weight.clone()
    adapters = apply_loras(model, [LoRASpec(path=str(path), scale=0.8)], "cpu", torch.float32, family_id="moss")
    assert len(adapters) == 1 and isinstance(adapters[0], QuantizedMergeAdapter)
    assert not torch.equal(model.layers[0].proj["q"].weight, before)


# ── 前向 / 采样口径 ──────────────────────────────────────────────────────────

def test_comfy_forward_matches_upstream_forward_math() -> None:
    from modeling.moss.moss_modeling import patchify, unpatchify
    from training.families.moss.comfy_forward import comfy_forward

    model = _model(seed=2).double()
    with torch.no_grad():
        for p in model.parameters():
            p.div_(3.0)  # 回到默认初始化：_model 放大 3 倍会把 fp32 舍入放大成混沌差异
    x32, ctx = _inputs(h=4, w=6, text=5)
    x128 = patchify(x32.squeeze(2).double())
    sigma = torch.tensor([0.6], dtype=torch.float64)
    with torch.no_grad():
        ref = patchify(model(unpatchify(x128), sigma, ctx.double()))
        out = comfy_forward(model, x128, sigma, ctx.double())
        skipped = comfy_forward(model, x128, sigma, ctx.double(), skip_layers=(1,))
        ref_skip = patchify(model(unpatchify(x128), sigma, ctx.double(), skip_layers=[1]))
    # 两条前向数学相同、舍入路径不同：Comfy 口径里 RoPE 频率表、时间嵌入与
    # rope_split_half 固定 fp32 计算，所以 fp64 下只差 fp32 舍入量级
    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    assert rel(out, ref) < 1e-5
    assert rel(skipped, ref_skip) < 1e-5
    assert rel(skipped, ref) > 1e-3  # 跳层确实生效


def test_percent_to_sigma_is_double_like_comfy() -> None:
    from training.families.moss.sampling import build_sigmas, percent_to_sigma

    assert percent_to_sigma(0.5) == 5 * 0.5 / (1 + 4 * 0.5)  # Python double，不经 fp32
    # SLG 窗口闭区间：fp32 σ 表里的 5/6 比 double 边界略小 → Comfy 在这一步不跳层
    s = float(build_sigmas(8)[4])
    assert s == pytest.approx(5 / 6) and s < percent_to_sigma(0.5)


def _comfy_er_sde_reference(denoise, x, sigmas, seed, shift):
    """ComfyUI ``sample_er_sde``（CONST、默认 noise sampler）的逐字移植，作对拍参照。"""
    from training.inference_samplers.er_sde import _offset_first_sigma_for_snr

    g = torch.Generator(device=x.device).manual_seed(seed + (1 if x.device.type == "cpu" else 0))
    sigmas = _offset_first_sigma_for_snr(sigmas.clone(), shift=shift)
    er_lambdas = sigmas.logit().neg().neg().exp()

    def ns(v):
        return v * ((v ** 0.3).exp() + 10.0)

    pts = torch.arange(0, 200.0, dtype=torch.float32)
    old, old_d = None, None
    for i in range(len(sigmas) - 1):
        den = denoise(x, sigmas[i])
        stage = min(3, i + 1)
        if sigmas[i + 1] == 0:
            x = den
        else:
            ls, lt = er_lambdas[i], er_lambdas[i + 1]
            a_s, a_t = sigmas[i] / ls, sigmas[i + 1] / lt
            r = ns(lt) / ns(ls)
            x = (a_t / a_s) * r * x + a_t * (1 - r) * den
            if stage >= 2:
                dt = lt - ls
                step = -dt / 200.0
                pos = lt + pts * step
                sp = ns(pos)
                s_ = torch.sum(1 / sp) * step
                d = (den - old) / (ls - er_lambdas[i - 1])
                x = x + a_t * (dt + s_ * ns(lt)) * d
                if stage >= 3:
                    s_u = torch.sum((pos - ls) / sp) * step
                    u = (d - old_d) / ((ls - er_lambdas[i - 2]) / 2)
                    x = x + a_t * ((dt ** 2) / 2 + s_u * ns(lt)) * u
                old_d = d
            noise = torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=g)
            x = x + a_t * noise * 1.0 * (lt ** 2 - ls ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old = den
    return x


def test_er_sde_matches_comfy_formulas_bitwise() -> None:
    from training.families.moss.sampling import build_sigmas
    from training.inference_samplers.er_sde import sample

    sigmas = build_sigmas(6)
    x0 = torch.randn(1, 4, 3, 3, generator=torch.Generator().manual_seed(1))

    def denoise(x, sigma):
        return x * 0.5 - 0.1 * float(sigma)

    ours = sample(denoise, x0.clone(), sigmas, seed=42, shift=5.0)
    ref = _comfy_er_sde_reference(denoise, x0.clone(), sigmas, 42, 5.0)
    assert torch.equal(ours, ref)


def test_load_dit_purpose_selects_forward(tmp_path) -> None:
    from safetensors.torch import save_file

    from training.families import get_family

    path = tmp_path / "moss-int8.safetensors"
    save_file(_comfy_state(_model(seed=4), fuse_qkv=True, int8=True), str(path))
    fam = get_family("moss")
    gen = fam.load_dit(str(path), "cpu", torch.float32, attention_backend="none", purpose="generate")
    train = fam.load_dit(str(path), "cpu", torch.float32, attention_backend="none")
    assert gen.moss_comfy_forward is True and gen.moss_fused_qkv is True
    assert train.moss_comfy_forward is False
    assert gen.layers[0].proj["q"].forward.__func__ is q8._int8_convrot_w8a8_forward
    assert train.layers[0].proj["q"].forward.__func__ is q8._int8_convrot_forward
    assert math.isclose(_CFG.dim, gen.config.dim)
