"""MOSS-Image DiT 的 ComfyUI 口径推理前向（出图 parity，MOSS 设计稿 §5.9 / M9）。

``modeling.moss`` 的前向按官方上游实现（训练用：调制量在 fp32、RoPE 转成 bf16 后
旋转、手写 RMSNorm），与 ComfyUI 的推理前向数学相同但舍入路径不同。本模块复用同一
套权重，按 ComfyUI 0.41.0 ``comfy/ldm/moss_image/model.py`` 的推理分支与
comfy-kitchen 0.2.38 eager 后端逐算子排列：

- 调制归一化 ``rms_adaln(x, w·(1+scale)−1, shift)``：``w·(1+scale)−1`` 在 bf16 里算，
  再 ``rms_norm(x) · (1+scale') + shift``
- q/k ``rms_rope_split_half``：``rms_norm(x, weight)`` 后与 fp32 频率做 split-half 旋转
  （乘积提升到 fp32），结果转回 bf16
- RoPE 频率：flux ``rope``（fp64 计算、fp32 存）；时间嵌入：flux ``timestep_embedding``
- 门控残差：原地 ``addcmul_``（一次舍入）
- 线性层：调用模型自己的 Linear（int8 层已按 ``w8a8`` 挂 comfy-kitchen eager 前向）

**不在逐位范围内**：DiT 注意力。ComfyUI 对 Comfy-Org 的 MOSS 文件用 comfy-kitchen
自带的 INT8 SageAttention CUDA 内核（闭源二进制），这里用 PyTorch SDPA；文本精炼块
在 ComfyUI 里走 xformers，这里同样用 SDPA。

派生署名见 THIRD_PARTY_NOTICES（ComfyUI GPL-3.0 + comfy_kitchen Apache-2.0）。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

_EPS = 1e-6
_ROPE_THETA = 10000


def _rms_norm(x: Tensor, weight: Tensor | None) -> Tensor:
    """comfy ``ops.RMSNorm``（权重转成输入 dtype 的 ``F.rms_norm``）。"""
    if weight is None:
        return F.rms_norm(x, (x.shape[-1],), eps=_EPS)
    return F.rms_norm(x, weight.shape, weight=weight.to(x.dtype), eps=_EPS)


def _rms_adaln(x: Tensor, norm_weight: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """ComfyUI ``modulated_rms_norm`` → comfy-kitchen eager ``rms_adaln``。"""
    folded = norm_weight.to(x.dtype) * (1 + scale) - 1
    return F.rms_norm(x, x.shape[-1:], eps=_EPS) * (1 + folded) + shift


def _apply_rope_split_half(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """comfy-kitchen eager ``apply_rope_split_half1``（逐句照搬）。"""
    t = x
    if torch.promote_types(x.dtype, freqs_cis.dtype) != freqs_cis.dtype:
        t = x.to(freqs_cis.dtype)
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis.unsqueeze(0)
    diagonal = freqs_cis.diagonal(dim1=-2, dim2=-1).movedim(-1, -2)
    out = t.unflatten(-1, (2, -1)) * diagonal
    half = x.shape[-1] // 2
    out[..., 0, :].addcmul_(t[..., half:], freqs_cis[..., 0, 1])
    out[..., 1, :].addcmul_(t[..., :half], freqs_cis[..., 1, 0])
    return out.reshape(x.shape).type_as(x)


def _rms_rope_split_half(x: Tensor, pe: Tensor, weight: Tensor) -> Tensor:
    """comfy-kitchen eager ``_rms_rope1(split_half=True)``。"""
    x_norm = F.rms_norm(x, (x.shape[-1],), weight=weight.to(x.dtype), eps=_EPS)
    return _apply_rope_split_half(x_norm, pe)


def _flux_rope(pos: Tensor, dim: int) -> Tensor:
    """ComfyUI ``ldm/flux/math.rope``：fp64 计算频率与相位，输出 fp32 的 2×2 旋转矩阵。"""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (_ROPE_THETA ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32).double(), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.to(dtype=torch.float32)


def _rope(head_dim: int, h: int, w: int, text_len: int, device) -> Tensor:
    """ComfyUI ``MOSSImage.rope`` + flux ``EmbedND``：图像在前（f = 文本长度，行列居中），
    文本在后；返回 ``[1, N, 1, head_dim/2, 2, 2]``。"""
    ids = torch.zeros(h * w + text_len, 4, device=device, dtype=torch.float32)
    ids[:h * w, 1] = text_len
    ids[:h * w, 2] = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
    ids[:h * w, 3] = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
    ids[h * w:, 1] = torch.arange(text_len, device=device, dtype=torch.float32)
    ids = ids.unsqueeze(0)
    axis_dim = head_dim // 4
    emb = torch.cat([_flux_rope(ids[..., i], axis_dim) for i in range(4)], dim=-3).unsqueeze(1)
    return emb.transpose(1, 2).contiguous()


def _timestep_embedding(t: Tensor, dim: int = 256, max_period: int = 10000) -> Tensor:
    """flux ``timestep_embedding``（``time_factor=1000``，fp32）。"""
    t = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding.to(t)


def _attention(q: Tensor, k: Tensor, v: Tensor, q_norm_w: Tensor, k_norm_w: Tensor,
               pe: Tensor, heads: int) -> Tensor:
    """ComfyUI ``moss_image.attention`` 推理分支；注意力内核用 SDPA（见模块 docstring）。"""
    b, n, _ = q.shape
    q = _rms_rope_split_half(q.view(b, n, heads, -1), pe, q_norm_w)
    k = _rms_rope_split_half(k.view(b, n, heads, -1), pe, k_norm_w)
    q, k = q.flatten(2), k.flatten(2)
    dim_head = q.shape[-1] // heads
    # comfy attention_pytorch：reshape 到 heads 后 transpose 成 (b, heads, n, d)
    q, k, v = (t.unsqueeze(3).reshape(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
    return _sdpa(q, k, v).transpose(1, 2).reshape(b, -1, heads * dim_head)


def _sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """comfy ``ops.scaled_dot_product_attention``：输入不小于 128K 元素时按
    FLASH → CUDNN → EFFICIENT → MATH 的优先级选 SDPA 后端（torch 默认顺序里 cuDNN
    排在 efficient 之后，两者数值不同）；小输入直接调用。"""
    if q.nelement() < 1024 * 128 or not q.is_cuda or _SDPA_PRIORITY is None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    from torch.nn.attention import sdpa_kernel

    with sdpa_kernel(_SDPA_PRIORITY, set_priority=True):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)


def _sdpa_priority():
    try:
        import inspect

        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return None
    if "set_priority" not in inspect.signature(sdpa_kernel).parameters:
        return None
    return [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


_SDPA_PRIORITY = _sdpa_priority()


def _situ(hidden: Tensor, gate: Tensor) -> Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _text_block(block, x: Tensor, pe: Tensor) -> Tensor:
    q, k, v = block.qkv(_rms_norm(x, block.norm1.weight)).chunk(3, dim=-1)
    x = x + block.proj(_attention(q, k, v, block.q_norm.weight, block.k_norm.weight, pe, block.heads))
    y = _rms_norm(x, block.norm2.weight)
    return x + block.ffn_down(_situ(block.ffn_up(y), block.ffn_gate(y)))


def _dit_block(block, x: Tensor, mod: tuple[Tensor, ...], pe: Tensor) -> Tensor:
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
    y = _rms_adaln(x, block.norm1.weight, scale_a, shift_a)
    q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)
    out = _attention(q, k, v, block.q_norm.weight, block.k_norm.weight, pe, block.heads)
    x = x.addcmul_(block.proj["o"](out), gate_a)
    y = _rms_adaln(x, block.norm2.weight, scale_f, shift_f)
    hidden, gate = block.proj["ffn_up_gate"](y).chunk(2, dim=-1)
    return x.addcmul_(block.proj["ffn_down"](_situ(hidden, gate)), gate_f)


@torch.no_grad()
def comfy_forward(model, x: Tensor, timestep: Tensor, context: Tensor, *,
                  skip_layers=()) -> Tensor:
    """ComfyUI ``MOSSImage._forward``。

    ``x``：``[B, 128, H/16, W/16]``（ComfyUI 的 patchify 后 latent 排布，已转 compute
    dtype）；``timestep``：σ（fp32 ``[B]``）；``context``：``[B, S, text_dim]``。
    ``skip_layers``：SkipLayerGuidanceDiTSimple 跳过的层（该层原样传递）。返回同 dtype
    的速度预测 ``[B, 128, H/16, W/16]``。
    """
    b, _, h, w = x.shape
    dtype, device = x.dtype, x.device
    heads = model.config.num_heads
    pe = _rope(model.config.dim // heads, h, w, context.shape[1], device)

    vec = model.time_embedding(_timestep_embedding(timestep, model.config.freq_dim).to(dtype))
    mod = model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    head_shift, head_scale = model.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

    txt = model.text_embedder.proj(context)
    for block in model.text_embedder.blocks:
        txt = _text_block(block, txt, pe[:, h * w:])
    hidden = torch.cat((model.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
    del txt

    skip = frozenset(skip_layers or ())
    for index, block in enumerate(model.layers):
        if index in skip:
            continue
        hidden = _dit_block(block, hidden, mod, pe)

    out = model.final_linear_video(
        _rms_adaln(hidden[:, :h * w], model.final_norm_video.weight, head_scale, head_shift)
    )
    return out.transpose(1, 2).reshape(b, model.config.latent_channels, h, w)
