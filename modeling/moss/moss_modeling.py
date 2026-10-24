# Copyright OpenMOSS Team. Licensed under the Apache License, Version 2.0.
# Ported from OpenMOSS-Team/MOSS-Image `pipeline_moss_image.py` (HF, 2026-10).
# 模块路径保持上游命名（= checkpoint 键名 = kohya LoRA 键名），ComfyUI 的 MOSS
# LoRA 映射（comfy/lora.py）按这套路径工作。本项目改动见各处注释，均不改数值：
#   - 输入 / 输出为 FLUX.2 VAE 原生排布（32ch、f8，BN 已归一化），DiT 入口 2×2
#     patchify 成上游的 128ch、出口 unpatchify（latent_spaces.FLUX2_F8C32_BN）
#   - timestep 取 σ∈[0,1]，内部 ×1000（与 ComfyUI timestep_embedding time_factor
#     同口径；上游 pipeline 传入的已是 σ×1000）
#   - RoPE 频率每次前向现算，不做 buffer（上游的 `rope.bands` 是派生常量，加载侧丢弃）
#   - 可选逐 block 梯度检查点
#   - NaViT 打包前向 forward_packed（多样本拼一条序列、按段块对角；数学同逐样本 forward）
#   - 注意力可选 flash-attn 包 / xformers（modeling.attention，按模型实例设置）
# 见 THIRD_PARTY_NOTICES。
"""MOSS-Image 5B 单流 DiT（结构定义层，只依赖 torch 与 modeling.attention）。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from modeling.attention import (
    ATTENTION_BACKENDS,
    SDPA,
    XFORMERS,
    attention_block_diagonal_bshd,
    attention_bshd,
)

_PATCH = 2


@dataclass(frozen=True)
class MossConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    #: DiT 视角的通道数（patchify 后）；VAE 原生通道 = latent_channels // 4
    latent_channels: int = 128
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.dim % self.num_heads or (self.dim // self.num_heads) % 8 or self.freq_dim % 2:
            raise ValueError("MOSS 要求 dim 整除 heads、head_dim 为 8 的倍数、freq_dim 为偶数")
        if self.latent_channels % (_PATCH * _PATCH):
            raise ValueError("MOSS latent_channels 须为 4 的倍数（2×2 patchify）")

    @property
    def vae_channels(self) -> int:
        return self.latent_channels // (_PATCH * _PATCH)


def patchify(z: Tensor) -> Tensor:
    """``[b, c, h, w]`` → ``[b, c·4, h/2, w/2]``，通道序 ``c·4 + pi·2 + pj``（同 FLUX.2 VAE）。"""
    b, c, h, w = z.shape
    z = z.reshape(b, c, h // _PATCH, _PATCH, w // _PATCH, _PATCH)
    return z.permute(0, 1, 3, 5, 2, 4).reshape(b, c * _PATCH * _PATCH, h // _PATCH, w // _PATCH)


def unpatchify(z: Tensor) -> Tensor:
    """``patchify`` 的逆。"""
    b, c4, h, w = z.shape
    c = c4 // (_PATCH * _PATCH)
    z = z.reshape(b, c, _PATCH, _PATCH, h, w)
    return z.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * _PATCH, w * _PATCH)


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads, backend=SDPA, seqlens=None):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).to(v.dtype)
    k = _rotate(k.reshape(shape), rope).to(v.dtype)
    v = v.reshape(shape)
    if seqlens is None:
        return attention_bshd(q, k, v, backend, stage="moss").flatten(2)
    # NaViT 打包序列：每个样本段只注意自己（forward_packed）
    return attention_block_diagonal_bshd(q, k, v, seqlens, backend, stage="moss").flatten(2)


def _per_sample(m, seqlens):
    """调制分量 ``[B, D]`` → 可与 ``[B, S, D]`` 广播；打包序列（给了段长）按段展开到 token。"""
    if seqlens is None:
        return m[:, None]
    repeats = torch.tensor(seqlens, device=m.device)
    return m.repeat_interleave(repeats, 0, output_size=sum(seqlens)).unsqueeze(0)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.attention_backend = SDPA
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, seqlens=None):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, self.attention_backend, seqlens))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, seqlens=None):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, seqlens)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.attention_backend = SDPA
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, dim, bias=False) for name in ('q', 'k', 'v', 'o')
        })
        self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, seqlens=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
            _per_sample(m, seqlens) for m in modulation.unbind(1)
        )
        y = self.norm1(x) * (1 + scale_a) + shift_a
        # 上游取 proj['q'].weight.dtype；int8 存储时那是 int8，改取计算精度（norm 权重）
        dtype = self.norm1.weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        out = _attention(q, k, self.proj['v'](y), rope, self.heads, self.attention_backend, seqlens)
        x = x + self.proj['o'](out) * gate_a
        y = self.norm2(x) * (1 + scale_f) + shift_f
        hidden, gate = self.proj['ffn_up_gate'](y.to(dtype)).chunk(2, -1)
        return x + self.proj['ffn_down'](_situ(hidden, gate)) * gate_f


def _rope(head_dim: int, base: float, height: int, width: int, text_length: int, device) -> Tensor:
    """4D RoPE（r, f, h, w）。图像在前、文本在后；图像 f = 文本长度，h/w 以 0 为中心。

    频率按上游 `_RoPE._default_bands` 的 CPU fp32 构造（训练同口径），再搬到 device。
    """
    num_bands = head_dim // 8
    exponent = torch.arange(num_bands, device="cpu", dtype=torch.float32) / num_bands
    bands = (1 / (base ** exponent)).to(device)
    h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def _validate_skip_layers(layers, depth) -> frozenset[int]:
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f'skip_layers 须为 [0, {depth}) 内的整数 block 序号')
    return frozenset(layers)


class MossImageTransformer(nn.Module):
    """MOSS-Image DiT。每个样本按真实文本长度单独前向（上游语义：padding 不参与
    text refiner 与联合注意力，也不改变图像 token 的 RoPE f 坐标）。"""

    def __init__(self, config: MossConfig = MossConfig()):
        super().__init__()
        self.config = config
        dim, ffn_dim, heads = config.dim, config.ffn_dim, config.num_heads
        self.gradient_checkpointing = False
        self.attention_backend = SDPA
        self.video_embedder = nn.Linear(config.latent_channels, dim)
        self.text_embedder = _TextProjection(config.text_dim, dim, ffn_dim, heads)
        self.time_embedding = nn.Sequential(nn.Linear(config.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, heads) for _ in range(config.num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, config.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def set_attention_backend(self, backend: str) -> None:
        """设置本模型（DiT 层与文本精炼块）的注意力后端，取值见 ``modeling.attention``。"""
        if backend not in ATTENTION_BACKENDS:
            raise ValueError(f"未知的 attention backend: {backend!r}")
        self.attention_backend = backend
        for module in self.modules():
            if isinstance(module, (_Block, _TextBlock)):
                module.attention_backend = backend

    def enable_xformers_memory_efficient_attention(self) -> None:
        """``training.model_loading.enable_xformers`` 的钩子（diffusers 同名约定）。"""
        self.set_attention_backend(XFORMERS)

    def _time(self, sigmas: Tensor, batch: int, device) -> tuple[Tensor, Tensor]:
        times = (sigmas.to(device=device, dtype=torch.float32).reshape(-1) * 1000.0).expand(batch)
        # 上游：正弦嵌入在 float64 计算后转 float32 喂 MLP（cos 在前）
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        return mod, head

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        attention_mask: Tensor | None = None,
        *,
        use_checkpoint: bool = False,
        skip_layers=None,
    ) -> Tensor:
        """``x``: ``[B, 32, H, W]`` 或 ``[B, 32, 1, H, W]``（FLUX.2 VAE 排布）；
        ``timesteps``: ``[B]`` 或标量，σ∈[0,1]；``context``: ``[B, S, text_dim]``；
        ``attention_mask``: ``[B, S]`` 0/1。返回与 ``x`` 同形的速度预测（ε − x₀）。"""
        skip = _validate_skip_layers(skip_layers, len(self.layers))
        temporal = x.ndim == 5
        if temporal:
            if x.shape[2] != 1:
                raise ValueError("MOSS 只支持 T==1 的 5D latent")
            x = x.squeeze(2)
        if x.ndim != 4 or x.shape[1] != self.config.vae_channels:
            raise ValueError(f"MOSS latent 应为 [B, {self.config.vae_channels}, H, W]，实际 {tuple(x.shape)}")
        if x.shape[-2] % _PATCH or x.shape[-1] % _PATCH:
            raise ValueError("MOSS latent 高宽须为 2 的倍数（像素须为 16 的倍数）")
        batch = x.shape[0]
        if context.ndim != 3 or context.shape[0] != batch:
            raise ValueError("MOSS context 应为 [B, S, text_dim] 且 batch 与 latent 一致")
        mask = attention_mask
        if mask is None:
            mask = torch.ones(context.shape[:2], dtype=torch.bool, device=x.device)
        if mask.shape != context.shape[:2]:
            raise ValueError("MOSS attention_mask 应为 [B, S]")
        mask = mask.to(device=x.device).bool()

        hidden_states = patchify(x)
        height, width = hidden_states.shape[-2:]
        mod, head = self._time(torch.as_tensor(timesteps), batch, x.device)
        checkpoint_blocks = (
            (use_checkpoint or self.gradient_checkpointing)
            and self.training
            and torch.is_grad_enabled()
        )

        outputs = []
        for index in range(batch):
            text_ctx = context[index:index + 1, mask[index]]
            if text_ctx.shape[1] == 0:
                raise ValueError("MOSS 每个样本至少需要 1 个有效文本 token")
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            rope = _rope(self.config.dim // self.config.num_heads, self.config.rope_base,
                         height, width, text_ctx.shape[1], x.device).to(video.dtype)
            text = self.text_embedder(text_ctx, rope[:, height * width:])
            h = torch.cat((video, text), 1)
            sample_mod = mod[index:index + 1]
            for layer_index, layer in enumerate(self.layers):
                if layer_index in skip:
                    continue
                if checkpoint_blocks:
                    from torch.utils.checkpoint import checkpoint

                    h = checkpoint(layer, h, rope, sample_mod, use_reentrant=False)
                else:
                    h = layer(h, rope, sample_mod)
            h = self.final_norm_video(h[:, :height * width].to(self.final_norm_video.weight.dtype))
            h = h * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            h = self.final_linear_video(h.to(self.final_norm_video.weight.dtype))
            outputs.append(h.transpose(1, 2).reshape(1, -1, height, width))
        out = unpatchify(torch.cat(outputs))
        return out.unsqueeze(2) if temporal else out

    def forward_packed(self, latents, timesteps: Tensor, contexts, *,
                       use_checkpoint: bool = False) -> list[Tensor]:
        """NaViT 打包前向：G 张尺寸各异的图（各带自己的文本）拼成一条序列一次算完。

        每个样本段 = [图像 token, 文本 token]，RoPE 与 ``forward`` 的单样本序列相同；
        联合注意力与 text refiner 按段块对角，调制按段展开到 token，Linear 在整条序列
        上一次算。与逐样本 ``forward`` 数学等价，且每层只调用一次（block swap 每层每步
        只搬一次权重）。

        ``latents``: G 个 ``[32, H, W]`` / ``[1, 32, H, W]`` / ``[1, 32, 1, H, W]``，H/W 可各不相同；
        ``timesteps``: ``[G]``，σ∈[0,1]；``contexts``: G 个 ``[L, text_dim]`` 或
        ``[1, L, text_dim]``（只含有效 token）。返回 G 个与对应输入同形的速度预测。
        """
        count = len(latents)
        if count == 0 or len(contexts) != count:
            raise ValueError("MOSS forward_packed 需要非空且一一对应的 latents / contexts")
        timesteps = torch.as_tensor(timesteps).reshape(-1)
        if timesteps.numel() != count:
            raise ValueError(f"MOSS forward_packed 需要 {count} 个 timestep，实际 {timesteps.numel()}")
        head_dim = self.config.dim // self.config.num_heads
        device = latents[0].device

        shapes, grids, videos, texts, ropes = [], [], [], [], []
        for latent, context in zip(latents, contexts):
            shapes.append(latent.shape)
            z = latent.reshape(1, *latent.shape[-3:]) if latent.ndim == 3 else latent
            if z.ndim == 5:
                if z.shape[2] != 1:
                    raise ValueError("MOSS 只支持 T==1 的 5D latent")
                z = z.squeeze(2)
            if z.ndim != 4 or z.shape[:2] != (1, self.config.vae_channels):
                raise ValueError(f"MOSS 打包 latent 应为单样本 [1, {self.config.vae_channels}, H, W]，实际 {tuple(latent.shape)}")
            if z.shape[-2] % _PATCH or z.shape[-1] % _PATCH:
                raise ValueError("MOSS latent 高宽须为 2 的倍数（像素须为 16 的倍数）")
            context = context.reshape(1, -1, context.shape[-1])
            if context.shape[1] == 0:
                raise ValueError("MOSS 每个样本至少需要 1 个有效文本 token")
            hidden = patchify(z)
            height, width = hidden.shape[-2:]
            grids.append((height, width))
            videos.append(hidden.flatten(2).transpose(1, 2))
            texts.append(context)
            ropes.append(_rope(head_dim, self.config.rope_base, height, width, context.shape[1], device))

        image_lens = [h * w for h, w in grids]
        text_lens = [t.shape[1] for t in texts]
        video = self.video_embedder(torch.cat(videos, 1).to(self.video_embedder.weight.dtype).contiguous())
        rope = torch.cat(ropes, 1).to(video.dtype)
        text_rope = torch.cat([r[:, n:] for r, n in zip(ropes, image_lens)], 1).to(video.dtype)
        text = self.text_embedder(torch.cat(texts, 1), text_rope, text_lens)
        h = torch.cat([
            part
            for pair in zip(video.split(image_lens, 1), text.split(text_lens, 1))
            for part in pair
        ], 1)
        seqlens = tuple(n + t for n, t in zip(image_lens, text_lens))

        mod, head = self._time(timesteps, count, device)
        checkpoint_blocks = (
            (use_checkpoint or self.gradient_checkpointing)
            and self.training
            and torch.is_grad_enabled()
        )
        for layer in self.layers:
            if checkpoint_blocks:
                from torch.utils.checkpoint import checkpoint

                h = checkpoint(layer, h, rope, mod, seqlens, use_reentrant=False)
            else:
                h = layer(h, rope, mod, seqlens)

        image = torch.cat([seg[:, :n] for seg, n in zip(h.split(seqlens, 1), image_lens)], 1)
        image = self.final_norm_video(image.to(self.final_norm_video.weight.dtype))
        image = image * (1 + _per_sample(head[:, 1], image_lens)) + _per_sample(head[:, 0], image_lens)
        image = self.final_linear_video(image.to(self.final_norm_video.weight.dtype))
        outputs = []
        for tokens, (height, width), shape in zip(image.split(image_lens, 1), grids, shapes):
            out = unpatchify(tokens.transpose(1, 2).reshape(1, -1, height, width))
            outputs.append(out.reshape(shape))
        return outputs
