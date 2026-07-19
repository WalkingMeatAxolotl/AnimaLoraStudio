"""Krea 2 single-stream MMDiT with ComfyUI-compatible module names.

Core structure derived from ComfyUI ``comfy/ldm/krea2/model.py`` (GPL-3.0):
Copyright Comfy-Org and ComfyUI contributors.
https://github.com/comfyanonymous/ComfyUI/blob/87d23b81765161624889febfb3b81f19f3c8435b/comfy/ldm/krea2/model.py

Training-oriented tensor layout, RoPE, and checkpointing were adapted from
kohya-ss/musubi-tuner ``krea2_mmdit.py`` (Apache-2.0):
Copyright 2026 Kohya S. and musubi-tuner contributors.
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2/krea2_mmdit.py

Pinned source revisions and file URLs are recorded in ``THIRD_PARTY_NOTICES.md``.
This file is distributed under the repository's GPL-3.0 license.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def _rope(pos: Tensor, dim: int, theta: float) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    angles = torch.einsum("...n,d->...nd", pos, omega)
    matrix = torch.stack(
        [torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)],
        dim=-1,
    )
    return rearrange(matrix, "b n d (i j) -> b n d i j", i=2, j=2).float()


def _apply_rope(q: Tensor, k: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    q_float = q.float().reshape(*q.shape[:-1], -1, 1, 2)
    k_float = k.float().reshape(*k.shape[:-1], -1, 1, 2)
    matrix = freqs[:, None, :, :, :]
    q_out = matrix[..., 0] * q_float[..., 0] + matrix[..., 1] * q_float[..., 1]
    k_out = matrix[..., 0] * k_float[..., 0] + matrix[..., 1] * k_float[..., 1]
    return q_out.reshape_as(q).to(q.dtype), k_out.reshape_as(k).to(k.dtype)


def _timestep_embedding(t: Tensor, dim: int, *, dtype: torch.dtype) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(1e4)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    angles = (t.float() * 1e3)[:, None, None] * freqs
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).to(dtype)


@dataclass(frozen=True)
class Krea2Config:
    """Architecture values for the public Krea-2 Raw/Turbo checkpoints."""

    features: int = 6144
    tdim: int = 256
    txtdim: int = 2560
    heads: int = 48
    multiplier: int = 4
    layers: int = 28
    patch: int = 2
    channels: int = 16
    bias: bool = False
    theta: float = 1e3
    kvheads: int = 12
    txtlayers: int = 12
    txtheads: int = 20
    txtkvheads: int = 20

    def __post_init__(self) -> None:
        if self.features <= 0 or self.features % self.heads:
            raise ValueError("Krea2 features 必须为正数且能被 heads 整除")
        if self.heads % self.kvheads:
            raise ValueError("Krea2 heads 必须能被 kvheads 整除")
        if self.txtdim <= 0 or self.txtdim % self.txtheads:
            raise ValueError("Krea2 txtdim 必须为正数且能被 txtheads 整除")
        if self.txtheads % self.txtkvheads:
            raise ValueError("Krea2 txtheads 必须能被 txtkvheads 整除")
        if self.tdim <= 0 or self.tdim % 2:
            raise ValueError("Krea2 tdim 必须为正偶数")
        if min(self.layers, self.patch, self.channels, self.txtlayers) <= 0:
            raise ValueError("Krea2 layers/patch/channels/txtlayers 必须为正数")
        if self.theta <= 0:
            raise ValueError("Krea2 theta 必须为正数")


KREA2_CONFIG = Krea2Config()


class RMSNorm(nn.Module):
    """RMSNorm using Krea2's zero-centered ``1 + scale`` convention."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        weight = self.scale.float() + 1.0
        return F.rms_norm(
            x.float(),
            (self.features,),
            weight=weight,
            eps=self.eps,
        ).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(
        self,
        features: int,
        multiplier: int,
        bias: bool = False,
        multiple: int = 128,
    ):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kvheads: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.kvheads = heads if kvheads is None else kvheads
        if dim % heads or heads % self.kvheads:
            raise ValueError("Krea2 attention 维度或 GQA heads 不可整除")
        self.headdim = dim // heads
        self.wq = nn.Linear(dim, self.headdim * heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        freqs: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        q = rearrange(self.wq(x), "b l (h d) -> b h l d", h=self.heads)
        k = rearrange(self.wk(x), "b l (h d) -> b h l d", h=self.kvheads)
        v = rearrange(self.wv(x), "b l (h d) -> b h l d", h=self.kvheads)
        gate = self.gate(x)

        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = _apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            repeat = self.heads // self.kvheads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.wo(out * torch.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor]:
        return (vec + self.lin.unsqueeze(0)).chunk(2, dim=1)


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor) -> tuple[Tensor, ...]:
        return (vec + self.lin).chunk(6, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, axes_dim: tuple[int, int, int], theta: float):
        super().__init__()
        self.axes_dim = axes_dim
        self.theta = theta

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [_rope(pos[..., axis], dim, self.theta) for axis, dim in enumerate(self.axes_dim)],
            dim=-3,
        )


class TextFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int | None = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        return x + self.mlp(self.postnorm(x))


class TextFusionTransformer(nn.Module):
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int | None = None,
    ):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, seq_len, num_layers, dim = x.shape
        x = x.reshape(batch * seq_len, num_layers, dim)
        for block in self.layerwise_blocks:
            x = block(x.contiguous())
        x = rearrange(x, "(b l) n d -> b l d n", b=batch, l=seq_len)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int | None = None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        attn_in = (1 + prescale) * self.prenorm(x) + preshift
        x = x + pregate * self.attn(attn_in, freqs=freqs, mask=mask)
        mlp_in = (1 + postscale) * self.postnorm(x) + postshift
        return x + postgate * self.mlp(mlp_in)


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        return self.linear((1 + scale) * self.norm(x) + shift)


class SingleStreamDiT(nn.Module):
    """Krea2 DiT; parameter paths intentionally match ComfyUI exactly."""

    def __init__(self, config: Krea2Config = KREA2_CONFIG):
        super().__init__()
        self.config = config
        head_dim = config.features // config.heads
        axes = (
            head_dim - 12 * (head_dim // 16),
            6 * (head_dim // 16),
            6 * (head_dim // 16),
        )
        if sum(axes) != head_dim or any(dim <= 0 or dim % 2 for dim in axes):
            raise ValueError(f"Krea2 RoPE axes 非法：axes={axes}, head_dim={head_dim}")

        self.posemb = PositionalEncoding(axes, theta=config.theta)
        self.first = nn.Linear(
            config.channels * config.patch**2,
            config.features,
            bias=True,
        )
        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features,
                    config.heads,
                    config.multiplier,
                    config.bias,
                    config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features * 6),
        )
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _normalize_context(self, context: Tensor) -> Tensor:
        config = self.config
        if context.ndim == 3:
            expected = config.txtlayers * config.txtdim
            if context.shape[-1] != expected:
                raise ValueError(
                    f"Krea2 context 最后一维应为 {expected}，实际 {context.shape[-1]}"
                )
            return context.reshape(
                context.shape[0],
                context.shape[1],
                config.txtlayers,
                config.txtdim,
            )
        if context.ndim != 4 or context.shape[-2:] != (
            config.txtlayers,
            config.txtdim,
        ):
            raise ValueError(
                "Krea2 context 应为 (B,L,txtlayers,txtdim) 或对应的扁平三维 tensor"
            )
        return context

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        attention_mask: Tensor | None = None,
        *,
        use_checkpoint: bool = False,
    ) -> Tensor:
        temporal = x.ndim == 5
        if temporal:
            if x.shape[2] != 1:
                raise ValueError("Krea2 v1 只支持 T==1 的 5D latent")
            x = x.squeeze(2)
        if x.ndim != 4:
            raise ValueError("Krea2 latent 应为 (B,C,H,W) 或 (B,C,1,H,W)")
        if x.shape[1] != self.config.channels:
            raise ValueError(
                f"Krea2 latent channels 应为 {self.config.channels}，实际 {x.shape[1]}"
            )

        context = self._normalize_context(context)
        batch, _, original_h, original_w = x.shape
        if context.shape[0] != batch:
            raise ValueError("Krea2 latent 与 context batch size 不一致")
        if timesteps.ndim != 1 or timesteps.numel() != batch:
            raise ValueError("Krea2 timesteps 应为 batch 长度的一维 tensor")

        patch = self.config.patch
        pad_h = (-original_h) % patch
        pad_w = (-original_w) % patch
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        height, width = x.shape[-2] // patch, x.shape[-1] // patch

        image = rearrange(
            x,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=patch,
            pw=patch,
        )
        image = self.first(image)
        time = self.tmlp(
            _timestep_embedding(
                timesteps,
                self.config.tdim,
                dtype=image.dtype,
            )
        )
        time_mod = self.tproj(time)

        text_len = context.shape[1]
        text_mask = None
        combined_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, text_len):
                raise ValueError("Krea2 attention_mask 应为 (B,text_len)")
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            text_mask = attention_mask[:, None, None, :]
            image_mask = torch.ones(
                batch,
                image.shape[1],
                device=x.device,
                dtype=torch.bool,
            )
            combined_mask = torch.cat((attention_mask, image_mask), dim=1)
            combined_mask = combined_mask[:, None, None, :]

        text = self.txtfusion(context, mask=text_mask)
        text = self.txtmlp(text)
        image_len = image.shape[1]
        combined = torch.cat((text, image), dim=1)

        text_pos = torch.zeros(
            batch,
            text_len,
            3,
            device=x.device,
            dtype=torch.float32,
        )
        image_pos = torch.zeros(
            height,
            width,
            3,
            device=x.device,
            dtype=torch.float32,
        )
        image_pos[..., 1] = torch.arange(height, device=x.device)[:, None]
        image_pos[..., 2] = torch.arange(width, device=x.device)[None, :]
        image_pos = image_pos.reshape(1, image_len, 3).expand(batch, -1, -1)
        freqs = self.posemb(torch.cat((text_pos, image_pos), dim=1))

        checkpoint_blocks = (
            (use_checkpoint or self.gradient_checkpointing)
            and self.training
            and torch.is_grad_enabled()
        )
        for block in self.blocks:
            if checkpoint_blocks:
                from torch.utils.checkpoint import checkpoint

                def custom_forward(hidden: Tensor, current=block) -> Tensor:
                    return current(hidden, time_mod, freqs, combined_mask)

                combined = checkpoint(custom_forward, combined, use_reentrant=False)
            else:
                combined = block(combined, time_mod, freqs, combined_mask)

        final = self.last(combined, time)
        output = final[:, text_len:text_len + image_len]
        output = rearrange(
            output,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=height,
            w=width,
            ph=patch,
            pw=patch,
            c=self.config.channels,
        )
        output = output[:, :, :original_h, :original_w]
        return output.unsqueeze(2) if temporal else output
