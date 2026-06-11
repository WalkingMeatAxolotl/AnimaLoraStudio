"""Standalone ComfyUI-style Qwen3 0.6B text encoder for Anima parity.

This is a narrow implementation of the text encoder path used by ComfyUI's
``comfy.text_encoders.anima.Qwen3_06B``. It intentionally covers only the
Anima/Qwen3-0.6B architecture needed by test-generation parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if q.device.type == "cuda" and q.nelement() >= 1024 * 128:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            priority = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
            with sdpa_kernel(priority, set_priority=True):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
                )
        except Exception:
            pass
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)


@dataclass
class Qwen3_06BConfig:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    head_dim: int = 128
    qkv_bias: bool = False
    q_norm: str = "gemma3"
    k_norm: str = "gemma3"
    final_norm: bool = True


class CastLinear(nn.Linear):
    """Linear with Comfy ``manual_cast`` semantics for text encoders."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


class CastEmbedding(nn.Embedding):
    def forward(self, input_ids: torch.Tensor, *, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        weight = self.weight.to(device=input_ids.device)
        return F.embedding(
            input_ids,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        ).to(dtype=out_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * weight


def _precompute_freqs_cis(
    head_dim: int,
    position_ids: torch.Tensor,
    theta: float,
    *,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (float(theta) ** (theta_numerator / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)
    split = sin.shape[-1] // 2
    return cos, sin[..., :split], -sin[..., split:]


def _apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    org_dtype = xq.dtype
    cos, sin, nsin = freqs_cis

    q_embed = xq * cos
    q_split = q_embed.shape[-1] // 2
    q_embed[..., :q_split].addcmul_(xq[..., q_split:], nsin)
    q_embed[..., q_split:].addcmul_(xq[..., :q_split], sin)

    k_embed = xk * cos
    k_split = k_embed.shape[-1] // 2
    k_embed[..., :k_split].addcmul_(xk[..., k_split:], nsin)
    k_embed[..., k_split:].addcmul_(xk[..., :k_split], sin)
    return q_embed.to(org_dtype), k_embed.to(org_dtype)


class Attention(nn.Module):
    def __init__(self, config: Qwen3_06BConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim

        self.q_proj = CastLinear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = CastLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = CastLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = CastLinear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq, xk = _apply_rope(xq, xk, freqs_cis)

        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        output = _scaled_dot_product_attention(xq, xk, xv, attn_mask=attention_mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.inner_size)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: Qwen3_06BConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.gate_proj = CastLinear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = CastLinear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = CastLinear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3_06BConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype)
        self.mlp = MLP(config, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask=attention_mask, freqs_cis=freqs_cis)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class Qwen3_06BBackbone(nn.Module):
    def __init__(self, config: Qwen3_06BConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = CastEmbedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids, out_dtype=torch.float32)
        batch, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        freqs_cis = _precompute_freqs_cis(
            self.config.head_dim,
            position_ids,
            self.config.rope_theta,
            device=x.device,
        )

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((batch, 1, 1, -1)).expand(batch, 1, seq_len, -1)
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

        if seq_len > 1:
            causal_mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device)
            causal_mask.fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            mask = causal_mask if mask is None else mask + causal_mask

        for layer in self.layers:
            x = layer(x, attention_mask=mask, freqs_cis=freqs_cis)
        return self.norm(x)


class ComfyQwen3Encoder(nn.Module):
    """HF-like wrapper around the Comfy-style Qwen backbone."""

    uses_comfy_clip_masking = True

    def __init__(self, *, device=None, dtype=torch.float16) -> None:
        super().__init__()
        self.model = Qwen3_06BBackbone(Qwen3_06BConfig(), device=device, dtype=dtype)
        self.num_layers = self.model.config.num_hidden_layers

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        use_cache: bool = False,
        **_kwargs,
    ):
        hidden = self.model(input_ids, attention_mask=attention_mask)
        if return_dict:
            return SimpleNamespace(last_hidden_state=hidden, hidden_states=[hidden] if output_hidden_states else None)
        return (hidden,)


def _resolve_qwen_safetensors_path(qwen_path: str | Path) -> Path:
    path = Path(qwen_path)
    if path.is_dir():
        candidate = path / "model.safetensors"
        if candidate.exists():
            return candidate
    return path


def load_comfy_qwen3_encoder(
    qwen_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float16,
) -> ComfyQwen3Encoder:
    from safetensors.torch import load_file

    ckpt_path = _resolve_qwen_safetensors_path(qwen_path)
    model = ComfyQwen3Encoder(device=device, dtype=dtype)
    state_dict = load_file(str(ckpt_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval().requires_grad_(False)
    return model
