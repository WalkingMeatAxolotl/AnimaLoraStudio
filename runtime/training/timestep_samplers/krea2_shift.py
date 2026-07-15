"""Krea2 resolution-aware training timestep sampler.

Derived formula: Copyright 2026 Kohya S. and musubi-tuner contributors.
Licensed under the Apache License, Version 2.0.

The dynamic-shift formula and Krea2 endpoints follow musubi-tuner's
``NetworkTrainer`` implementation and Krea2 timestep tests (Apache-2.0),
cross-checked against the Hugging Face diffusers Krea2 inference pipeline:

- https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/training/trainer_base.py
- https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/tests/test_krea2_timesteps.py
- https://github.com/huggingface/diffusers/blob/bc529a5f677db9c4b3fc72c76962c4e2f61567e1/src/diffusers/pipelines/krea2/pipeline_krea2.py

This is deliberately separate from ``apply_resolution_shift``: that helper is
the existing SD3 sqrt(token/base) correction, while Krea2 linearly interpolates
``mu`` from image sequence length and applies ``exp(mu)`` as a Möbius shift.
"""

from __future__ import annotations

import math

import torch


BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 6400
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15


def krea2_mu(token_counts, *, device=None) -> torch.Tensor:
    """Return Krea2's per-image dynamic-shift ``mu`` (without endpoint clamp)."""
    counts = torch.as_tensor(token_counts, dtype=torch.float32, device=device)
    if counts.ndim != 1:
        raise ValueError("Krea2 token_counts 必须是一维序列")
    if not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Krea2 token_counts 必须全部为有限正数")

    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    intercept = BASE_SHIFT - slope * BASE_IMAGE_SEQ_LEN
    return counts * slope + intercept


def apply_krea2_dynamic_shift(t: torch.Tensor, token_counts) -> torch.Tensor:
    """Apply Krea2's per-image ``exp(mu)`` Möbius shift to base timesteps."""
    if t.ndim != 1:
        raise ValueError("Krea2 timestep 必须是一维 tensor")
    mu = krea2_mu(token_counts, device=t.device)
    if mu.numel() != t.numel():
        raise ValueError(
            f"Krea2 token_counts 数量必须等于 batch size：{mu.numel()} != {t.numel()}"
        )
    shift = mu.exp().to(dtype=t.dtype)
    return ((t * shift) / (1 + (shift - 1) * t)).clamp(1e-4, 1 - 1e-4)


class Krea2ShiftTimestepSampler:
    """Logit-normal sampling followed by Krea2's per-resolution dynamic shift."""

    requires_token_counts = True
    applies_resolution_shift = True

    def __init__(self, sigmoid_scale: float = 1.0):
        self.sigmoid_scale = float(sigmoid_scale)
        if not math.isfinite(self.sigmoid_scale) or self.sigmoid_scale <= 0:
            raise ValueError("Krea2 sigmoid_scale 必须为有限正数")

    def sample(self, bs: int, device, *, token_counts=None) -> torch.Tensor:
        if token_counts is None:
            raise ValueError("Krea2 timestep sampler 需要每个样本的 token_counts")
        counts = torch.as_tensor(token_counts)
        if counts.ndim != 1 or counts.numel() != bs:
            raise ValueError(
                f"Krea2 token_counts 数量必须等于 batch size：{counts.numel()} != {bs}"
            )
        base_t = torch.randn(bs, device=device, dtype=torch.float32)
        base_t = torch.sigmoid(base_t * self.sigmoid_scale)
        return apply_krea2_dynamic_shift(base_t, token_counts)

    def record(self, t: torch.Tensor, raw_mse: torch.Tensor) -> None:
        return None

    def maybe_refresh(self, global_step: int) -> None:
        return None

    def status(self) -> dict:
        return {
            "kind": "krea2_shift",
            "sigmoid_scale": self.sigmoid_scale,
            "base_image_seq_len": BASE_IMAGE_SEQ_LEN,
            "max_image_seq_len": MAX_IMAGE_SEQ_LEN,
            "base_shift": BASE_SHIFT,
            "max_shift": MAX_SHIFT,
        }

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


def build(args, total_steps) -> Krea2ShiftTimestepSampler:
    """Build the Krea2 sampler; Phase 3 currently fixes upstream scale at 1.0."""
    return Krea2ShiftTimestepSampler(sigmoid_scale=1.0)
