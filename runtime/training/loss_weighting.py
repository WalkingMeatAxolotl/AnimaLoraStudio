"""Loss 加权方案：min_snr / detail_inv_t / cosmap 等。

抽自原 runtime/anima_train.py L1898-1937（ADR 0003 PR-A）。

各 scheme 共享同一签名 (t, gamma, cap) -> (B,)，不引入 schema 字段（用
schema.loss_weighting: str），所以一文件多 if 分支即可，不需要 plugin
subfolder。
"""

from __future__ import annotations

import math

import torch


def compute_loss_weight(
    t: torch.Tensor,
    scheme: str = "none",
    min_snr_gamma: float = 5.0,
    weight_cap_ratio: float = 0.0,
) -> torch.Tensor:
    """返回每样本 loss 权重 (B,)，Flow Matching CONST 调度下：SNR(t) = ((1-t)/t)^2。

    scheme:
      none          — 全 1，与原始行为一致
      min_snr       — w = min(gamma/SNR, 1)，下调高 SNR 简单步（推荐基础款）
      detail_inv_t  — w = 1/t clamp [1,5]，温和细节强化，小 batch + Prodigy 友好
      cosmap        — SD3 cosmap weighting，中间 t 更均匀（max/min ≈ 1.81×）

    weight_cap_ratio — batch 内 max/min 比上限（0=禁用），防单样本主导破坏 Prodigy d 估计
    """
    scheme = (scheme or "none").lower()
    if scheme == "none":
        return torch.ones_like(t)

    eps = 1e-4
    t_c = t.clamp(eps, 1 - eps)

    if scheme == "min_snr":
        snr = ((1 - t_c) / t_c) ** 2
        w = torch.minimum(torch.tensor(float(min_snr_gamma), device=t.device) / snr, torch.ones_like(t_c))
    elif scheme == "detail_inv_t":
        w = (1.0 / t_c).clamp(min=1.0, max=5.0)
    elif scheme == "cosmap":
        bot = (1 - 2 * t_c + 2 * t_c ** 2).clamp(min=eps)
        w = 2.0 / (math.pi * bot)
    else:
        return torch.ones_like(t)

    if weight_cap_ratio and weight_cap_ratio > 1.0:
        w_min = w.min().clamp(min=eps)
        w = w.clamp(max=w_min * float(weight_cap_ratio))

    return w
