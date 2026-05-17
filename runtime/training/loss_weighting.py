"""Loss 加权方案：min_snr / detail_inv_t / cosmap 等。

抽自原 runtime/anima_train.py L1898-1937（ADR 0003 PR-A）。

各 scheme 共享 compute_loss_weight 入口；scheme-specific 参数（如
detail_inv_t_min/max）通过 keyword arg 透传。当前 scheme 总数较少仍用
单文件多 if 分支；若未来 scheme 数量增长 / scheme-specific 字段进一步
扩张，应迁移到 plugin 模式（参考 runtime/training/losses/）。
"""

from __future__ import annotations

import math

import torch


def compute_loss_weight(
    t: torch.Tensor,
    scheme: str = "none",
    min_snr_gamma: float = 5.0,
    weight_cap_ratio: float = 0.0,
    detail_inv_t_min: float = 1.0,
    detail_inv_t_max: float = 5.0,
) -> torch.Tensor:
    """返回每样本 loss 权重 (B,)，Flow Matching CONST 调度下：SNR(t) = ((1-t)/t)^2。

    scheme:
      none          — 全 1，与原始行为一致
      min_snr       — w = min(gamma/SNR, 1)，下调高 SNR 简单步（推荐基础款）
      detail_inv_t  — w = 1/t clamp [detail_inv_t_min, detail_inv_t_max]，
                      温和细节强化，小 batch + Prodigy 友好；默认 [1,5]
      cosmap        — SD3 cosmap weighting，中间 t 更均匀（max/min ≈ 1.81×）

    weight_cap_ratio — batch 内 max/min 比上限（0=禁用），防单样本主导破坏 Prodigy d 估计
    detail_inv_t_min/max — detail_inv_t 加权曲线的下/上限。
                           默认 [1, 5]：t=1 时 w=1，t=0.2 时 w=5（被 clamp）。
                           降 max（如 3）→ 雾蒙蒙/低饱和画风更稳；
                           升 max（如 8）→ 细节学得更激进但 Prodigy d 估计易被单样本主导。
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
        lo = float(detail_inv_t_min)
        hi = float(detail_inv_t_max)
        if lo > hi:
            lo, hi = hi, lo
        w = (1.0 / t_c).clamp(min=lo, max=hi)
    elif scheme == "cosmap":
        bot = (1 - 2 * t_c + 2 * t_c ** 2).clamp(min=eps)
        w = 2.0 / (math.pi * bot)
    else:
        return torch.ones_like(t)

    if weight_cap_ratio and weight_cap_ratio > 1.0:
        w_min = w.min().clamp(min=eps)
        w = w.clamp(max=w_min * float(weight_cap_ratio))

    return w
