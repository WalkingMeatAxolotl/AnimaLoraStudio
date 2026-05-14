"""Flow Matching timestep 采样：logit_normal / uniform / mode 等模式。

抽自原 runtime/anima_train.py L1817-1846（ADR 0003 PR-A）。

ADR 0003 把不同 mode 放同一文件（一文件多 fn）—— 每个 mode 是一行 sigmoid /
shift / clamp 数学，不引入 schema 字段（用 schema.timestep_sampling: str），
不需要 plugin subfolder。
"""

from __future__ import annotations

import torch


def sample_t(bs, device, mode: str = "logit_normal", shift: float = 3.0) -> torch.Tensor:
    """采样 Flow Matching 时间步 t ∈ (0, 1)。

    mode:
      logit_normal      — SD3/Anima 默认，偏向中间 t；shift>1 推向高噪声端
      uniform           — 均匀采样，对细节端和结构端覆盖更均衡
      logit_normal_low  — logit-normal 反向 shift，偏向低噪声/细节端
      mode              — SD3 mode-distribution，集中在某个 sigma 附近
    """
    mode = (mode or "logit_normal").lower()
    u = torch.sigmoid(torch.randn(bs, device=device))

    if mode == "uniform":
        return torch.rand(bs, device=device).clamp(1e-4, 1 - 1e-4)

    if mode == "logit_normal_low":
        s = max(float(shift), 1e-4)
        u = (u * (1.0 / s)) / (1 + (1.0 / s - 1) * u)
        return u.clamp(1e-4, 1 - 1e-4)

    if mode == "mode":
        s = float(shift)
        u = 1 - u - s * (torch.cos(torch.pi * 0.5 * u) ** 2 - 1 + u)
        return u.clamp(1e-4, 1 - 1e-4)

    # logit_normal（默认）+ shift
    s = float(shift)
    u = (u * s) / (1 + (s - 1) * u)
    return u.clamp(1e-4, 1 - 1e-4)
