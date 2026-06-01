"""CosineAnnealingWarmRestarts scheduler build wrapper（ADR 0003 PR-C）。"""

from __future__ import annotations

import logging
from typing import Optional


logger = logging.getLogger(__name__)


def build(args, optimizer, total_steps: Optional[int]):
    """实例化 CosineAnnealingWarmRestarts。total_steps 未传也不阻断，
    跟原 main() 老逻辑等价。"""
    import torch

    t0 = int(getattr(args, "lr_scheduler_t0", 500) or 500)
    t_mult = int(getattr(args, "lr_scheduler_t_mult", 2) or 2)
    eta_min = float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min,
    )
    logger.info(f"学习率调度: cosine_with_restart (T_0={t0}, T_mult={t_mult}, eta_min={eta_min})")
    return scheduler
