"""CosineAnnealingLR scheduler build wrapper（ADR 0003 PR-C）。"""

from __future__ import annotations

import logging
from typing import Optional


logger = logging.getLogger(__name__)


def build(args, optimizer, total_steps: Optional[int]):
    """实例化 CosineAnnealingLR。total_steps 未知时记 warn + 返回 None
    （回退到 no-scheduler）—— 跟原 main() 老逻辑等价。"""
    if total_steps is None:
        logger.warning("cosine 调度器需要已知 total_steps，回退到 none")
        return None
    import torch

    eta_min = float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=eta_min,
    )
    logger.info(f"学习率调度: cosine (T_max={total_steps}, eta_min={eta_min})")
    return scheduler
