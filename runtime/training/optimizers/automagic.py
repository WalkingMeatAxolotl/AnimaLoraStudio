"""Automagic optimizer build wrapper."""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="automagic",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        min_lr=float(getattr(args, "automagic_min_lr", 1e-7)),
        max_lr=float(getattr(args, "automagic_max_lr", 1e-3)),
        lr_bump=float(getattr(args, "automagic_lr_bump", 1e-6)),
        eps=float(getattr(args, "automagic_eps", 1e-30)),
        clip_threshold=float(getattr(args, "automagic_clip_threshold", 1.0)),
        beta2=float(getattr(args, "automagic_beta2", 0.999)),
    )


def validate(args) -> None:
    scheduler = (getattr(args, "lr_scheduler", "none") or "none").lower()
    if scheduler != "none":
        raise ValueError("Automagic manages learning rates internally and requires lr_scheduler=none")
