"""Prodigy optimizer build wrapper（ADR 0003 PR-C）。"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    """实例化 Prodigy，读 args.prodigy_d_coef + args.prodigy_safeguard_warmup。"""
    from utils.optimizer_utils import create_optimizer
    return create_optimizer(
        optimizer_type="prodigy",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        d_coef=float(getattr(args, "prodigy_d_coef", 1.0)),
        safeguard_warmup=bool(getattr(args, "prodigy_safeguard_warmup", True)),
    )
