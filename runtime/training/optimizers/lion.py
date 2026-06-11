"""Lion optimizer build wrapper（ADR 0003 PR-C）。"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="lion",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "lion_beta1", 0.9)),
            float(getattr(args, "lion_beta2", 0.99)),
        ),
    )
