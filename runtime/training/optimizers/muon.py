"""Muon optimizer build wrapper (ADR 0003 PR-C).

Newton-Schulz orthogonalized momentum optimizer for 2D params,
AdamW fallback for 1D. Zero preconditioner memory (no GG/Q matrices).
"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    """Instantiate Muon, reading muon_* parameters."""
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="muon",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "muon_beta1", 0.9)),
            float(getattr(args, "muon_beta2", 0.999)),
        ),
        momentum=float(getattr(args, "muon_momentum", 0.95)),
        nesterov=bool(getattr(args, "muon_nesterov", True)),
        ns_steps=int(getattr(args, "muon_ns_steps", 5)),
        correct_bias=bool(getattr(args, "muon_correct_bias", True)),
    )
