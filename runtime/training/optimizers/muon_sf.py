"""Schedule-Free Muon optimizer build wrapper (ADR 0003 PR-C).

Schedule-Free trajectory + Newton-Schulz orthogonalization.
Built-in LR scheduling (no external scheduler); exposes train()/eval().
"""

from __future__ import annotations


def validate(args) -> None:
    """lr_scheduler must be none for Schedule-Free optimizers."""
    lr_sched_cfg = (getattr(args, "lr_scheduler", "none") or "none").lower()
    if lr_sched_cfg != "none":
        raise SystemExit(
            f"muon_sf (Schedule-Free Muon) requires lr_scheduler=none "
            f"(Schedule-Free is scheduler-free by construction); got "
            f"lr_scheduler={lr_sched_cfg!r}. Set lr_scheduler=none or pick a "
            f"different optimizer."
        )


def build(args, params, lr: float, weight_decay: float):
    """Instantiate MuonScheduleFree, reading muon_* / muon_sf_* parameters."""
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="muon_sf",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "muon_beta1", 0.9)),
            float(getattr(args, "muon_beta2", 0.95)),
        ),
        ns_steps=int(getattr(args, "muon_ns_steps", 5)),
        weight_lr_power=float(getattr(args, "muon_sf_weight_lr_power", 2.0)),
        r=float(getattr(args, "muon_sf_r", 0.0)),
        warmup_steps=int(getattr(args, "muon_sf_warmup_steps", 0)),
        correct_bias=bool(getattr(args, "muon_correct_bias", True)),
    )
