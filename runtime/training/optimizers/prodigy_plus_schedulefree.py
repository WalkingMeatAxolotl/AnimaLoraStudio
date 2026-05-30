"""ProdigyPlusScheduleFree optimizer build wrapper（ADR 0003 PR-C）。

PPSF 是 Schedule-Free 系：内置 LR 调度，跟外部 LR scheduler 互斥。
本模块同时暴露 validate(args) 给 PR-C registry 在启动期跑兼容性检查。
"""

from __future__ import annotations


def validate(args) -> None:
    """启动期兼容性检查：lr_scheduler 必须 none，否则直接 SystemExit。"""
    lr_sched_cfg = (getattr(args, "lr_scheduler", "none") or "none").lower()
    if lr_sched_cfg != "none":
        raise SystemExit(
            f"ProdigyPlusScheduleFree requires lr_scheduler=none "
            f"(Schedule-Free is scheduler-free by construction); got "
            f"lr_scheduler={lr_sched_cfg!r}. Set lr_scheduler=none or pick a "
            f"different optimizer."
        )


def build(args, params, lr: float, weight_decay: float):
    """实例化 PPSF，读 7 个 ppsf_* 参数 + betas override。"""
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="prodigy_plus_schedulefree",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "ppsf_beta1", 0.9)),
            float(getattr(args, "ppsf_beta2", 0.99)),
        ),
        d_coef=float(getattr(args, "ppsf_d_coef", 1.0)),
        prodigy_steps=int(getattr(args, "ppsf_prodigy_steps", 0)),
        split_groups=bool(getattr(args, "ppsf_split_groups", True)),
        split_groups_mean=bool(getattr(args, "ppsf_split_groups_mean", False)),
        use_speed=bool(getattr(args, "ppsf_use_speed", False)),
        fused_back_pass=bool(getattr(args, "ppsf_fused_back_pass", False)),
        use_stableadamw=bool(getattr(args, "ppsf_use_stableadamw", True)),
    )
