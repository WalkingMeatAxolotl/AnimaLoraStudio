"""Schedule-Free SOAP optimizer build wrapper（ADR 0003 PR-C）。

SOAP 预条件 + Schedule-Free 轨迹（Defazio et al., 2024, arxiv 2405.15682）。
跟 PPSF 同属 schedule-free 系：内置 LR 调度，跟外部 lr_scheduler 互斥；暴露
train()/eval()，由 trainer 的 optimizer_eval_mode + state/resume 守护统一处理。
本模块同时暴露 validate(args) 给 PR-C registry 在启动期跑兼容性检查。
"""

from __future__ import annotations


def validate(args) -> None:
    """启动期兼容性检查：lr_scheduler 必须 none，否则直接 SystemExit。"""
    lr_sched_cfg = (getattr(args, "lr_scheduler", "none") or "none").lower()
    if lr_sched_cfg != "none":
        raise SystemExit(
            f"soap_sf (Schedule-Free SOAP) requires lr_scheduler=none "
            f"(Schedule-Free is scheduler-free by construction); got "
            f"lr_scheduler={lr_sched_cfg!r}. Set lr_scheduler=none or pick a "
            f"different optimizer."
        )


def build(args, params, lr: float, weight_decay: float):
    """实例化 SOAPScheduleFree，读 soap_* / soap_sf_* 参数。"""
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="soap_sf",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "soap_beta1", 0.9)),
            float(getattr(args, "soap_beta2", 0.95)),
        ),
        shampoo_beta=float(getattr(args, "soap_shampoo_beta", -1.0)),
        precondition_frequency=int(getattr(args, "soap_precondition_frequency", 10)),
        max_precond_dim=int(getattr(args, "soap_max_precond_dim", 10000)),
        precond_in_state=bool(getattr(args, "soap_precond_in_state", True)),
        weight_lr_power=float(getattr(args, "soap_sf_weight_lr_power", 2.0)),
        r=float(getattr(args, "soap_sf_r", 0.0)),
        warmup_steps=int(getattr(args, "soap_sf_warmup_steps", 0)),
    )
