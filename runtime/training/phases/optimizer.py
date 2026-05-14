"""optimizer_phase：optimizer dispatch + grad_clip + total_steps + lr_scheduler。

抽自 main() L344-437（ADR 0003 PR-B）。

注：optimizer dispatch 这次保留 if-elif 老风格（adamw / prodigy /
prodigy_plus_schedulefree），PR-C 会把它换成 plugin registry。
"""

from __future__ import annotations

import logging

import torch

from training.context import TrainingContext


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """
    - injector.get_param_groups + optimizer kwargs 派发 + create_optimizer
    - grad_clip / trainable_params
    - 计算 total_steps（min(by_epochs, by_max_steps)）
    - lr_scheduler 派发（cosine / cosine_with_restart / none）
    """
    args = ctx.args

    # 优化器
    ctx.weight_decay = float(getattr(args, "weight_decay", 0.01) or 0.0)
    param_groups = ctx.injector.get_param_groups(ctx.weight_decay)
    ctx.optimizer_type = (getattr(args, "optimizer_type", "adamw") or "adamw").lower()
    from utils.optimizer_utils import create_optimizer
    optimizer_extra: dict = {}
    optimizer_overrides: dict = {}
    if ctx.optimizer_type == "prodigy":
        optimizer_extra["d_coef"] = float(getattr(args, "prodigy_d_coef", 1.0))
        optimizer_extra["safeguard_warmup"] = bool(getattr(args, "prodigy_safeguard_warmup", True))
    elif ctx.optimizer_type == "prodigy_plus_schedulefree":
        # Schedule-Free 不需要 scheduler，启动期强校验
        lr_sched_cfg = (getattr(args, "lr_scheduler", "none") or "none").lower()
        if lr_sched_cfg != "none":
            raise SystemExit(
                f"ProdigyPlusScheduleFree requires lr_scheduler=none "
                f"(Schedule-Free is scheduler-free by construction); got "
                f"lr_scheduler={lr_sched_cfg!r}. Set lr_scheduler=none or pick a "
                f"different optimizer."
            )
        optimizer_extra["d_coef"] = float(getattr(args, "ppsf_d_coef", 1.0))
        optimizer_extra["prodigy_steps"] = int(getattr(args, "ppsf_prodigy_steps", 0))
        optimizer_extra["split_groups"] = bool(getattr(args, "ppsf_split_groups", True))
        optimizer_extra["split_groups_mean"] = bool(getattr(args, "ppsf_split_groups_mean", False))
        optimizer_extra["use_speed"] = bool(getattr(args, "ppsf_use_speed", False))
        optimizer_extra["fused_back_pass"] = bool(getattr(args, "ppsf_fused_back_pass", False))
        optimizer_extra["use_stableadamw"] = bool(getattr(args, "ppsf_use_stableadamw", True))
        optimizer_overrides["betas"] = (
            float(getattr(args, "ppsf_beta1", 0.9)),
            float(getattr(args, "ppsf_beta2", 0.99)),
        )
    ctx.optimizer = create_optimizer(
        optimizer_type=ctx.optimizer_type,
        params=param_groups,
        learning_rate=args.learning_rate,
        weight_decay=ctx.weight_decay,
        **optimizer_overrides,
        **optimizer_extra,
    )
    if ctx.weight_decay > 0:
        wd_info = f"{ctx.optimizer_type} weight_decay={ctx.weight_decay}"
        if ctx.injector.use_lokr:
            wd_info += "（w1 排除 weight_decay）"
        logger.info(wd_info)
    ctx.grad_clip = float(getattr(args, "grad_clip_max_norm", 0) or 0)
    if ctx.grad_clip > 0:
        logger.info(f"梯度裁剪 max_norm={ctx.grad_clip}")
    ctx.trainable_params = [p for group in ctx.optimizer.param_groups for p in group["params"]]

    # 计算总步数
    try:
        ctx.steps_per_epoch = len(ctx.dataloader) // args.grad_accum
    except Exception:
        ctx.steps_per_epoch = None

    # total_steps：训练实际会跑到的步数。终止条件是「epoch 上限和 max_steps
    # 哪个先到就停」(见下方 max_steps break + for epoch 自然退出)，所以
    # 取两个候选的 min，进度条才不会出现「100 epoch 跑完了但只显示 86%」。
    by_epochs = (
        ctx.steps_per_epoch * args.epochs
        if ctx.steps_per_epoch is not None and args.epochs and args.epochs > 0
        else None
    )
    by_max_steps = (
        args.max_steps if (args.max_steps and args.max_steps > 0) else None
    )
    candidates = [c for c in (by_epochs, by_max_steps) if c is not None and c > 0]
    ctx.total_steps = min(candidates) if candidates else None

    logger.info(
        f"数据集大小: {len(ctx.dataset)}, 每 epoch 步数: {ctx.steps_per_epoch}, "
        f"总步数: {ctx.total_steps} (by_epochs={by_epochs}, by_max_steps={by_max_steps})"
    )

    # 学习率调度器
    ctx.scheduler = None
    lr_sched = getattr(args, "lr_scheduler", "none") or "none"
    if lr_sched == "cosine":
        eta_min = float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0)
        if ctx.total_steps is None:
            logger.warning("cosine 调度器需要已知 total_steps，回退到 none")
        else:
            ctx.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                ctx.optimizer, T_max=ctx.total_steps, eta_min=eta_min
            )
            logger.info(f"学习率调度: cosine (T_max={ctx.total_steps}, eta_min={eta_min})")
    elif lr_sched == "cosine_with_restart":
        t0 = int(getattr(args, "lr_scheduler_t0", 500) or 500)
        t_mult = int(getattr(args, "lr_scheduler_t_mult", 2) or 2)
        eta_min = float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0)
        ctx.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            ctx.optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
        )
        logger.info(f"学习率调度: cosine_with_restart (T_0={t0}, T_mult={t_mult}, eta_min={eta_min})")
