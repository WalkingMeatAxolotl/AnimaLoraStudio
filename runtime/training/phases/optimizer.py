"""optimizer_phase：optimizer dispatch + grad_clip + total_steps + lr_scheduler。

抽自 main() L344-437（ADR 0003 PR-B）。

注：optimizer dispatch 这次保留 if-elif 老风格（adamw / prodigy /
prodigy_plus_schedulefree），PR-C 会把它换成 plugin registry。
"""

from __future__ import annotations

import logging

from training.context import TrainingContext


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """
    - injector.get_param_groups + build_optimizer(args, ...) via training.optimizers
    - validate_optimizer 启动期约束检查（如 PPSF lr_scheduler=none）
    - grad_clip / trainable_params
    - 计算 total_steps（min(by_epochs, by_max_steps)）
    - build_scheduler(args, optimizer, total_steps) via training.schedulers
    """
    args = ctx.args

    # 优化器：PR-C 通过 optimizers/ plugin registry 派发
    ctx.weight_decay = float(getattr(args, "weight_decay", 0.01) or 0.0)
    param_groups = ctx.injector.get_param_groups(ctx.weight_decay)
    ctx.optimizer_type = (getattr(args, "optimizer_type", "adamw") or "adamw").lower()

    from training.optimizers import build_optimizer, validate_optimizer
    validate_optimizer(args)  # PPSF 检查 lr_scheduler=none 等启动期约束
    ctx.optimizer = build_optimizer(args, param_groups, args.learning_rate, ctx.weight_decay)
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

    # 学习率调度器：PR-C 通过 schedulers/ plugin registry 派发；"none" 自动返回 None
    from training.schedulers import build_scheduler
    ctx.scheduler = build_scheduler(args, ctx.optimizer, ctx.total_steps)

    # InfoNoise 自适应调度器（total_steps 确定后才能算 N_warm）
    from training.infonoise import build_info_noise
    ctx.info_noise = build_info_noise(args, ctx.total_steps)
