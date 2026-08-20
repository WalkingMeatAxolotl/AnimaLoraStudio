"""optimizer_phase：optimizer dispatch + grad_clip + total_steps + lr_scheduler。

抽自 main() L344-437（ADR 0003 PR-B）。

注：optimizer dispatch 这次保留 if-elif 老风格（adamw / prodigy /
prodigy_plus_schedulefree），PR-C 会把它换成 plugin registry。
"""

from __future__ import annotations

import logging

from studio.infrastructure.log_messages import msg
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

    # SRA v2：projection MLP 的参数加入 optimizer（weight_decay=0，可选独立 lr）
    if ctx.sra_aligner is not None:
        sra_groups = ctx.sra_aligner.get_param_groups(lr=None)
        param_groups = param_groups + sra_groups
        logger.debug(
            "sra: projection layer params added to optimizer params=%.1fM",
            sum(p.numel() for g in sra_groups for p in g["params"]) / 1e6,
        )

    from training.optimizers import build_optimizer, validate_optimizer
    validate_optimizer(args)  # PPSF 检查 lr_scheduler=none 等启动期约束
    ctx.optimizer = build_optimizer(args, param_groups, args.learning_rate, ctx.weight_decay)
    if ctx.weight_decay > 0:
        # LoKr 的 w1 不参与 weight_decay —— 条件后缀做成整句变体而不是拼片段
        # （i18n 惯例：译者要能调整整句语序）。
        logger.info(msg(
            "train.weight_decay_lokr" if ctx.injector.use_lokr else "train.weight_decay",
            optimizer=ctx.optimizer_type, wd=ctx.weight_decay,
        ))
    ctx.grad_clip = float(getattr(args, "grad_clip_max_norm", 0) or 0)
    if ctx.grad_clip > 0:
        logger.info(msg("train.grad_clip", value=ctx.grad_clip))
    ctx.trainable_params = [p for group in ctx.optimizer.param_groups for p in group["params"]]

    # 计算总步数
    try:
        # ceil：最后一组不满 grad_accum 也算一个 update step（loop 末批会 step），
        # 与 _accumulation_step 的「尾组不满也 step」一致，scheduler 步数才对得上。
        # 注：NaViT 打包器的 __len__ 是 epoch-0 采样的包数（next-fit/窗口 FFD 顺序依赖，
        # 每 epoch 略有波动）。scheduler horizon 在此处一次性构造、无法中途重算，故 navit 下
        # steps_per_epoch/total_steps 为估计值，LR 曲线触底点可能有几步偏差（可接受）；
        # 梯度累积边界的精确性由 loop.py 每 epoch 刷新 dl_len 保证。
        _dl_len = len(ctx.dataloader)
        ctx.steps_per_epoch = (_dl_len + args.grad_accum - 1) // args.grad_accum
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

    logger.info(msg(
        "train.step_plan",
        samples=len(ctx.dataset), steps_per_epoch=ctx.steps_per_epoch,
        total_steps=ctx.total_steps,
    ))
    logger.debug(
        "steps: by_epochs=%s by_max_steps=%s chosen=%s",
        by_epochs, by_max_steps, ctx.total_steps,
    )

    # 学习率调度器：PR-C 通过 schedulers/ plugin registry 派发；"none" 自动返回 None
    from training.schedulers import build_scheduler
    ctx.scheduler = build_scheduler(args, ctx.optimizer, ctx.total_steps)

    # Timestep 采样器（baseline 或 InfoNoise；total_steps 确定后才能算 N_warm）
    from training.timestep_samplers import build_timestep_sampler
    ctx.timestep_sampler = build_timestep_sampler(args, ctx.total_steps)
