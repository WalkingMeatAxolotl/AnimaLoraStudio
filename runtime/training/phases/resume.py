"""resume_phase：progress 初始化 + state recovery + 信号注册 + step 0 baseline 采样。

抽自 main() L439-594（ADR 0003 PR-B）。
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path

from training.bootstrap import init_progress
from training.context import TrainingContext
from training.observability import render_curve_panel
from training.sample_runner import run_sample
from training.state import load_training_state


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """
    - init_progress + 可选 Rich Live（含 loss curve panel）
    - 如有 --resume-state：load_training_state + restore monitor 历史 loss
    - 注册 SIGINT → ctx.handle_interrupt
    - 准备 sample_prompts 列表（多角色轮换）
    - global_step==0 时跑 baseline 采样（最多 3 prompt）
    """
    args = ctx.args

    # 初始化进度显示
    ctx.progress, ctx.task_id, progress_kind = init_progress(not args.no_progress, ctx.total_steps)
    ctx.use_rich = progress_kind == "rich"
    ctx.use_plain = ctx.progress == "plain"
    ctx.live = None
    ctx.loss_history = []
    ctx.speed_ema = None

    if ctx.use_rich:
        try:
            from rich.console import Group
            from rich.live import Live
            curve_panel = None
            if args.loss_curve_steps > 0 and not args.no_live_curve:
                curve_panel = render_curve_panel([], width=min(60, args.loss_curve_steps), height=10)
            group = Group(ctx.progress, curve_panel) if curve_panel is not None else Group(ctx.progress)
            ctx.live = Live(group, refresh_per_second=10)
            ctx.live.start()
        except Exception:
            ctx.live = None
            ctx.progress.start()

    # 训练循环初始状态
    ctx.global_step = 0
    ctx.start_epoch = 0

    # 从训练状态恢复（断点续训）
    if getattr(args, "resume_state", "") and Path(args.resume_state).exists():
        ctx.start_epoch, ctx.global_step, ctx.loss_history, saved_monitor_state = load_training_state(
            args.resume_state, ctx.injector, ctx.optimizer, ctx.scheduler,
        )
        ctx.emit(f"从断点恢复训练: epoch={ctx.start_epoch}, step={ctx.global_step}")

        # 恢复监控面板的历史数据（loss 曲线等）
        if ctx.monitor_server and saved_monitor_state:
            try:
                from train_monitor import restore_monitor_state
                restore_monitor_state(
                    losses=saved_monitor_state.get("losses"),
                    lr_history=saved_monitor_state.get("lr_history"),
                    epoch=ctx.start_epoch,
                    step=ctx.global_step,
                    total_steps=ctx.total_steps,
                )
                ctx.emit(f"监控面板历史数据已恢复: {len(saved_monitor_state.get('losses', []))} 个 loss 点")
            except Exception as e:
                ctx.emit(f"监控数据恢复失败: {e}")

    # Ctrl+C 信号处理：handle_interrupt 由 TrainingContext 自带
    signal.signal(signal.SIGINT, ctx.handle_interrupt)

    ctx.current_epoch = ctx.start_epoch
    ctx.model.train()
    if ctx.optimizer_type == "prodigy_plus_schedulefree" and hasattr(ctx.optimizer, "train"):
        ctx.optimizer.train()
    # step_start_time 由 train_loop 内自己重置；这里不需要

    # 设置采样提示词列表（支持多角色轮换）
    ctx.sample_prompts = getattr(args, "sample_prompts", []) or []
    if not ctx.sample_prompts and args.sample_prompt:
        ctx.sample_prompts = [args.sample_prompt]
    ctx.sample_prompt_idx = 0

    # Step 0 初始采样（基线效果，测试所有提示词）
    # 只在新训练时执行（global_step == 0），resume 时跳过
    sampling_enabled = args.sample_steps > 0 or args.sample_every > 0
    if ctx.global_step == 0 and sampling_enabled:
        ctx.emit("采样中 (step 0, 基线)...")
        for i, prompt in enumerate(ctx.sample_prompts[:3]):  # 最多测试 3 个
            sample_path = ctx.sample_dir / f"step_0_baseline_{i}.png"
            run_sample(
                ctx,
                prompt=prompt,
                sample_path=sample_path,
                wandb_key="samples/baseline",
                wandb_caption=f"step 0 baseline {i}: {prompt}",
                wandb_step=0,
                seed_offset=i,
            )
    elif ctx.global_step > 0 and sampling_enabled:
        ctx.emit(f"跳过启动基线采样（从 step {ctx.global_step} 恢复，非 step 0）")
