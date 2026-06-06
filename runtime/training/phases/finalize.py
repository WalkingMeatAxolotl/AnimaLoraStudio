"""finalize_phase：训练循环结束后的最终保存 + 清理 + 最终曲线 + wandb finish。

抽自 main() L886-905（ADR 0003 PR-B）。
"""

from __future__ import annotations

import logging

from training.context import TrainingContext
from training.observability import render_loss_curve
from training.snapshot import emit_event
from utils.optimizer_utils import optimizer_eval_mode


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """
    - 最终 LoRA safetensors 落盘（PPSF 走 averaged weights）
    - 清理 Rich Live / Progress
    - 打印最终 loss 曲线
    - wandb finish
    """
    args = ctx.args

    # 最终保存
    final_path = ctx.output_dir / f"{args.output_name}.safetensors"
    # PPSF：最终输出走 averaged weights
    with optimizer_eval_mode(ctx.optimizer):
        ctx.injector.save(final_path)
    emit_event("eval_checkpoint_saved", {
        "checkpoint_path": str(final_path),
        "epoch": int(ctx.current_epoch or 0),
        "step": int(ctx.global_step or 0),
        "trigger": "final",
    })

    # 清理 SRA v2 hook（MLP 不保存到 LoRA safetensors，训练完即丢弃）
    if ctx.sra_aligner is not None:
        ctx.sra_aligner.remove_hooks()
        ctx.sra_aligner = None

    # 清理进度显示
    if ctx.live:
        ctx.live.stop()
    elif ctx.use_rich:
        ctx.progress.stop()

    # 显示最终 loss 曲线
    if args.loss_curve_steps and ctx.loss_history:
        chart = render_loss_curve(ctx.loss_history, width=min(80, len(ctx.loss_history)), height=10)
        ctx.emit(f"Loss curve (first {len(ctx.loss_history)} steps):\n{chart}")

    ctx.emit(f"Saved final LoRA: {final_path}")
    ctx.wandb_monitor.upload_model(final_path)
    ctx.wandb_monitor.finish()
    logger.info("训练完成!")
