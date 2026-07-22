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


def _release_block_swap(ctx: TrainingContext) -> None:
    """训练收尾：摘 block swap 钩子并把 pinned 内存还给系统。

    进程马上就要退出、由 OS 回收，为什么还要主动还？因为**退出前还有一段尾巴**：
    wandb 上传最终 LoRA（可能几十秒到几分钟），而此时 supervisor 已经收到
    `eval_training_finished` 排了训练后评估 job —— 评估进程要加载模型，却撞上
    本进程还占着 11GB+ 页锁定内存（连换页都不行）。在内存紧张的机器上（正是
    block swap 要服务的那批）这会直接把评估拖垮。

    放在这里而不是更早：训练循环期间 pinned 装的就是模型权重，中途还等于自毁。
    """
    swap = getattr(ctx, "block_swap", None)
    if swap is None:
        return
    from training.block_swap import release_pinned_host_cache

    freed = getattr(swap, "pinned_bytes", 0)
    try:
        # close() 而不是 detach()：pinned 主副本被 param.data 引用，而持有 block
        # 的不止 ctx.model（injector 持 org_module、optimizer 持参数），逐个去找
        # 持有者不可靠 —— 让组件自己把参数指走
        swap.close()
        ctx.block_swap = None
        import gc

        gc.collect()
        release_pinned_host_cache()
        logger.info("block swap 已释放：归还 pinned 内存 %.2fGB", freed / 1024**3)
    except Exception:  # noqa: BLE001
        logger.debug("block swap 释放失败（不影响训练结果）", exc_info=True)


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
    # 训练全部结束 → supervisor 排训练后评估（inline / checkpoint-trigger 已移除，
    # 评估统一走训练后的独立 job）。
    emit_event("eval_training_finished", {
        "epoch": int(ctx.current_epoch or 0),
        "step": int(ctx.global_step or 0),
    })

    # 清理 SRA v2 hook（MLP 不保存到 LoRA safetensors，训练完即丢弃）
    if ctx.sra_aligner is not None:
        ctx.sra_aligner.remove_hooks()
        ctx.sra_aligner = None

    _release_block_swap(ctx)

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
