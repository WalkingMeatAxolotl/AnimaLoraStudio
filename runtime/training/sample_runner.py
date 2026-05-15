"""周期采样 helper —— 消掉原 main() 里 3 处近乎逐行重复的 sample 块。

抽自 main() L550-594 / L757-795 / L840-872（ADR 0003 PR-B + memory P0）。

公开：
- run_sample — 单次采样：取 args.sample_* 参数 + 调 sample_image + 存 + wandb +
  monitor。所有调用方共用 PPSF averaged-weights 切换、model.eval/train 包夹、
  异常兜底等逻辑。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from training.context import TrainingContext
from training.sampling import sample_image
from utils.optimizer_utils import optimizer_eval_mode


logger = logging.getLogger(__name__)


def run_sample(
    ctx: TrainingContext,
    *,
    prompt: str,
    sample_path: Path,
    wandb_key: Optional[str] = None,
    wandb_caption: Optional[str] = None,
    wandb_step: Optional[int] = None,
    seed_offset: int = 0,
) -> None:
    """单次采样并保存到 sample_path。

    - PPSF：训练期间走 averaged weights 出图，事后切回训练权重
    - 异常兜底：sample 出错不应中断训练，只 log warn
    - wandb：wandb_key 传入则 log_image；caption / step 也传过去
    - monitor_state.json：永远尝试 push sample_path 给前端预览
    - seed_offset：baseline 模式下用 i 偏移让多 prompt 测出不同图

    sample_path 必须由 caller 决定（baseline 编号 / step / epoch 不在本函数判断）。
    """
    args = ctx.args
    s_w = int(getattr(args, "sample_width", 0) or 0) or int(args.resolution)
    s_h = int(getattr(args, "sample_height", 0) or 0) or int(args.resolution)
    s_cfg = float(getattr(args, "sample_cfg_scale", 4.0) or 4.0)
    s_neg = str(getattr(args, "sample_negative_prompt", "") or "")
    s_seed = int(getattr(args, "sample_seed", 0) or 0)
    s_steps = int(getattr(args, "sample_infer_steps", 25) or 25)
    s_sampler = str(getattr(args, "sample_sampler_name", "er_sde") or "er_sde")
    s_sched = str(getattr(args, "sample_scheduler", "simple") or "simple")

    with optimizer_eval_mode(ctx.optimizer):
        ctx.model.eval()
        if s_seed:
            torch.manual_seed(s_seed + seed_offset)
        img = sample_image(
            ctx.model, ctx.vae, ctx.qwen_model, ctx.qwen_tok, ctx.t5_tok,
            prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
            negative_prompt=(s_neg or None),
            sampler_name=s_sampler,
            scheduler=s_sched,
            device=ctx.device, dtype=ctx.dtype,
        )
        img.save(sample_path)
        ctx.emit(f"采样保存: {sample_path.name}")
        if wandb_key:
            ctx.wandb_monitor.log_image(
                wandb_key,
                sample_path,
                caption=wandb_caption or prompt,
                step=wandb_step,
            )
        if ctx.monitor_server:
            try:
                from train_monitor import update_monitor
                update_monitor(sample_path=sample_path)
            except Exception:
                pass
        ctx.model.train()
