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
    # resolution 可能是列表（多分辨率）；采样预览用首档（base）分辨率。
    _res = args.resolution
    base_reso = int(_res[0]) if isinstance(_res, (list, tuple)) and _res else int(_res)
    s_w = int(getattr(args, "sample_width", 0) or 0) or base_reso
    s_h = int(getattr(args, "sample_height", 0) or 0) or base_reso
    # 必须 align_px（VAE stride 8 × patch_spatial 2 = 16）的倍数，
    # 否则 cosmos_predict2 spatial_patch 断言失败
    spec = ctx.family.spec
    _align = spec.latent.align_px
    s_w = max(_align, (s_w // _align) * _align)
    s_h = max(_align, (s_h // _align) * _align)
    s_cfg_value = getattr(args, "sample_cfg_scale", spec.sampling.default_cfg)
    s_cfg = float(spec.sampling.default_cfg if s_cfg_value is None else s_cfg_value)
    s_neg = str(getattr(args, "sample_negative_prompt", "") or "")
    s_seed = int(getattr(args, "sample_seed", 0) or 0)
    s_steps = int(
        getattr(args, "sample_infer_steps", spec.sampling.default_steps)
        or spec.sampling.default_steps
    )
    s_sampler = str(
        getattr(args, "sample_sampler_name", spec.sampling.default_sampler)
        or spec.sampling.default_sampler
    )
    s_sched = str(
        getattr(args, "sample_scheduler", spec.sampling.default_scheduler)
        or spec.sampling.default_scheduler
    )

    # T-LoRA：与 ControlGenAI/T-LoRA 官方推理一致 —— sample 阶段不应用 timestep
    # mask。官方 inferencer 不传 sigma_mask, forward 内 fallback 出全 1 mask =
    # 满 rank 推理。这里显式清零 PR 的训练态 mask；下一次 on_step_begin 会
    # 重新按 sigma_t 写入，无需事后恢复。
    # non-tlora adapter (lokr / loha / lora) 没这个方法, getattr 返回 None 安全跳过。
    clear_fn = getattr(ctx.injector, "clear_timestep_mask", None)
    if callable(clear_fn):
        clear_fn()

    was_training = bool(getattr(ctx.model, "training", True))
    # 采样前归还训练积累的 allocator 碎片（多 bucket 形状的 reserved 段）。
    # 采样分辨率 ≠ 训练 bucket 形状 → 采样激活是全新分配；不清理时
    # 「训练 reserved + 采样新段」的瞬时提交可能顶穿 dedicated 上限，
    # 触发 WDDM 把训练张量 demote 到共享内存（=系统 RAM）——之后每步
    # 训练都走 PCIe，表现为采样后持续整机卡顿（显存/内存数字反而稳定）。
    # 采样前这次顺带结算「上一段训练步」的峰值并重置，于是采样后那次报出的
    # 就是纯采样期峰值 —— 两个数字合起来才能回答「这张卡够不够」
    _log_vram_watermark("采样前", peak_label="训练步")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        with optimizer_eval_mode(ctx.optimizer):
            ctx.model.eval()
            if s_seed:
                torch.manual_seed(s_seed + seed_offset)
            img = ctx.family.sample_image(
                ctx.model, ctx.vae, ctx.text_stack,
                prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
                negative_prompt=s_neg,
                sampler_name=s_sampler,
                scheduler=s_sched,
                device=ctx.device, dtype=ctx.dtype,
                seed=(s_seed + seed_offset) if s_seed else None,
            )
            img.save(sample_path)
            ctx.emit(f"采样保存: {sample_path.name}")
            if wandb_key and ctx.wandb_monitor.log_samples:
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
    except Exception as exc:
        logger.warning("采样失败，已跳过本次预览，不中断训练: %s", exc, exc_info=True)
    finally:
        if was_training:
            ctx.model.train()
        else:
            ctx.model.eval()
        # 归还采样期的新形状分配，训练继续时 allocator 干净
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_vram_watermark("采样后", peak_label="采样期")


def _log_vram_watermark(stage: str, *, peak_label: str = "") -> None:
    """采样前后水位一行日志：alloc/reserved（torch 视角）+ 全卡（NVML）。

    reserved 与全卡的差额变化用于定位 WDDM demote（共享内存曲线跳升时
    torch 侧数字反而稳定）。

    ``peak_label``：同时报告**上一段的峰值**（`max_memory_allocated` 高水位）
    并重置计数。瞬时值读不出段内尖峰 —— 训练步稳态和采样期峰值可能差好几 GB，
    而决定「这张卡够不够」的是峰值不是稳态（block swap 场景尤其关键：常驻降下来
    之后，采样期就成了新的天花板）。失败静默——日志不阻塞训练。"""
    try:
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        line = f"[{stage}] 显存 alloc={alloc:.1f}GB reserved={reserved:.1f}GB"
        if peak_label:
            peak = torch.cuda.max_memory_allocated() / 1e9
            line += f" | {peak_label}峰值={peak:.1f}GB"
            torch.cuda.reset_peak_memory_stats()
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(0))
                line += f" 全卡={info.used / 1e9:.1f}GB"
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            pass
        logger.info(line)
    except Exception:
        pass
