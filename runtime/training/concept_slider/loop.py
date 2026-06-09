"""Concept slider 训练循环（POC）。

四前向公式（image-pair, Gandikota+ 2023）：
  pred_pos_base = model(noisy_pos, t, c) | LoRA off, no_grad
  pred_neg_base = model(noisy_neg, t, c) | LoRA off, no_grad
  pred_lora     = model(noisy_pos, t, c) | LoRA on,  grad
  target        = pred_pos_base + eta * (pred_pos_base - pred_neg_base)
  loss          = MSE(pred_lora, target)

方向：weight=+1 推理时 → 更高饱和（pos 方向）；weight=-1 → 更低饱和。

跟 training/loop.py 的差异：
- 数据是 pair（pixel_pos / pixel_neg）；同 noise + 同 t 算 noisy_pos / noisy_neg
- 不读 batch["loss_weight"] / "is_reg"（POC 不支持 reg 集）
- 不走 loss_weighting / huber / InfoNoise record（POC 强制 MSE + uniform t）
- 训练中 multi-weight 采样（fixed seed × {-1, 0, +1}）直接看 comp leakage
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn.functional as F

from training.context import TrainingContext
from training.model_loading import forward_with_optional_checkpoint
from training.noise import make_noise
from training.sample_runner import run_sample
from training.text_encoding import (
    _build_qwen_text_from_prompt,
    encode_qwen,
    tokenize_t5_weighted,
)
from utils.optimizer_utils import get_optimizer_monitor_metrics, optimizer_eval_mode


logger = logging.getLogger(__name__)


def _encode_text(ctx: TrainingContext, captions: list[str]) -> torch.Tensor:
    """qwen + T5 编码 → cross；与 training/loop.py 同一公式（POC 不开 kv_trim）。"""
    with torch.no_grad():
        qwen_texts = [_build_qwen_text_from_prompt(c) for c in captions]
        qwen_emb, _ = encode_qwen(ctx.qwen_model, ctx.qwen_tok, qwen_texts, ctx.device)
        t5_ids, t5_attn, t5_w = tokenize_t5_weighted(ctx.t5_tok, captions, max_length=512)
        t5_ids = t5_ids.to(ctx.device)
        t5_attn = t5_attn.to(ctx.device)
        t5_w = t5_w.to(ctx.device, dtype=torch.float32)
        cross = ctx.model.preprocess_text_embeds(qwen_emb, t5_ids, t5xxl_weights=t5_w)
        if cross.shape[1] < 512:
            cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))
    return cross


def _emit_multi_weight_samples(ctx: TrainingContext, step: int) -> None:
    """fixed seed × {-1.0, 0.0, +1.0} 出 3 张图：训练中肉眼看 comp leakage / 饱和度变化。"""
    weights = (-1.0, 0.0, 1.0)
    prompt = ctx.get_next_sample_prompt()
    prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
    ctx.emit(f"采样中 (step {step}, slider w∈{weights}): {prompt_short}")
    saved_scale: list[float] = []
    try:
        if ctx.injector.network is not None:
            saved_scale = [m.multiplier for m in ctx.injector.network.loras]
        for w in weights:
            ctx.injector.set_multiplier(w)
            sample_path = ctx.sample_dir / f"step_{step}_w{w:+.1f}.png"
            run_sample(
                ctx,
                prompt=prompt,
                sample_path=sample_path,
                wandb_key=None,  # POC 不传 wandb
                wandb_caption=None,
                wandb_step=step,
            )
    finally:
        # 还原训练 multiplier
        if ctx.injector.network is not None and saved_scale:
            for m, s in zip(ctx.injector.network.loras, saved_scale):
                m.multiplier = s
        elif ctx.injector.network is not None:
            ctx.injector.set_multiplier(1.0)


def run(ctx: TrainingContext) -> None:
    """跑 concept slider 训练循环直到 args.epochs 或 args.max_steps 上限。"""
    args = ctx.args
    eta = float(getattr(args, "slider_eta", 1.0) or 1.0)
    logger.info(f"concept slider 训练启动: eta={eta}, save_every_steps={args.save_every_steps}, sample_steps={args.sample_steps}")

    step_start_time = time.perf_counter()

    for epoch in range(ctx.start_epoch, args.epochs):
        ctx.current_epoch = epoch
        epoch_loss_sum = 0.0
        epoch_step_count = 0

        for batch_idx, batch in enumerate(ctx.dataloader):
            if batch_idx % args.grad_accum == 0:
                step_start_time = time.perf_counter()

            captions = batch["captions"]
            pixel_pos = batch["pixel_pos"].to(ctx.device, dtype=ctx.dtype)
            pixel_neg = batch["pixel_neg"].to(ctx.device, dtype=ctx.dtype)

            # VAE encode pos / neg；POC 不走 cached 路径
            with torch.no_grad():
                latents_pos = ctx.vae.model.encode(pixel_pos.unsqueeze(2), ctx.vae.scale)
                latents_neg = ctx.vae.model.encode(pixel_neg.unsqueeze(2), ctx.vae.scale)
            bs = latents_pos.shape[0]

            # 文本（pos / neg / lora 三个前向共用）
            cross = _encode_text(ctx, captions)

            # 同 noise + 同 t：这是 slider 隔离的关键
            t = ctx.timestep_sampler.sample(bs, ctx.device)
            t_exp = t.view(-1, 1, 1, 1, 1)
            noise = make_noise(
                latents_pos,
                noise_offset=float(getattr(args, "noise_offset", 0.0) or 0.0),
                pyramid_iters=0,
            )
            noisy_pos = (1 - t_exp) * latents_pos + t_exp * noise
            noisy_neg = (1 - t_exp) * latents_neg + t_exp * noise
            pad_mask = torch.zeros(
                bs, 1, latents_pos.shape[-2], latents_pos.shape[-1],
                device=ctx.device, dtype=ctx.dtype,
            )

            # Base 前向（LoRA off, no_grad）
            with torch.no_grad(), ctx.injector.disabled():
                with torch.autocast("cuda", dtype=ctx.dtype):
                    pred_pos_base = forward_with_optional_checkpoint(
                        ctx.model, noisy_pos, t.view(-1, 1), cross, pad_mask,
                        use_checkpoint=False,  # base 不需要 grad，关 checkpoint 省 wall-clock
                    )
                    pred_neg_base = forward_with_optional_checkpoint(
                        ctx.model, noisy_neg, t.view(-1, 1), cross, pad_mask,
                        use_checkpoint=False,
                    )

            # LoRA 前向（有 grad）；target 用 base 算出来，对 pred_lora 拉到 pos+eta·(pos-neg)
            with torch.autocast("cuda", dtype=ctx.dtype):
                pred_lora = forward_with_optional_checkpoint(
                    ctx.model, noisy_pos, t.view(-1, 1), cross, pad_mask,
                    use_checkpoint=args.grad_checkpoint,
                )
                target = pred_pos_base.float() + eta * (pred_pos_base.float() - pred_neg_base.float())
                loss = F.mse_loss(pred_lora.float(), target)

            if not torch.isfinite(loss):
                logger.warning(f"step {ctx.global_step} micro-batch {batch_idx}: loss={loss.item():.4g}，跳过")
                ctx.optimizer.zero_grad()
                continue

            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                has_nan_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in ctx.trainable_params
                )
                if has_nan_grad:
                    logger.warning(f"step {ctx.global_step}: 梯度含 NaN/Inf，跳过 optimizer.step()")
                    ctx.optimizer.zero_grad()
                    continue

                if ctx.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ctx.trainable_params, max_norm=ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None and ctx.optimizer_type != "prodigy_plus_schedulefree":
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()
                ctx.global_step += 1

                loss_val = float(loss.item() * args.grad_accum)
                epoch_loss_sum += loss_val
                epoch_step_count += 1
                if args.loss_curve_steps and len(ctx.loss_history) < args.loss_curve_steps:
                    ctx.loss_history.append(loss_val)

                # 进度日志（最简版，砍掉 rich live curve / monitor server / wandb）
                now = time.perf_counter()
                optimizer_metrics = get_optimizer_monitor_metrics(ctx.optimizer)
                lr = optimizer_metrics["lr"]
                dt_step = now - step_start_time
                steps_per_sec = (1.0 / dt_step) if dt_step > 0 else 0.0
                ctx.speed_ema = steps_per_sec if ctx.speed_ema is None else (0.9 * ctx.speed_ema + 0.1 * steps_per_sec)

                log_payload: dict[str, Any] = {
                    "train/loss": loss_val,
                    "train/lr": float(lr),
                    "train/speed_it_s": float(ctx.speed_ema or 0),
                }
                ctx.wandb_monitor.log(log_payload, step=ctx.global_step)

                if args.log_every and ctx.global_step % args.log_every == 0:
                    print(
                        f"[slider] epoch={epoch} step={ctx.global_step} loss={loss_val:.6f} "
                        f"lr={lr:.2e} speed={steps_per_sec:.2f} it/s"
                    )

                # multi-weight 采样：训练中看 comp leakage
                if args.sample_steps > 0 and ctx.global_step % args.sample_steps == 0:
                    _emit_multi_weight_samples(ctx, ctx.global_step)

                # 保存 LoRA checkpoint
                save_every_steps = getattr(args, "save_every_steps", 0)
                if save_every_steps > 0 and ctx.global_step % save_every_steps == 0:
                    lora_path = ctx.output_dir / f"{args.output_name}_step{ctx.global_step}.safetensors"
                    with optimizer_eval_mode(ctx.optimizer):
                        ctx.injector.save(lora_path)
                    ctx.emit(f"Saved LoRA: {lora_path}")

                if args.max_steps and ctx.global_step >= args.max_steps:
                    break

        # epoch 结束：epoch-level checkpoint（如果用户开了 save_every_epochs）
        ctx.current_epoch = epoch + 1
        if epoch_step_count > 0:
            ctx.wandb_monitor.log(
                {"train/loss_epoch": epoch_loss_sum / epoch_step_count, "train/epoch": ctx.current_epoch},
                step=ctx.global_step,
            )
        if not args.max_steps or ctx.global_step < args.max_steps:
            if args.save_every_epochs > 0 and ctx.current_epoch % args.save_every_epochs == 0:
                save_path = ctx.output_dir / f"{args.output_name}_epoch{ctx.current_epoch}.safetensors"
                with optimizer_eval_mode(ctx.optimizer):
                    ctx.injector.save(save_path)
                ctx.emit(f"Saved LoRA: {save_path}")

        if args.max_steps and ctx.global_step >= args.max_steps:
            break
