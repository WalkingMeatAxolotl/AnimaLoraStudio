"""主训练循环：for epoch / for batch / 累积 / forward / loss / 周期 IO。

抽自 main() L596-884（ADR 0003 PR-B）。

放在 training/ 顶层（不在 phases/ 下）—— 它不是一次性 setup，是迭代主体；
但 run(ctx) 签名跟 phase 一致，方便 main() 编排。
"""

from __future__ import annotations

import logging
import time

import torch
import torch.nn.functional as F

from training.context import TrainingContext
from training.loss_weighting import compute_loss_weight
from training.model_loading import forward_with_optional_checkpoint
from training.noise import make_noise
from training.observability import render_curve_panel
from training.sample_runner import run_sample
from training.state import save_training_state
from training.text_encoding import (
    _build_qwen_text_from_prompt,
    encode_qwen,
    tokenize_t5_weighted,
)
from training.timestep_sampling import sample_t
from utils.optimizer_utils import optimizer_eval_mode


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """跑训练直到 args.epochs 或 args.max_steps 上限。"""
    args = ctx.args

    step_start_time = time.perf_counter()

    for epoch in range(ctx.start_epoch, args.epochs):
        ctx.current_epoch = epoch
        if ctx.use_cached and hasattr(ctx.dataloader, "batch_sampler") and hasattr(ctx.dataloader.batch_sampler, "set_epoch"):
            ctx.dataloader.batch_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(ctx.dataloader):
            # 在累积周期开始时记录时间
            if batch_idx % args.grad_accum == 0:
                step_start_time = time.perf_counter()

            captions = batch["captions"]

            # 获取 latents（缓存模式或实时编码）
            if ctx.use_cached:
                latents = batch["latents"].to(ctx.device, dtype=ctx.dtype)
            else:
                pixels = batch["pixel_values"].to(ctx.device, dtype=ctx.dtype)
                with torch.no_grad():
                    pixels_5d = pixels.unsqueeze(2)  # [B,C,1,H,W]
                    latents = ctx.vae.model.encode(pixels_5d, ctx.vae.scale)

            bs = latents.shape[0]

            # 文本编码
            with torch.no_grad():
                # 参考指南/ComfyUI：Qwen 通道不传权重；T5 通道提供 token 权重
                qwen_texts = [_build_qwen_text_from_prompt(c) for c in captions]
                qwen_emb, qwen_attn = encode_qwen(ctx.qwen_model, ctx.qwen_tok, qwen_texts, ctx.device)
                t5_ids, t5_attn, t5_w = tokenize_t5_weighted(ctx.t5_tok, captions, max_length=512)
                t5_ids = t5_ids.to(ctx.device)
                t5_attn = t5_attn.to(ctx.device)
                t5_w = t5_w.to(ctx.device, dtype=torch.float32)
                cross = ctx.model.preprocess_text_embeds(qwen_emb, t5_ids)
                if cross.shape[1] < 512:
                    cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))
                # KV trim：把 padding 截到最近有效 token bucket（64/128/256/512）
                # t5_attn=1 表示有效 token；取批次内最大实际长度再 round up
                if getattr(args, "kv_trim", False):
                    _actual = int(t5_attn.sum(dim=-1).max().item())
                    _bucket = 512  # _actual > 512 时兜底（不裁，保持原行为）
                    for _b in (64, 128, 256, 512):
                        if _b >= _actual:
                            _bucket = _b
                            break
                    cross = cross[:, :_bucket, :].contiguous()

            # Flow Matching
            t = sample_t(
                bs, ctx.device,
                mode=str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal"),
                shift=float(getattr(args, "timestep_shift", 3.0) or 3.0),
            )

            # PR-C：adapter hook — 允许变体按 sigma_t / step 调整运行时结构
            # （T-LoRA / AdaLoRA / B-LoRA 等）。LyCORIS 走默认 no-op。
            from training.adapters.protocol import StepContext
            step_ctx = StepContext(
                global_step=ctx.global_step,
                total_steps=ctx.total_steps,
                epoch=epoch,
                sigma_t=t,
                args=args,
            )
            ctx.injector.on_step_begin(step_ctx)

            t_exp = t.view(-1, 1, 1, 1, 1)
            noise = make_noise(
                latents,
                noise_offset=float(getattr(args, "noise_offset", 0.0) or 0.0),
                pyramid_iters=int(getattr(args, "pyramid_noise_iters", 0) or 0),
                pyramid_discount=float(getattr(args, "pyramid_noise_discount", 0.35) or 0.35),
            )
            noisy = (1 - t_exp) * latents + t_exp * noise
            target = noise - latents

            # 前向
            pad_mask = torch.zeros(bs, 1, latents.shape[-2], latents.shape[-1], device=ctx.device, dtype=ctx.dtype)
            with torch.autocast("cuda", dtype=ctx.dtype):
                pred = forward_with_optional_checkpoint(
                    ctx.model, noisy, t.view(-1, 1), cross, pad_mask,
                    use_checkpoint=args.grad_checkpoint,
                )
                loss_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none")
                # 按样本加权（正则集可降低权重）
                if "loss_weight" in batch:
                    w = batch["loss_weight"].to(ctx.device).view(-1, *([1] * (loss_per_sample.dim() - 1)))
                    loss_per_sample = loss_per_sample * w
                # timestep-dependent loss 权重
                lw_scheme = str(getattr(args, "loss_weighting", "none") or "none")
                if lw_scheme != "none":
                    lw = compute_loss_weight(
                        t,
                        scheme=lw_scheme,
                        min_snr_gamma=float(getattr(args, "min_snr_gamma", 5.0) or 5.0),
                        weight_cap_ratio=float(getattr(args, "weight_cap_ratio", 0.0) or 0.0),
                    ).to(device=ctx.device, dtype=torch.float32)
                    loss_per_sample = loss_per_sample * lw.view(-1, *([1] * (loss_per_sample.dim() - 1)))
                loss = loss_per_sample.mean()

                # PR-C：adapter hook — 变体可加正则项（OFT orth penalty /
                # Ortho-Hydra balance loss 等）。LyCORIS 返回 None，noop。
                reg = ctx.injector.regularization_loss(step_ctx)
                if reg is not None:
                    loss = loss + reg

            # NaN 检测：forward 出 NaN 时跳过本 micro-batch
            if not torch.isfinite(loss):
                logger.warning(f"step {ctx.global_step} micro-batch {batch_idx}: loss={loss.item():.4g}，跳过")
                ctx.optimizer.zero_grad()
                continue

            # 反向传播
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                # NaN 梯度检测：跳过本次 update，清零继续
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

                # 记录 loss 历史
                loss_val = float(loss.item() * args.grad_accum)
                if args.loss_curve_steps and len(ctx.loss_history) < args.loss_curve_steps:
                    ctx.loss_history.append(loss_val)

                # 更新进度显示
                now = time.perf_counter()
                lr = ctx.optimizer.param_groups[0]["lr"] if ctx.optimizer.param_groups else 0.0

                # 更新训练监控面板
                if ctx.monitor_server:
                    try:
                        from train_monitor import update_monitor
                        update_monitor(
                            loss=loss_val, lr=lr, epoch=epoch + 1,
                            total_epochs=int(args.epochs or 0),
                            step=ctx.global_step,
                            total_steps=ctx.total_steps, speed=ctx.speed_ema or 0,
                        )
                    except Exception:
                        pass
                dt_step = now - step_start_time
                steps_per_sec = (1.0 / dt_step) if dt_step > 0 else 0.0
                ctx.speed_ema = steps_per_sec if ctx.speed_ema is None else (0.9 * ctx.speed_ema + 0.1 * steps_per_sec)
                ctx.wandb_monitor.log(
                    {
                        "train/loss": loss_val,
                        "train/lr": float(lr),
                        "train/epoch": epoch + 1,
                        "train/speed_it_s": float(ctx.speed_ema or 0),
                    },
                    step=ctx.global_step,
                )

                if ctx.use_rich:
                    desc = f"epoch {epoch+1}/{args.epochs} step {ctx.global_step}/{ctx.total_steps or '?'}"
                    ctx.progress.update(
                        ctx.task_id, advance=1, description=desc,
                        loss=loss_val, lr=float(lr), speed=float(ctx.speed_ema or 0),
                    )
                    if ctx.live and args.loss_curve_steps > 0 and not args.no_live_curve:
                        panel = render_curve_panel(ctx.loss_history, width=min(60, args.loss_curve_steps), height=10)
                        if panel is not None:
                            from rich.console import Group
                            ctx.live.update(Group(ctx.progress, panel))
                elif ctx.use_plain:
                    print(f"epoch {epoch+1}/{args.epochs} step {ctx.global_step} loss={loss_val:.6f} lr={lr:.2e} speed={ctx.speed_ema:.2f} it/s", end="\r", flush=True)
                elif args.log_every and ctx.global_step % args.log_every == 0:
                    print(f"epoch={epoch} step={ctx.global_step} loss={loss_val:.6f} lr={lr:.2e} speed={steps_per_sec:.2f} it/s")

                # 按 step 采样（轮换提示词）
                if args.sample_steps > 0 and ctx.global_step % args.sample_steps == 0:
                    prompt = ctx.get_next_sample_prompt()
                    prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    ctx.emit(f"采样中 (step {ctx.global_step}): {prompt_short}")
                    run_sample(
                        ctx,
                        prompt=prompt,
                        sample_path=ctx.sample_dir / f"step_{ctx.global_step}.png",
                        wandb_key="samples/step",
                        wandb_caption=f"step {ctx.global_step}: {prompt}",
                        wandb_step=ctx.global_step,
                    )

                # 定期保存 LoRA 权重（按 step）
                save_every_steps = getattr(args, "save_every_steps", 0)
                if save_every_steps > 0 and ctx.global_step % save_every_steps == 0:
                    lora_path = ctx.output_dir / f"{args.output_name}_step{ctx.global_step}.safetensors"
                    # PPSF：保存 averaged weights 的 LoRA
                    with optimizer_eval_mode(ctx.optimizer):
                        ctx.injector.save(lora_path)
                    ctx.emit(f"Saved LoRA: {lora_path}")

                # 定期保存训练状态（断点续训）
                save_state_every = getattr(args, "save_state_every", 0)
                if save_state_every > 0 and ctx.global_step % save_state_every == 0:
                    state_path = ctx.output_dir / f"training_state_step{ctx.global_step}.pt"
                    # 获取监控面板数据用于恢复 loss 曲线
                    monitor_data = None
                    if ctx.monitor_server:
                        try:
                            from train_monitor import get_state
                            monitor_data = get_state()
                        except Exception:
                            pass
                    # PPSF：state + LoRA 都走 averaged weights
                    with optimizer_eval_mode(ctx.optimizer):
                        save_training_state(
                            state_path, ctx.injector, ctx.optimizer, epoch, ctx.global_step,
                            ctx.loss_history, monitor_state=monitor_data, scheduler=ctx.scheduler,
                        )
                        # 同时保存 LoRA 权重
                        lora_path = ctx.output_dir / f"{args.output_name}_step{ctx.global_step}.safetensors"
                        ctx.injector.save(lora_path)

                # 检查 max_steps
                if args.max_steps and ctx.global_step >= args.max_steps:
                    break

        # epoch 结束后的操作
        ctx.current_epoch = epoch + 1
        if not args.max_steps or ctx.global_step < args.max_steps:
            # 保存 checkpoint
            if args.save_every > 0 and ctx.current_epoch % args.save_every == 0:
                save_path = ctx.output_dir / f"{args.output_name}_epoch{ctx.current_epoch}.safetensors"
                # PPSF：保存 averaged weights 的 LoRA
                with optimizer_eval_mode(ctx.optimizer):
                    ctx.injector.save(save_path)
                ctx.emit(f"Saved LoRA: {save_path}")

            # 采样（轮换提示词）
            if args.sample_every > 0 and ctx.current_epoch % args.sample_every == 0:
                prompt = ctx.get_next_sample_prompt()
                prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                ctx.emit(f"采样中 (epoch {ctx.current_epoch}): {prompt_short}")
                run_sample(
                    ctx,
                    prompt=prompt,
                    sample_path=ctx.sample_dir / f"epoch_{ctx.current_epoch}.png",
                    wandb_key="samples/epoch",
                    wandb_caption=f"epoch {ctx.current_epoch}: {prompt}",
                    wandb_step=ctx.global_step,
                )

        # 检查 max_steps
        if args.max_steps and ctx.global_step >= args.max_steps:
            break
