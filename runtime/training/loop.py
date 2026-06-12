"""主训练循环：for epoch / for batch / 累积 / forward / loss / 周期 IO。

抽自 main() L596-884（ADR 0003 PR-B）。

放在 training/ 顶层（不在 phases/ 下）—— 它不是一次性 setup，是迭代主体；
但 run(ctx) 签名跟 phase 一致，方便 main() 编排。
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from training.context import TrainingContext
from training.leap import (
    bridge_training_step,
    lagrange_training_step,
    leap_training_step,
    sample_activation_timesteps,
    sample_two_timesteps,
    sparse_training_step,
)
from training.loss_weighting import compute_loss_weight
from training.model_loading import forward_with_optional_checkpoint
from training.noise import make_noise
from training.observability import render_curve_panel
from training.sample_runner import run_sample
from training.snapshot import (
    build_auto_epoch_config_path,
    build_auto_epoch_state_path,
    emit_event,
    write_config_snapshot,
)
from training.state import save_training_state
from training.text_encoding import (
    _build_qwen_text_from_prompt,
    encode_qwen,
    tokenize_t5_comfy_literal,
    tokenize_t5_weighted,
)
from utils.optimizer_utils import get_optimizer_monitor_metrics, optimizer_eval_mode


logger = logging.getLogger(__name__)


def _resolve_sra_weight(args: Any) -> float:
    """Read sra_weight without treating explicit 0 as a missing value."""
    raw = getattr(args, "sra_weight", 0.2)
    return 0.2 if raw is None else float(raw)


def _resolve_sra_effective_weight(args: Any, global_step: int, total_steps: int | None) -> float:
    """Apply the configured SRA schedule to the base SRA weight."""
    base = _resolve_sra_weight(args)
    if base == 0.0:
        return 0.0

    decay_type = str(getattr(args, "sra_decay_type", "none") or "none").lower()
    if decay_type == "none" or not total_steps or total_steps <= 0:
        return base

    start = float(getattr(args, "sra_decay_start_ratio", 0.2) or 0.0)
    end = float(getattr(args, "sra_decay_end_ratio", 0.3) or 0.0)
    progress = max(0.0, min(1.0, float(global_step) / float(total_steps)))

    if decay_type == "jump":
        return base if progress < start else 0.0

    if progress <= start:
        return base
    if progress >= end:
        return 0.0

    x = (progress - start) / max(end - start, 1e-8)
    if decay_type == "linear":
        scale = 1.0 - x
    elif decay_type == "cosine":
        scale = 0.5 * (1.0 + math.cos(math.pi * x))
    else:
        scale = 1.0
    return base * max(0.0, min(1.0, scale))


def _accumulation_step(batch_idx, dl_len, grad_accum):
    """返回 (group_size, is_group_end)：该 micro-batch 所在梯度累积组的实际大小，
    以及它是否是该组最后一个（触发 optimizer.step / zero_grad）。

    对齐 kohya-ss / HF Trainer：
    - epoch 最后一个 batch 即使凑不满 grad_accum 也 step —— 否则尾部 `len % ga` 个
      batch 的梯度会被丢（单 epoch）或泄漏进下一 epoch 第一个 step（多 epoch，因为
      zero_grad 只在 step 后调）。
    - 不满的尾组按**实际** micro-batch 数归一（group_size），而非恒 ÷grad_accum，
      免得末步梯度被削弱。

    dl_len=None（dataloader 无 __len__）时回退旧行为（恒 grad_accum，仅整除处 step）。
    """
    if dl_len is None or grad_accum <= 1:
        return grad_accum, (batch_idx + 1) % grad_accum == 0
    remainder = dl_len % grad_accum
    in_tail = remainder != 0 and batch_idx >= dl_len - remainder
    group_size = remainder if in_tail else grad_accum
    is_group_end = (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == dl_len
    return group_size, is_group_end


def run(ctx: TrainingContext) -> None:
    """跑训练直到 args.epochs 或 args.max_steps 上限。"""
    args = ctx.args

    step_start_time = time.perf_counter()

    # dataloader 每 epoch 的 batch 数（用于尾组「不满也 step」+ 按实际大小归一）。
    # 少数无 __len__ 的 dataloader 回退旧行为（见 _accumulation_step）。
    try:
        dl_len = len(ctx.dataloader)
    except TypeError:
        dl_len = None

    for epoch in range(ctx.start_epoch, args.epochs):
        ctx.current_epoch = epoch
        epoch_loss_sum = 0.0
        epoch_step_count = 0
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
                if bool(getattr(args, "caption_comfy_encoding", True)):
                    # Comfy-style（默认）：raw caption 进 Qwen（不清洗）；T5 整段
                    # 字面 tokenize，不解析权重语法——booru 括号 tag 保持字面，
                    # 等价于 CUI 用户推理时转义后 T5 看到的序列。与测试出图 /
                    # 训练预览的 conditioning 同一链路。
                    qwen_texts = [str(c) for c in captions]
                    qwen_emb, qwen_attn = encode_qwen(ctx.qwen_model, ctx.qwen_tok, qwen_texts, ctx.device)
                    t5_ids, t5_attn, t5_w = tokenize_t5_comfy_literal(ctx.t5_tok, captions, max_length=512)
                else:
                    # legacy（A/B 对照 / 旧 state 续训）：清洗 Qwen 文本；T5 按
                    # 逗号逐 tag 分词 + (tag:1.3) 权重语法
                    qwen_texts = [_build_qwen_text_from_prompt(c) for c in captions]
                    qwen_emb, qwen_attn = encode_qwen(ctx.qwen_model, ctx.qwen_tok, qwen_texts, ctx.device)
                    t5_ids, t5_attn, t5_w = tokenize_t5_weighted(ctx.t5_tok, captions, max_length=512)
                t5_ids = t5_ids.to(ctx.device)
                t5_attn = t5_attn.to(ctx.device)
                t5_w = t5_w.to(ctx.device, dtype=torch.float32)
                # t5_w 透传到 preprocess_text_embeds 内乘到 LLMAdapter 输出上（与
                # ComfyUI 原生 `comfy/ldm/anima/model.py:198-206` 对齐）。
                cross = ctx.model.preprocess_text_embeds(qwen_emb, t5_ids, t5xxl_weights=t5_w)
                if cross.shape[1] < 512:
                    cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))
                # KV trim：把 padding 截到最近有效 token bucket（64/128/256/512）
                # t5_attn=1 表示有效 token；取批次内最大实际长度再 round up
                # torch.compile 模式禁用 kv_trim（trim 引入动态 seq 维度，导致 recompile）
                if getattr(args, "kv_trim", False) and not getattr(args, "torch_compile", False):
                    _actual = int(t5_attn.sum(dim=-1).max().item())
                    _bucket = 512  # _actual > 512 时兜底（不裁，保持原行为）
                    for _b in (64, 128, 256, 512):
                        if _b >= _actual:
                            _bucket = _b
                            break
                    cross = cross[:, :_bucket, :].contiguous()

            # Flow Matching：统一通过 timestep_sampler plugin 接口采样
            # （baseline = 4 种 mode；adaptive = InfoNoise 等；接口在 ADR 0003 plugin registry）
            t = ctx.timestep_sampler.sample(bs, ctx.device)

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

            leap_enabled = bool(getattr(args, "leap_enabled", False))
            # 方式 A 混合训练：每个 micro-batch 按 leap_ratio 概率掷骰子决定走哪条目标。
            # leap 管全局结构、传统管细节锐度，两股梯度叠在同一组 LoRA 权重上各取所长。
            # ratio=1.0 纯 leap；0.0 纯传统；0.6 大头 leap 留点细节（默认）。
            # 用 Python random（bootstrap 设过 random.seed）而非 torch.rand，避免每步消耗 torch
            # global RNG 状态——否则"同种子换 leap_ratio"的对照实验里，标准路径的 noise / timestep
            # 会随 leap_ratio 漂移。
            # 注：缺省/None 才回落到 0.6；leap_ratio=0.0（纯传统）是合法值，不能被 `or` 吞成默认。
            _leap_ratio = getattr(args, "leap_ratio", 0.6)
            use_leap_this_step = leap_enabled and (
                random.random() < (0.6 if _leap_ratio is None else float(_leap_ratio))
            )

            # 前向
            pad_mask = torch.zeros(bs, 1, latents.shape[-2], latents.shape[-1], device=ctx.device, dtype=ctx.dtype)
            denoise_loss_log = None
            sra_align_loss_log = None
            sra_weighted_loss_log = None
            sra_effective_weight_log = None
            with torch.autocast("cuda", dtype=ctx.dtype):
                if use_leap_this_step:
                    # ── LeapAlign / FlowBP 轨迹自蒸馏路径（四 variant）──
                    # 用真实 latent 当 x0，沿解析构造的代理轨迹积分出 x̂0，
                    # 自蒸馏 loss = MSE(x̂0, 真实 x0)。variant 决定轨迹结构，详见 training/leap.py：
                    #   original  两步跳 + straight-through connector（K=2，行为同历史版）
                    #   sparse    K 点 Euler 重放，纯直接项求和（零 connector / 零雅可比）
                    #   bridge    两步跳 + Euler 重构 connector（无 straight-through 偏差）
                    #   lagrange  两段跳 + 每段三点 Simpson 积分（6× 前向）
                    leap_variant = str(getattr(args, "leap_variant", "original") or "original")
                    _leap_min_gap = float(getattr(args, "leap_min_gap", 0.1) or 0.1)
                    _leap_tsw = bool(getattr(args, "leap_traj_sim_weighting", False))
                    _leap_tsm = float(getattr(args, "leap_traj_sim_min", 0.1) or 0.1)
                    _leap_ngc = float(getattr(args, "leap_nested_grad_coe", 0.3))
                    if leap_variant == "sparse":
                        # K 点激活集（K× 前向 + K× activation 显存）
                        t_steps = sample_activation_timesteps(
                            bs, ctx.device,
                            k=int(getattr(args, "leap_activation_k", 3) or 3),
                            dtype=torch.float32,
                        )
                        loss_per_sample = sparse_training_step(
                            ctx.model, latents, noise, cross, pad_mask, t_steps,
                            traj_sim_weighting=_leap_tsw, traj_sim_min=_leap_tsm,
                            use_checkpoint=args.grad_checkpoint,
                        )
                    else:
                        # original / bridge / lagrange 共用两时刻 (k,j) 拓扑
                        t_k, t_j = sample_two_timesteps(
                            bs, ctx.device, min_gap=_leap_min_gap, dtype=torch.float32,
                        )
                        if leap_variant == "bridge":
                            _step_fn = bridge_training_step
                        elif leap_variant == "lagrange":
                            _step_fn = lagrange_training_step
                        else:  # original（默认，行为零变化）
                            _step_fn = leap_training_step
                        loss_per_sample = _step_fn(
                            ctx.model, latents, noise, cross, pad_mask, t_k, t_j,
                            nested_grad_coe=_leap_ngc,
                            traj_sim_weighting=_leap_tsw, traj_sim_min=_leap_tsm,
                            use_checkpoint=args.grad_checkpoint,
                        )
                    # leap 路径有意跳过两个标准机制（互斥校验已在 TrainingConfig 强制关闭）：
                    #   - InfoNoise record：双 timestep 与单 t 的 I-MMSE 语义不匹配
                    #   - loss_weighting：依赖单一 t 算 SNR 权重；leap 自带 traj_sim 加权
                    # 仍尊重 batch 的 loss_weight（正则集降权），与标准路径一致。
                    if "loss_weight" in batch:
                        w = batch["loss_weight"].to(ctx.device).view(-1, *([1] * (loss_per_sample.dim() - 1)))
                        loss_per_sample = loss_per_sample * w
                    loss = loss_per_sample.mean()
                    denoise_loss_log = loss.detach()
                else:
                    # ── 标准 rectified flow 路径（零行为变化）──
                    noisy = (1 - t_exp) * latents + t_exp * noise
                    target = noise - latents
                    pred = forward_with_optional_checkpoint(
                        ctx.model, noisy, t.view(-1, 1), cross, pad_mask,
                        use_checkpoint=args.grad_checkpoint,
                    )
                    # 训练 loss 通过 losses/ plugin registry 派发（mse / huber / ...）
                    loss_per_sample = ctx.loss_fn.compute(pred.float(), target.float(), t)
                    # 自适应采样器（如 InfoNoise）记录原始 per-sample MSE（不受 huber/loss_weighting 等
                    # 加工影响）；跟训练 loss 解耦保证 InfoNoise 论文一致性。
                    # baseline 采样器是 no-op，无需 if 守卫。
                    # 用 no_grad 避免构造 autograd 元数据（比 .detach() 少一份 grad_fn 开销）。
                    with torch.no_grad():
                        _raw_mse_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none")
                        _raw_mse = _raw_mse_per_sample.mean(
                            dim=list(range(1, _raw_mse_per_sample.dim()))
                        )
                    # 仅 main 集样本进 InfoNoise schedule 学习：I-MMSE 假设单一数据分布，
                    # reg 集典型是通用图（booru）vs main 集是单一主题，混入 record 学到的是
                    # mixture MMSE 不是 mmse_main(t)。用 is_reg flag 而非 loss_weight 阈值
                    # 是因为 distribution identity 跟 gradient 权重是两条独立轴
                    # （reg_weight=1.0 时 loss_weight=1.0 但 reg 仍是不同分布）。
                    # 见 docs/todo/infonoise-reg-policy-reeval.md 未来重评估条件。
                    if "is_reg" in batch:
                        _main_mask = ~batch["is_reg"].to(t.device)
                        if _main_mask.any():
                            ctx.timestep_sampler.record(t.detach()[_main_mask], _raw_mse[_main_mask])
                    else:
                        ctx.timestep_sampler.record(t.detach(), _raw_mse)
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
                            detail_inv_t_min=float(getattr(args, "detail_inv_t_min", 1.0) or 1.0),
                            detail_inv_t_max=float(getattr(args, "detail_inv_t_max", 5.0) or 5.0),
                        ).to(device=ctx.device, dtype=torch.float32)
                        loss_per_sample = loss_per_sample * lw.view(-1, *([1] * (loss_per_sample.dim() - 1)))
                    loss = loss_per_sample.mean()
                    denoise_loss_log = loss.detach()

                # SRA v2 表征对齐 loss（标准路径；leap 路径不适用）
                if ctx.sra_aligner is not None and not use_leap_this_step:
                    sra_weight = _resolve_sra_effective_weight(args, ctx.global_step, ctx.total_steps)
                    sra_effective_weight_log = sra_weight
                    if sra_weight != 0.0:
                        align_loss = ctx.sra_aligner.compute(
                            latents,
                            sample_weight=batch.get("loss_weight"),
                        )
                        weighted_align_loss = sra_weight * align_loss
                        loss = loss + weighted_align_loss
                        sra_align_loss_log = align_loss.detach()
                        sra_weighted_loss_log = weighted_align_loss.detach()
                    else:
                        sra_weighted_loss_log = loss.new_tensor(0.0).detach()

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

            # 反向传播。尾组（len % grad_accum）不满时按实际 micro-batch 数归一，
            # 且 epoch 末批不满也 step —— 修尾批丢弃 + 跨 epoch 梯度泄漏（见 _accumulation_step）。
            group_size, is_group_end = _accumulation_step(batch_idx, dl_len, args.grad_accum)
            loss = loss / group_size
            loss.backward()

            if is_group_end:
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

                # 自适应采样器：刷新采样分布；baseline 是 no-op
                ctx.timestep_sampler.maybe_refresh(ctx.global_step)

                # 记录 loss 历史
                loss_val = float(loss.item() * group_size)
                denoise_loss_val = (
                    float(denoise_loss_log.item())
                    if denoise_loss_log is not None else loss_val
                )
                sra_align_loss_val = (
                    float(sra_align_loss_log.item())
                    if sra_align_loss_log is not None else None
                )
                sra_weighted_loss_val = (
                    float(sra_weighted_loss_log.item())
                    if sra_weighted_loss_log is not None else None
                )
                epoch_loss_sum += loss_val
                epoch_step_count += 1
                if args.loss_curve_steps and len(ctx.loss_history) < args.loss_curve_steps:
                    ctx.loss_history.append(loss_val)

                # 更新进度显示
                now = time.perf_counter()
                optimizer_metrics = get_optimizer_monitor_metrics(ctx.optimizer)
                lr = optimizer_metrics["lr"]

                # 更新训练监控面板
                if ctx.monitor_server:
                    try:
                        from train_monitor import update_monitor
                        monitor_metrics = dict(optimizer_metrics)
                        monitor_metrics["denoise_loss"] = denoise_loss_val
                        if sra_align_loss_val is not None:
                            monitor_metrics["sra_align_loss"] = sra_align_loss_val
                        if sra_weighted_loss_val is not None:
                            monitor_metrics["sra_weighted_loss"] = sra_weighted_loss_val
                        if sra_effective_weight_log is not None:
                            monitor_metrics["sra_effective_weight"] = float(sra_effective_weight_log)
                        update_monitor(
                            loss=loss_val, lr=lr, epoch=epoch + 1,
                            total_epochs=int(args.epochs or 0),
                            step=ctx.global_step,
                            total_steps=ctx.total_steps, speed=ctx.speed_ema or 0,
                            optimizer_metrics=monitor_metrics,
                        )
                    except Exception:
                        pass
                dt_step = now - step_start_time
                steps_per_sec = (1.0 / dt_step) if dt_step > 0 else 0.0
                ctx.speed_ema = steps_per_sec if ctx.speed_ema is None else (0.9 * ctx.speed_ema + 0.1 * steps_per_sec)
                log_payload: dict[str, Any] = {
                    "train/loss": loss_val,
                    "train/denoise_loss": denoise_loss_val,
                    "train/lr": float(lr),
                    "train/speed_it_s": float(ctx.speed_ema or 0),
                }
                if sra_align_loss_val is not None:
                    log_payload["train/sra_align_loss"] = sra_align_loss_val
                if sra_weighted_loss_val is not None:
                    log_payload["train/sra_weighted_loss"] = sra_weighted_loss_val
                if sra_effective_weight_log is not None:
                    log_payload["train/sra_effective_weight"] = float(sra_effective_weight_log)
                if "d" in optimizer_metrics:
                    log_payload["train/optimizer_d"] = float(optimizer_metrics["d"])
                if "base_lr" in optimizer_metrics:
                    log_payload["train/base_lr"] = float(optimizer_metrics["base_lr"])
                if "effective_lr" in optimizer_metrics:
                    log_payload["train/effective_lr"] = float(optimizer_metrics["effective_lr"])
                # 自适应采样器可观测性（P1-1）：CDF 是否就绪 + 退化次数
                if (
                    ctx.global_step % args.log_every == 0
                    and ctx.timestep_sampler.status().get("kind") == "infonoise"
                ):
                    status = ctx.timestep_sampler.status()
                    log_payload["infonoise/cdf_ready"] = float(status["cdf_ready"])
                    log_payload["infonoise/refresh_degraded_count"] = status["refresh_degraded_count"]
                ctx.wandb_monitor.log(log_payload, step=ctx.global_step)

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
                    sra_suffix = (
                        f" denoise={denoise_loss_val:.6f}"
                        f" sra={sra_align_loss_val:.6f}"
                        f" sra_w={sra_weighted_loss_val:.6f}"
                        if sra_align_loss_val is not None and sra_weighted_loss_val is not None
                        else f" denoise={denoise_loss_val:.6f}"
                    )
                    print(f"epoch {epoch+1}/{args.epochs} step {ctx.global_step} loss={loss_val:.6f}{sra_suffix} lr={lr:.2e} speed={ctx.speed_ema:.2f} it/s", end="\r", flush=True)
                elif args.log_every and ctx.global_step % args.log_every == 0:
                    sra_suffix = (
                        f" denoise={denoise_loss_val:.6f}"
                        f" sra={sra_align_loss_val:.6f}"
                        f" sra_w={sra_weighted_loss_val:.6f}"
                        if sra_align_loss_val is not None and sra_weighted_loss_val is not None
                        else f" denoise={denoise_loss_val:.6f}"
                    )
                    print(f"epoch={epoch} step={ctx.global_step} loss={loss_val:.6f}{sra_suffix} lr={lr:.2e} speed={steps_per_sec:.2f} it/s")

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
                    ctx.wandb_monitor.upload_model(lora_path)

                # 定期保存训练状态（断点续训）
                save_state_every_steps = getattr(args, "save_state_every_steps", 0)
                if save_state_every_steps > 0 and ctx.global_step % save_state_every_steps == 0:
                    state_path = ctx.state_dir() / f"training_state_step{ctx.global_step}.pt"
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
                            timestep_sampler=ctx.timestep_sampler,
                            sra_aligner=ctx.sra_aligner,
                        )
                        # 同时保存 LoRA 权重
                        lora_path = ctx.output_dir / f"{args.output_name}_step{ctx.global_step}.safetensors"
                        ctx.injector.save(lora_path)
                    ctx.emit(f"Saved training state (step {ctx.global_step}): {state_path.name}")
                    ctx.wandb_monitor.upload_state_manual(state_path)

                # 检查 max_steps
                if args.max_steps and ctx.global_step >= args.max_steps:
                    break

        # epoch 结束后的操作
        ctx.current_epoch = epoch + 1
        if epoch_step_count > 0:
            ctx.wandb_monitor.log(
                {
                    "train/loss_epoch": epoch_loss_sum / epoch_step_count,
                    "train/epoch": ctx.current_epoch,
                },
                step=ctx.global_step,
            )
        if not args.max_steps or ctx.global_step < args.max_steps:
            # 保存 checkpoint
            if args.save_every_epochs > 0 and ctx.current_epoch % args.save_every_epochs == 0:
                save_path = ctx.output_dir / f"{args.output_name}_epoch{ctx.current_epoch}.safetensors"
                # PPSF：保存 averaged weights 的 LoRA
                with optimizer_eval_mode(ctx.optimizer):
                    ctx.injector.save(save_path)
                ctx.emit(f"Saved LoRA: {save_path}")
                ctx.wandb_monitor.upload_model(save_path)

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

            # 定期保存训练状态（epoch 版）
            # ADR 0006 Addendum 1：epoch 字段顺手修 off-by-one（dev current_epoch 在 L297
            # 已推进 = epoch+1，这里传 ctx.current_epoch 而非 epoch，避免 resume `for epoch
            # in range(start, N)` inclusive 重训整个 epoch）。
            save_state_every_epochs = int(getattr(args, "save_state_every_epochs", 0) or 0)
            if save_state_every_epochs > 0 and ctx.current_epoch % save_state_every_epochs == 0:
                state_path = ctx.state_dir() / f"training_state_epoch{ctx.current_epoch}.pt"
                monitor_data = None
                if ctx.monitor_server:
                    try:
                        from train_monitor import get_state
                        monitor_data = get_state()
                    except Exception:
                        pass
                with optimizer_eval_mode(ctx.optimizer):
                    save_training_state(
                        state_path, ctx.injector, ctx.optimizer, ctx.current_epoch, ctx.global_step,
                        ctx.loss_history, monitor_state=monitor_data, scheduler=ctx.scheduler,
                        timestep_sampler=ctx.timestep_sampler,
                        sra_aligner=ctx.sra_aligner,
                    )
                    lora_path = ctx.output_dir / f"{args.output_name}_epoch{ctx.current_epoch}.safetensors"
                    if not lora_path.exists():
                        ctx.injector.save(lora_path)
                ctx.emit(f"Saved training state (epoch {ctx.current_epoch}): {state_path.name}")
                ctx.wandb_monitor.upload_state_manual(state_path)

            # ADR 0006 Addendum 1 方案 Δ：每 epoch 末尾**强制**写 auto_epoch_state.pt（覆盖式）。
            # 跟用户主动开的 save_state_every_epochs / save_state_every_steps（多份历史归档）独立，无 args gate ——
            # 这是系统级 pause 后盾，给 handle_interrupt 暂停时引用。
            # 时机：放在 user-opt epoch save 之后，确保即使 user-opt 没开也有 backup。
            # Addendum 2：落 auto_state_dir()（task 档案 tasks/<id>/state/；CLI fallback
            # 到 state_dir()）—— 用户周期 save 仍走上面的 state_dir() 不动。
            auto_state_path = build_auto_epoch_state_path(ctx.auto_state_dir())
            auto_config_path = build_auto_epoch_config_path(ctx.auto_state_dir())
            monitor_data = None
            if ctx.monitor_server:
                try:
                    from train_monitor import get_state
                    monitor_data = get_state()
                except Exception:
                    pass
            # config snapshot 先写 — 体积小、失败概率低
            write_config_snapshot(auto_config_path, args, ctx.sample_prompts)
            with optimizer_eval_mode(ctx.optimizer):
                save_training_state(
                    auto_state_path, ctx.injector, ctx.optimizer,
                    ctx.current_epoch, ctx.global_step, ctx.loss_history,
                    monitor_state=monitor_data, scheduler=ctx.scheduler,
                    timestep_sampler=ctx.timestep_sampler,
                    sra_aligner=ctx.sra_aligner,
                )
            ctx.wandb_monitor.upload_state_auto(auto_state_path)
            # 更新 ctx 字段供 handle_interrupt emit pause_state 用
            ctx.last_auto_epoch_state_path = auto_state_path
            ctx.last_auto_epoch_config_path = auto_config_path
            # supervisor 端 `_on_line` 抓此 event → 标 slot.last_auto_epoch_state_path
            # → is_pausable 升级条件满足 → SSE 解锁 UI 暂停按钮（ADR Addendum 1 §UI）
            emit_event("auto_epoch_backup_written", {
                "state_path": str(auto_state_path),
                "config_path": str(auto_config_path),
                "epoch": ctx.current_epoch,
                "step": ctx.global_step,
            })

        # 检查 max_steps
        if args.max_steps and ctx.global_step >= args.max_steps:
            break
