"""训练状态保存/恢复（断点续训）。

抽自原 runtime/anima_train.py L1073-1142（ADR 0003 PR-A）。被 tests/test_lycoris_resume.py
直接 import 使用。

公开：
- save_training_state — 保存 LoRA / optimizer / scheduler / rng / monitor 一次性 ckpt
- load_training_state — 反向恢复，返回 (epoch, global_step, loss_history, monitor_state)
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


def save_training_state(
    path, injector, optimizer, epoch, global_step,
    loss_history=None, rng_state=None, monitor_state=None,
    scheduler=None, timestep_sampler=None, sra_aligner=None,
    scaler=None, model_family=None,
):
    """保存完整训练状态，支持断点续训。

    timestep_sampler（ADR 0006 Addendum 1）：自适应采样器（InfoNoise）的 EMA / CDF / FIFO buffer。
    无状态采样器（baseline）的 state_dict() 是 {}，跳过不存，避免 ckpt 文件无谓增大。
    """
    state = {
        "lora_state_dict": injector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss_history": loss_history or [],
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "random": random.getstate(),
        },
        "monitor_state": monitor_state,  # 保存监控面板数据（用于恢复 loss 曲线）
        # 多模型 D13：族标记；load 侧跨族 fail-fast（strict=False 会静默冷启动）
        "model_family": str(model_family or "anima"),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if sra_aligner is not None and hasattr(sra_aligner, "state_dict"):
        try:
            state["sra_aligner_state"] = sra_aligner.state_dict()
        except Exception as e:
            logger.warning(f"sra_aligner.state_dict() 失败（跳过）: {e}")
    if timestep_sampler is not None and hasattr(timestep_sampler, "state_dict"):
        # hasattr 防御：Protocol 不提供 default dispatch，未来新加的 sampler 若忘记
        # 实现这两个 hook，要静默跳过而非崩溃（训练 8 小时不能因 resume hook 缺失废）
        try:
            sampler_state = timestep_sampler.state_dict()
        except Exception as e:
            logger.warning(f"timestep_sampler.state_dict() 失败（跳过）: {e}")
            sampler_state = None
        if sampler_state:  # 空 dict（baseline）不存
            state["timestep_sampler_state"] = sampler_state
    if scaler is not None:
        # fp16 GradScaler 的 scale 因子 / growth tracker。resume 不带 → 重置成默认 2^16，
        # 头几步重新溢出空跳直到收敛。bf16/fp32 时 ctx.scaler 为 None，跳过不存。
        state["scaler_state"] = scaler.state_dict()
    # ADR 0006 Addendum 2：tmp + os.replace 原子落盘。auto_epoch_state.pt 是
    # 覆盖式单文件恢复点，直接 torch.save 在断电 / 强杀砸中写盘窗口时会把
    # 唯一恢复点写成半截；先写 sibling tmp 再 rename，旧文件要么完整保留
    # 要么被完整新文件替换。
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info(f"训练状态已保存: {path} (epoch={epoch}, step={global_step})")


def load_training_state(path, injector, optimizer, scheduler=None, timestep_sampler=None, sra_aligner=None, scaler=None, expected_family=None):
    """加载训练状态，返回 (epoch, global_step, loss_history, monitor_state)。

    timestep_sampler（ADR 0006 Addendum 1）：如 ckpt 含 timestep_sampler_state 且 sampler
    实现了 load_state_dict，把 EMA / CDF / FIFO 灌回去；否则保持冷启动（warning 提示）。
    """
    logger.info(f"加载训练状态: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)

    # 多模型 D13：跨族 resume fail-fast。无标记的存量 state grandfather 为 anima。
    saved_family = str(state.get("model_family") or "anima")
    if expected_family is not None and saved_family != str(expected_family):
        raise RuntimeError(
            f"跨模型族 resume 被拒绝：恢复点属于 '{saved_family}'，"
            f"当前 model_family='{expected_family}'。"
            f"（strict=False 加载会静默变成全 missing 冷启动，比崩溃更糟）"
        )

    # 加载 LoRA 权重（lycoris-lora backend）— 一次性导入 state_dict
    # 旧自实现 ckpt 在 Stage 4 plan 决策中**不做迁移**，strict=False 让缺失键
    # 走默认初始化路径而非崩溃；用户应当从头训练新格式 ckpt。
    lora_sd = state["lora_state_dict"]
    result = injector.load_state_dict(lora_sd, strict=False)
    missing = len(getattr(result, "missing_keys", [])) if hasattr(result, "missing_keys") else 0
    unexpected = len(getattr(result, "unexpected_keys", [])) if hasattr(result, "unexpected_keys") else 0
    if missing or unexpected:
        logger.warning(
            f"resume LoRA: missing={missing}, unexpected={unexpected}（旧格式 ckpt？）"
        )

    # 恢复 SRA v2 projection MLP（如启用）。必须在 optimizer state 恢复前完成，
    # 保证后续训练从同一组投影权重继续，而不是随机新 MLP + 旧 optimizer moments。
    if sra_aligner is not None:
        if "sra_aligner_state" in state and hasattr(sra_aligner, "load_state_dict"):
            try:
                sra_aligner.load_state_dict(state["sra_aligner_state"])
                logger.info("SRA v2 projection MLP 状态已恢复")
            except Exception as e:
                logger.warning(f"SRA v2 projection MLP 状态恢复失败（冷启动）: {e}")
        else:
            logger.warning("checkpoint 不含 SRA v2 状态；projection MLP 将冷启动")

    # 加载优化器状态
    optimizer.load_state_dict(state["optimizer_state_dict"])

    # 加载调度器状态
    if scheduler is not None and "scheduler_state_dict" in state:
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"调度器状态恢复失败（将从头开始）: {e}")

    # 恢复 GradScaler（fp16）。老 ckpt / bf16·fp32 run 无此 key → 保持默认 scale 冷启动。
    if scaler is not None and "scaler_state" in state:
        try:
            scaler.load_state_dict(state["scaler_state"])
            logger.info("GradScaler 状态已恢复")
        except Exception as e:
            logger.warning(f"GradScaler 状态恢复失败（按默认 scale 冷启动）: {e}")

    # 恢复随机数状态
    if "rng_state" in state:
        rng = state["rng_state"]
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
        if rng.get("random") is not None:
            random.setstate(rng["random"])

    # 恢复 timestep sampler 内部状态（InfoNoise CDF / EMA / FIFO 等；baseline 为 no-op）
    if (
        timestep_sampler is not None
        and "timestep_sampler_state" in state
        and hasattr(timestep_sampler, "load_state_dict")
    ):
        try:
            timestep_sampler.load_state_dict(state["timestep_sampler_state"])
            logger.info("timestep_sampler 状态已恢复（自适应 schedule 接力）")
        except Exception as e:
            logger.warning(f"timestep_sampler 状态恢复失败（冷启动重 warmup）: {e}")

    # ADR 0006 Addendum 1 第 7 条：Schedule-Free 系优化器（PPSF 等）resume 守护。
    # PPSF 内部维护 group['train_mode'] flag + Polyak averaged x/y/z 三组权重；
    # load_state_dict 把 train_mode 恢复到 save 那刻的值（save 在 `optimizer_eval_mode`
    # 内 = train_mode False） → resume 后第一步 step() 抛 "Not in train mode!"。
    # 显式调一次 .train()：set_train_mode(True) lerp p.data 从 averaged x 反推回 y
    # 并设 train_mode=True，跟 dev 训练循环起始状态对齐。Spike 验证 2000 步 bit-exact
    # 跟 ground truth 一致（不漂移）。AdamW / Prodigy 无 .train 方法走 hasattr 静默跳过。
    if hasattr(optimizer, "train") and callable(getattr(optimizer, "train")):
        try:
            optimizer.train()
        except Exception as e:
            logger.warning(f"optimizer.train() 调用失败（PPSF 可能 broken）: {e}")

    epoch = state.get("epoch", 0)
    global_step = state.get("global_step", 0)
    loss_history = state.get("loss_history", [])
    monitor_state = state.get("monitor_state", None)  # 恢复监控数据

    logger.info(f"训练状态已恢复: epoch={epoch}, step={global_step}")
    return epoch, global_step, loss_history, monitor_state
