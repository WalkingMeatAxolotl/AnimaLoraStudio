"""训练状态保存/恢复（断点续训）。

抽自原 runtime/anima_train.py L1073-1142（ADR 0003 PR-A）。被 tests/test_lycoris_resume.py
直接 import 使用。

公开：
- save_training_state — 保存 LoRA / optimizer / scheduler / rng / monitor 一次性 ckpt
- load_training_state — 反向恢复，返回 (epoch, global_step, loss_history, monitor_state)
"""

from __future__ import annotations

import logging
import random

import torch


logger = logging.getLogger(__name__)


def save_training_state(path, injector, optimizer, epoch, global_step, loss_history=None, rng_state=None, monitor_state=None, scheduler=None):
    """保存完整训练状态，支持断点续训。"""
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
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)
    logger.info(f"训练状态已保存: {path} (epoch={epoch}, step={global_step})")


def load_training_state(path, injector, optimizer, scheduler=None):
    """加载训练状态，返回 (epoch, global_step, loss_history, monitor_state)。"""
    logger.info(f"加载训练状态: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)

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

    # 加载优化器状态
    optimizer.load_state_dict(state["optimizer_state_dict"])

    # 加载调度器状态
    if scheduler is not None and "scheduler_state_dict" in state:
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"调度器状态恢复失败（将从头开始）: {e}")

    # 恢复随机数状态
    if "rng_state" in state:
        rng = state["rng_state"]
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
        if rng.get("random") is not None:
            random.setstate(rng["random"])

    epoch = state.get("epoch", 0)
    global_step = state.get("global_step", 0)
    loss_history = state.get("loss_history", [])
    monitor_state = state.get("monitor_state", None)  # 恢复监控数据

    logger.info(f"训练状态已恢复: epoch={epoch}, step={global_step}")
    return epoch, global_step, loss_history, monitor_state
