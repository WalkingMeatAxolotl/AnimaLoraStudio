"""Muon optimizer build wrapper（ADR 0003 PR-C）。

Muon = 动量 + Newton-Schulz 正交化（Keller Jordan, 2024）。真矩阵参数走 Muon，
1D / 向量形状参数（DoRA magnitude / SRA bias / Ortho lambda 等）走同一 optimizer 内的辅助 AdamW。
常规优化器：真实 lr（match_rms_adamw 口径下与 AdamW 同量级）、可配 lr_scheduler、
无 train()/eval() 切换。实现在 utils/muon_optimizer.py。
"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    """实例化 MuonWithAuxAdam，读 muon_* 参数。"""
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="muon",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        momentum=float(getattr(args, "muon_momentum", 0.95)),
        ns_steps=int(getattr(args, "muon_ns_steps", 5)),
        adjust_lr_fn=str(getattr(args, "muon_adjust_lr", "match_rms_adamw")),
    )
