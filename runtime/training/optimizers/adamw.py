"""AdamW optimizer build wrapper（ADR 0003 PR-C）。"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    """实例化 AdamW。

    AdamW 不需要 args.* 之外的额外参数；保持签名跟其他 builder 一致以便
    registry 统一派发。
    """
    from utils.optimizer_utils import create_optimizer
    return create_optimizer(
        optimizer_type="adamw",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
    )
