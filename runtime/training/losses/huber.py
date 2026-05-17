"""Huber loss（constant delta）。

Huber 公式（per-element）：
    L(x) = 0.5 * x^2                 if |x| < delta
    L(x) = delta * (|x| - 0.5*delta) otherwise

相比 MSE 对 outlier 更鲁棒，缓解极端 sample 的梯度爆炸。EDM/Karras /
kohya-ss 等主流 trainer 都用 constant δ。

**若未来要引入 t-dependent δ schedule，必须附 (a) 论文 DOI / (b) 上游
trainer 实现链接 / (c) 本项目自跑 ablation 数据**——参考
[[feedback_verify_paper_before_fixing_algo]]（P0-2 EMA 翻转事故同根）。
"""

from __future__ import annotations

import torch


class HuberLoss:
    """Huber loss with constant delta（per-element，reduction='none'）。"""

    def __init__(self, c: float = 0.15):
        self.c = float(c)

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,  # noqa: ARG002 — LossProtocol 一致性，constant huber 不使用
    ) -> torch.Tensor:
        diff = pred - target
        abs_diff = diff.abs()
        quad = 0.5 * diff * diff
        lin = self.c * (abs_diff - 0.5 * self.c)
        return torch.where(abs_diff < self.c, quad, lin)


def build(args) -> HuberLoss:
    return HuberLoss(
        c=float(getattr(args, "huber_c", 0.15) or 0.15),
    )
