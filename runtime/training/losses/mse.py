"""MSE loss：包装 F.mse_loss(reduction='none')，提供 LossProtocol 接口。

跟 PR-A/B/C 之前的 loop.py 行为字节级一致（同 pred/target/t 输入下，
mse.compute() 输出等价于 F.mse_loss(reduction='none')）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MseLoss:
    """Stateless MSE：不依赖 t，纯转发给 F.mse_loss。"""

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction="none")


def build(args) -> MseLoss:
    """MSE 没有专属字段，args 用不到（保持签名一致便于 dispatch）。"""
    return MseLoss()
