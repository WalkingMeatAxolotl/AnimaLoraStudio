"""Huber loss with time-dependent delta schedule。

Huber 公式（per-element）：
    L(x) = 0.5 * x^2                 if |x| < delta
    L(x) = delta * (|x| - 0.5*delta) otherwise

相比 MSE 对 outlier 更鲁棒；diffusion / flow matching 训练中，低 t（细节端）
loss 数值小但梯度方差大，高 t（结构端）loss 数值大但梯度方差小。huber_schedule
让 delta 跟随 t 变化，平衡两端的梯度尺度。

支持三种 schedule：
- constant — delta = huber_c（全程不变）
- snr      — delta = huber_c * sqrt((1-t) / t)；Flow Matching 下 SNR(t) =
             ((1-t)/t)^2，sqrt(SNR) 跟 σ⁻¹ 同阶；低 t 处 delta 大（接近 MSE），
             高 t 处 delta 小（更鲁棒）
- sigma    — delta = huber_c * t/(1-t)；σ = t/(1-t)。低 t 处 delta 小（鲁棒），
             高 t 处 delta 大；跟 snr 方向相反
"""

from __future__ import annotations

import torch


class HuberLoss:
    """Stateful Huber：delta 由 huber_c + huber_schedule + t 决定。"""

    def __init__(self, c: float = 0.15, schedule: str = "constant"):
        self.c = float(c)
        self.schedule = (schedule or "constant").lower()
        if self.schedule not in ("constant", "snr", "sigma"):
            raise ValueError(
                f"huber_schedule={schedule!r} 不支持；可选 constant / snr / sigma"
            )

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        delta = self._compute_delta(t).to(dtype=pred.dtype, device=pred.device)
        # delta 形状 (B,)，pred/target 形状 (B, C, *spatial)；广播 delta 到 pred 维数
        delta_b = delta.view(-1, *([1] * (pred.dim() - 1)))

        diff = pred - target
        abs_diff = diff.abs()
        quad = 0.5 * diff * diff
        lin = delta_b * (abs_diff - 0.5 * delta_b)
        return torch.where(abs_diff < delta_b, quad, lin)

    def _compute_delta(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule == "constant":
            return torch.full_like(t, self.c)

        # clamp t 到开区间防 0 / inf
        t_c = t.clamp(1e-4, 1 - 1e-4)
        if self.schedule == "snr":
            return self.c * ((1.0 - t_c) / t_c).sqrt()
        # sigma
        return self.c * (t_c / (1.0 - t_c))


def build(args) -> HuberLoss:
    return HuberLoss(
        c=float(getattr(args, "huber_c", 0.15) or 0.15),
        schedule=str(getattr(args, "huber_schedule", "constant") or "constant"),
    )
