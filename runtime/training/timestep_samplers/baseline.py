"""Baseline timestep 采样器：包装 training.timestep_sampling.sample_t。

非自适应；record / maybe_refresh 是 no-op。
覆盖 4 种 mode：logit_normal / uniform / logit_normal_low / mode。
"""

from __future__ import annotations

import torch

from training.timestep_sampling import sample_t


class BaselineTimestepSampler:
    """sample_t 的 thin wrapper，使它符合 TimestepSamplerProtocol。"""

    def __init__(self, mode: str = "logit_normal", shift: float = 3.0):
        self.mode = mode
        self.shift = shift

    def sample(self, bs: int, device) -> torch.Tensor:
        return sample_t(bs, device, mode=self.mode, shift=self.shift)

    def record(self, t: torch.Tensor, raw_mse: torch.Tensor) -> None:
        return None

    def maybe_refresh(self, global_step: int) -> None:
        return None

    def status(self) -> dict:
        return {"kind": "baseline", "mode": self.mode, "shift": self.shift}


def build(args, total_steps) -> BaselineTimestepSampler:
    """按 args 构建 BaselineTimestepSampler。total_steps 此采样器用不到。"""
    return BaselineTimestepSampler(
        mode=str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal"),
        shift=float(getattr(args, "timestep_shift", 3.0) or 3.0),
    )
