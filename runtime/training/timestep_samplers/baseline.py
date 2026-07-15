"""Baseline timestep 采样器：包装 training.timestep_sampling.sample_t。

非自适应；record / maybe_refresh 是 no-op。
覆盖 6 种 mode：logit_normal / uniform / logit_normal_low / mode /
mixed_uniform_low / mixed_uniform_logit。
"""

from __future__ import annotations

import torch

from training.timestep_sampling import sample_t


class BaselineTimestepSampler:
    """sample_t 的 thin wrapper，使它符合 TimestepSamplerProtocol。"""

    def __init__(
        self,
        mode: str = "logit_normal",
        shift: float = 3.0,
        mix_low_prob: float = 0.0,
        timestep_schedule_shift: float = 1.0,
    ):
        self.mode = mode
        self.shift = shift
        self.mix_low_prob = mix_low_prob
        self.timestep_schedule_shift = timestep_schedule_shift

    def sample(self, bs: int, device, *, token_counts=None) -> torch.Tensor:
        return sample_t(
            bs,
            device,
            mode=self.mode,
            shift=self.shift,
            mix_low_prob=self.mix_low_prob,
            timestep_schedule_shift=self.timestep_schedule_shift,
        )

    def record(self, t: torch.Tensor, raw_mse: torch.Tensor) -> None:
        return None

    def maybe_refresh(self, global_step: int) -> None:
        return None

    def status(self) -> dict:
        return {
            "kind": "baseline",
            "mode": self.mode,
            "shift": self.shift,
            "mix_low_prob": self.mix_low_prob,
            "timestep_schedule_shift": self.timestep_schedule_shift,
        }

    # ─── pause/resume（ADR 0006 Addendum 1）：无状态采样器是真的无状态，no-op ───
    # Protocol body 里的 default 仅给 type checker；运行时不分发，必须显式实现。
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


def build(args, total_steps) -> BaselineTimestepSampler:
    """按 args 构建 BaselineTimestepSampler。total_steps 此采样器用不到。"""
    return BaselineTimestepSampler(
        mode=str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal"),
        shift=float(getattr(args, "timestep_shift", 3.0) or 3.0),
        mix_low_prob=float(getattr(args, "timestep_mix_low_prob", 0.0) or 0.0),
        timestep_schedule_shift=float(getattr(args, "timestep_schedule_shift", 1.0) or 1.0),
    )
