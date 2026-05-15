"""TimestepSamplerProtocol：所有 timestep 采样器的统一接口（ADR 0003 plugin registry）。

设计模仿 training/adapters/protocol.py：
- 必需 1 个方法：sample(bs, device) -> Tensor
- 3 个可选 hook：record / maybe_refresh / status —— 默认 no-op；
  自适应采样器（InfoNoise 等）按需 override，纯分布采样器（logit_normal 等）保持 no-op

加新采样器步骤（参考 ADR 0003 PR-C registry 模式）：
1. 写 training/timestep_samplers/{name}.py 含 `build(args, total_steps) -> TimestepSamplerProtocol`
2. timestep_samplers/__init__.py 的 BUILDERS 字典加一行
3. 完。phases/optimizer.py / loop.py / TrainingContext 0 改动。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class TimestepSamplerProtocol(Protocol):
    """timestep 采样器统一接口。

    用 Protocol 而不是 ABC：baseline 用 dataclass 实现 / InfoNoise 用 class 实现，
    不想强制继承。runtime_checkable 让单测 `isinstance` 校验仍然能用。
    """

    def sample(self, bs: int, device) -> torch.Tensor:
        """采样 bs 个 t ∈ (0, 1)。"""
        ...

    # ─── 可选 hook：默认 no-op；自适应采样器按需 override ───

    def record(self, t: torch.Tensor, raw_mse: torch.Tensor) -> None:
        """每 micro-batch 之后记录 per-sample 原始 MSE，供自适应采样器更新分布。

        非自适应采样器（baseline）保持 no-op。
        """
        return None

    def maybe_refresh(self, global_step: int) -> None:
        """每 optimizer step 之后调用，让采样器决定是否刷新内部状态（如 CDF）。

        非自适应采样器保持 no-op。
        """
        return None

    def status(self) -> dict:
        """暴露内部状态供 wandb 监控 / debug；自适应采样器 override 提供有意义信息。"""
        return {}
