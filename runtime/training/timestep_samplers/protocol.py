"""TimestepSamplerProtocol：所有 timestep 采样器的统一接口（ADR 0003 plugin registry）。

设计模仿 training/adapters/protocol.py：
- 必需 1 个方法：sample(bs, device, *, token_counts=None) -> Tensor
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

    def sample(self, bs: int, device, *, token_counts=None) -> torch.Tensor:
        """采样 bs 个 t ∈ (0, 1)。

        ``token_counts`` 是可选 batch context；需要它的 resolution-aware sampler
        用 ``requires_token_counts = True`` 声明，loop 才会计算并传入。普通 sampler
        不承担额外的逐 batch 计算。
        """
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

    # ─── pause/resume 支持（ADR 0006 Addendum 1）：自适应采样器须 override，无状态采样器保持默认 no-op ───

    def state_dict(self) -> dict:
        """序列化内部状态用于断点续训。

        无状态采样器（baseline / 纯分布）返回 {} —— save_training_state 跳过不存。
        自适应采样器（InfoNoise 等）override 此方法导出 EMA / CDF / FIFO buffer，否则
        resume 后会回到冷启动，已学到的 schedule 全部丢失。
        """
        return {}

    def load_state_dict(self, state: dict) -> None:
        """从 state_dict 恢复内部状态；无状态采样器保持默认 no-op。

        实现者应在形状不匹配（如 K / B 改变）时 log warning 并保持冷启动，不要抛异常 ——
        训练已经跑了几小时，resume 不应因为配置改了某个 hyperparameter 就崩溃。
        """
        return None
