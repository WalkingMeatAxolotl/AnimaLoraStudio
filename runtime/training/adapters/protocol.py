"""AdapterProtocol：所有 LoRA 变体的统一接口（ADR 0003 PR-C）。

设计原则：
- 必需 4 个方法（inject / get_param_groups / save / load）—— 所有 adapter 都要实现
- 3 个可选 hook（on_step_begin / regularization_loss / excludes_weight_decay）——
  默认 no-op；动态/per-step 行为类变体（T-LoRA / AdaLoRA / OFT 等）按需 override
- runtime_checkable —— 测试可以直接 `isinstance(adapter, AdapterProtocol)` 验

hook 设计参考 ADR 0003 "Case 3-5" 章节里 T-LoRA / OFT / Ortho-Hydra 的真实需求。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StepContext:
    """每 micro-batch 前向开始时传给 adapter.on_step_begin 的最小上下文。

    最小化字段：只传 hook 真正会用的（避免 dataclass 字段爆炸 + 让 hook
    实现不依赖 TrainingContext 全部内部状态）。
    """
    global_step: int
    total_steps: Optional[int]
    epoch: int
    sigma_t: Tensor          # shape [B]，本 micro-batch 的 sigma
    args: object             # parse_args 出来的 Namespace；按需读字段


@runtime_checkable
class AdapterProtocol(Protocol):
    """LoRA / LoKr / LoHa / 论文级变体共用接口。

    用 Protocol 而不是 ABC：现有 AnimaLycorisAdapter 已经实现了前 4 个必需方法
    （duck-typed），不想强制用户继承。runtime_checkable 让单测 `isinstance`
    校验仍然能用。
    """

    # ─── 必需 ───
    def inject(self, model: nn.Module) -> None:
        """把 LoRA 层注入到 model（替换 Linear 等）。"""
        ...

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        """返回 optimizer param_groups。一组或多组（LoRA+ 可分 A/B 不同 lr，
        Ortho-Hydra 可单独 router lr）。"""
        ...

    def save(self, path: Path) -> None:
        """落盘 safetensors。"""
        ...

    def load(self, path: Path) -> None:
        """读取 safetensors。"""
        ...

    # ─── 可选 hook：默认 no-op；按需 override ───

    def on_step_begin(self, ctx: StepContext) -> None:
        """每 micro-batch 前向之前调用。

        T-LoRA / AdaLoRA / B-LoRA 在此按 sigma_t / step 调整 rank mask、
        激活子集、列丢弃等"运行时结构调整"。默认 no-op。
        """
        return None

    def regularization_loss(self, ctx: StepContext) -> Optional[Tensor]:
        """返回要加到主 loss 上的正则项；None=无。

        OFT 返回 orthogonality penalty；Ortho-Hydra 返回 expert balance loss；
        默认 None。train_loop 收到 None 时不做任何额外操作。
        """
        return None

    def excludes_weight_decay(self, param_name: str) -> bool:
        """该 param 是否应排除 weight_decay。

        替代原代码 `injector.use_lokr` 硬编码检查。LoKr 实现里：
        return "w1" in param_name。默认 False。
        """
        return False
