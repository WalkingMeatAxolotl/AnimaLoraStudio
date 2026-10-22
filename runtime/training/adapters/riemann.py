"""Riemann（固定秩流形）LoRA adapter builder（POC）。

只与 optimizer_type=riemannion 搭配：Riemannion 直接改写 A_L / B / B_R，普通
optimizer 对 lora_b / lora_z1 做欧氏更新会破坏 A_L 正交与 Z1≡0 的不变量。
实际类在 utils/riemann_lora_adapter.py。
"""

from __future__ import annotations

from typing import Any

from training.adapters.protocol import AdapterProtocol


def build(args, *, preset: dict[str, Any]) -> AdapterProtocol:
    optimizer_type = (getattr(args, "optimizer_type", "adamw") or "adamw").lower()
    if optimizer_type != "riemannion":
        raise ValueError(
            f"lora_type=riemann 只能配 optimizer_type=riemannion（当前 {optimizer_type!r}）"
        )
    from utils.riemann_lora_adapter import RiemannLoRAAdapter

    return RiemannLoRAAdapter(
        rank=int(args.lora_rank),
        preset=preset,
        seed=int(getattr(args, "seed", 0) or 0),
    )
