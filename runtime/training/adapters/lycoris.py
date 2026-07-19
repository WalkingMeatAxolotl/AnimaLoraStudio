"""LyCORIS 后端的 adapter build 函数（lokr / loha / lora）。

抽自 phases/models.py 里的 LycorisAdapter 实例化逻辑（ADR 0003 PR-C）。

由 training/adapters/__init__.py 的 BUILDERS 字典派发：
    BUILDERS["lokr"] = build
    BUILDERS["loha"] = build
    BUILDERS["lora"] = build

实际 adapter 类在 utils/lycoris_adapter.py，本文件只做"读 args → 调构造器"
的轻量 wrapper。
"""

from __future__ import annotations

from typing import Any

from training.adapters.protocol import AdapterProtocol


def build(args, *, preset: dict[str, Any]) -> AdapterProtocol:
    """从 args 与显式 family preset 实例化 LycorisAdapter。"""
    from utils.lycoris_adapter import LycorisAdapter
    return LycorisAdapter(
        preset=preset,
        algo=args.lora_type,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        factor=args.lokr_factor,
        dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
        rank_dropout=float(getattr(args, "lora_rank_dropout", 0.0) or 0.0),
        module_dropout=float(getattr(args, "lora_module_dropout", 0.0) or 0.0),
        weight_decompose=bool(getattr(args, "lora_dora", False)),
        rs_lora=bool(getattr(args, "lora_rs", False)),
        lora_reg_dims=getattr(args, "lora_reg_dims", None) or None,
    )
