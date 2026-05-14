"""LyCORIS 后端的 adapter build 函数（lokr / loha / lora）。

抽自 phases/models.py 里的 AnimaLycorisAdapter 实例化逻辑（ADR 0003 PR-C）。

由 training/adapters/__init__.py 的 BUILDERS 字典派发：
    BUILDERS["lokr"] = build
    BUILDERS["loha"] = build
    BUILDERS["lora"] = build

实际 adapter 类在 utils/lycoris_adapter.py，本文件只做"读 args → 调构造器"
的轻量 wrapper。
"""

from __future__ import annotations

from training.adapters.protocol import AdapterProtocol


def build(args) -> AdapterProtocol:
    """从 args 读 lora_type / lora_rank / ... 实例化 AnimaLycorisAdapter。"""
    from utils.lycoris_adapter import AnimaLycorisAdapter
    return AnimaLycorisAdapter(
        algo=args.lora_type,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        factor=args.lokr_factor,
        dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
        rank_dropout=float(getattr(args, "lora_rank_dropout", 0.0) or 0.0),
        module_dropout=float(getattr(args, "lora_module_dropout", 0.0) or 0.0),
        weight_decompose=bool(getattr(args, "lora_dora", False)),
        rs_lora=bool(getattr(args, "lora_rs", False)),
    )
