"""Re-export shim — PR-7 真实模块 studio.infrastructure.secrets。

763 行单文件 + 170 行 legacy migration 暂未拆 3 文件（planning 的 models/
store/migrations 拆分推到 0.11.1 — Pydantic v2 validator 跨文件循环风险，独立 PR
更稳）。本 PR 仅完成位置搬迁 + shim。
"""
import sys as _sys

from .infrastructure import secrets as _real

_sys.modules[__name__] = _real
