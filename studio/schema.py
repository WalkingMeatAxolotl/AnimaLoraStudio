"""Re-export shim — 真实定义在 studio.domain.*（PR-2 拆分自原 976 行单文件）。

历史上本模块汇聚所有 pydantic schema（TrainingConfig 等）。0.11.0 重构后：
  - 训练 / 生成 / 正则 / LoRA / XY 矩阵各类拆到 `studio.domain.*` 子模块
  - 本文件保留作为兼容垫片，所有 `from studio.schema import X` 仍可工作
  - 新代码请直接 `from studio.domain import X`
"""
from .domain import (
    GROUP_ORDER,
    AttentionBackend,
    GenerateConfig,
    LoraEntry,
    RegAiConfig,
    TrainingConfig,
    XYAxisSpec,
    XYAxisType,
    XYMatrixSpec,
    _check_axis_values,
    _meta,
    migrate_legacy_attention,
    migrate_legacy_save_keys,
    migrate_noise_enhancement_type,
)

__all__ = [
    "AttentionBackend",
    "GROUP_ORDER",
    "GenerateConfig",
    "LoraEntry",
    "RegAiConfig",
    "TrainingConfig",
    "XYAxisSpec",
    "XYAxisType",
    "XYMatrixSpec",
    "_check_axis_values",
    "_meta",
    "migrate_legacy_attention",
    "migrate_legacy_save_keys",
    "migrate_noise_enhancement_type",
]
