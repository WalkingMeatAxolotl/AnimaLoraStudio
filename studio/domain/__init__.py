"""studio.domain — pydantic schema 单一权威源（无 I/O）。

从原 studio/schema.py 976 行拆出。结构：
  - common.py      _meta() helper, AttentionBackend, GROUP_ORDER
  - migrations.py  migrate_legacy_save_keys
  - training.py    TrainingConfig（最大类 643 行；本次结构 PR 保持单类单文件）
  - lora.py        LoraEntry
  - xy_matrix.py   XYAxisType, XYAxisSpec, XYMatrixSpec, _check_axis_values
  - generate.py    GenerateConfig
  - reg.py         RegAiConfig

注意：不使用 `from __future__ import annotations`——Pydantic v2 + Python 3.12+
在延迟求值模式下会将 typing._SpecialForm 当成 schema key，触发 AttributeError。
"""
from .common import AttentionBackend, GROUP_ORDER, _meta
from .generate import GenerateConfig
from .lora import LoraEntry
from .migrations import migrate_legacy_save_keys, migrate_noise_enhancement_type
from .reg import RegAiConfig
from .training import TrainingConfig
from .xy_matrix import XYAxisSpec, XYAxisType, XYMatrixSpec, _check_axis_values

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
    "migrate_legacy_save_keys",
    "migrate_noise_enhancement_type",
]
