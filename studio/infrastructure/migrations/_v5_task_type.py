"""v4 → v5: tasks 表加 task_type 列（区分 train / reg_ai / generate）。

- train (默认): 现有训练任务，跑 runtime/anima_train.py。所有老 task 都 fallback 这个。
- reg_ai: 先验生成（base 模型对每张训练图反向出对照图作正则集），
  跑 runtime/anima_reg_ai.py。PR-9 commit 3 引入。
- generate: 测试出图（用户手动跑 prompt 看效果），跑 runtime/anima_generate.py。
  PR-9 commit 5 引入。

DEFAULT 'train' 让旧库升级时所有现有 task 自动归类训练，无需 backfill。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "tasks", "task_type",
        "task_type TEXT NOT NULL DEFAULT 'train'",
    )
