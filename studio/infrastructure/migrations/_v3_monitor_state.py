"""v2 → v3: tasks 表加 `monitor_state_path` 列（PP6.1）。

每个训练任务都有自己的 monitor state 文件路径（per-version 或 per-task 兜底）。
端点 `/api/state?task_id=N` 用此列定位文件。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "tasks", "monitor_state_path", "monitor_state_path TEXT"
    )
