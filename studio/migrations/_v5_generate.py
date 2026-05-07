"""v4 → v5: tasks 表加 task_type 列（区分 train / generate）。"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "tasks", "task_type",
        "task_type TEXT NOT NULL DEFAULT 'train'"
    )
