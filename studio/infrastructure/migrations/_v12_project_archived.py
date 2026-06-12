"""v11 → v12: projects.archived_at — 项目归档（软隐藏）。

加列 `projects.archived_at REAL`（NULL = 未归档）。归档项目不在项目页默认
列表显示；UI 上原"删除"入口先归档，归档视图里再点才真删。归档不动
updated_at —— 恢复后排序位置保持原样。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "projects", "archived_at", "archived_at REAL"
    )
