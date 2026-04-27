"""Schema 迁移：按顺序应用 SQL 升级，由 PRAGMA user_version 跟踪进度。

`db.init_db()` 先 executescript 基础 SCHEMA（v1，定义 tasks 表），然后调用
`apply_all()` 把 user_version 推到最新。新增迁移就往 MIGRATIONS 末尾加一个
回调，user_version 自动 +1。

约定：
- 任何 ALTER TABLE 必须容忍「列已存在」（IF NOT EXISTS 不适用于 ADD COLUMN，
  所以用 try/except 兜一下）；这样老 DB 升上来不会因为重复 ADD 而失败。
- 不允许向后改写已有列；只能加列 / 加表 / 加索引。
"""
from __future__ import annotations

import sqlite3
from typing import Callable

from ._v2_projects import migrate as _migrate_v2

Migration = Callable[[sqlite3.Connection], None]

# 索引位置即版本号（1-based）。v1 = 基础 SCHEMA（不在此列表）。
MIGRATIONS: list[Migration] = [
    _migrate_v2,  # v2: projects / versions / project_jobs + tasks 扩字段
]


def current_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def apply_all(conn: sqlite3.Connection) -> int:
    """把 user_version 推到 len(MIGRATIONS) + 1（基础 = 1）。返回最终版本号。"""
    target = len(MIGRATIONS) + 1
    cur = current_version(conn)
    if cur == 0:
        # 全新库 / 旧库未 set user_version：v1 已由 SCHEMA 建好
        cur = 1
        conn.execute("PRAGMA user_version = 1")
    while cur < target:
        migration = MIGRATIONS[cur - 1]  # cur=1 → MIGRATIONS[0] 推到 v2
        migration(conn)
        cur += 1
        conn.execute(f"PRAGMA user_version = {cur}")
    conn.commit()
    return cur
