"""v5 → v6: ADR 0006 PR-2 — pause/resume backend 骨架。

加列：

- `paused_state_path`：暂停时 `.pt` 路径（pause_step_<N>.pt）
- `paused_config_path`：暂停时 config snapshot 路径（pause_step_<N>.config.json）
- `paused_step`：global_step 快照（picker / UI 提示用）
- `paused_at`：UNIX 秒，"在 step N 暂停于 …"显示用

加表 `queue_settings`：kv 单行存储，跨重启保留。当前只有一个 key
`queue.held` (true/false)，dispatcher 看它决定是否跳过本轮调度（ADR §3.2）。

DDL 设计原则：所有 paused_* 列 NULLABLE（task 没暂停过时全 NULL），不需要
backfill。queue_settings 表全新，老库升上来空表。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "paused_state_path",
                           "paused_state_path TEXT")
    _add_column_if_missing(conn, "tasks", "paused_config_path",
                           "paused_config_path TEXT")
    _add_column_if_missing(conn, "tasks", "paused_step",
                           "paused_step INTEGER")
    _add_column_if_missing(conn, "tasks", "paused_at",
                           "paused_at REAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS queue_settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
