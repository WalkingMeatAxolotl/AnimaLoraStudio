"""v14 → v15: tasks 加 scheduled_at（0.17 P-B 计划任务）。

计划任务 = 独立 `scheduled` 状态（不是 pending 的子集）：入队时带未来时间 →
status='scheduled' + scheduled_at；supervisor 每秒 tick 到点把它提升为 pending，
之后走既有调度（等槽 → running → terminal）。dispatcher 只看 pending，
scheduled 对派活天然不可见，无需改 next_pending / _next_pending_task_in。

- scheduled_at：计划开始时间（unix 秒）。提升为 pending 后保留作记录
  （「原计划 3:00，实际手动提前」可追溯）。

NULLABLE，不需要 backfill；非计划任务恒 NULL。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "scheduled_at", "scheduled_at REAL")
