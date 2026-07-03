"""v17 → v18: 冻结旧 project_jobs 表（R-3 写路径翻转配套）。

R-3 起数据作业统一写入 tasks（task_type = kind），supervisor 不再从
project_jobs 派发。升级瞬间残留的 pending / running 旧行如果不处理会永远
停在原状态（没人再调度它们）——一次性标 canceled 并注明原因。

旧表其余行保留只读遗留（Q-R2：不迁移不展示，避免 ID 重映射污染 eval run
引用与旧日志路径）。
"""
from __future__ import annotations

import sqlite3
import time


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE project_jobs SET status = 'canceled', finished_at = ?, "
        "error_msg = 'superseded by unified ledger (0.17 R-3)' "
        "WHERE status IN ('pending', 'running')",
        (time.time(),),
    )
