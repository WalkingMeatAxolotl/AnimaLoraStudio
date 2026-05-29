"""v9 → v10: tasks 加 request_trace_id 列（ADR-0009 PR-1 C6 trace_id 跨进程贯穿）。

API endpoint 入 task 时写 contextvar trace_id（HTTP 请求那一刻的 ID）；
supervisor dispatcher 拉起 task 时读这个列 → env 注入 worker 子进程 →
worker bootstrap bind contextvar。

不是这样做的话：用户点"开始训练"那一刻的 trace_id 跟 dispatcher 后台
spawn 那一刻的 trace_id 是两个不同 ID，trace_id 链路在 spawn 那一步断开。
用户截图 toast 的 trace 跟 worker log 里的 trace 对不上 — PR-1 C5 引入
trace_id 的价值就废了。

A round2 §4.4 强调："必须并入 PR-LOG-3 否则 trace_id 链路是断的"。
"""
from __future__ import annotations

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE tasks ADD COLUMN request_trace_id TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        # 列已存在（migration 容错）。
        pass
