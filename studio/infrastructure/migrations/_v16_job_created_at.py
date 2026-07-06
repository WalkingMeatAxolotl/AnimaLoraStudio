"""v15 → v16: project_jobs 加 created_at（0.17 P-G 数据作业详情页）。

project_jobs 从建表起只有 started_at / finished_at，入队时间从未记录——
数据作业详情页要显示「入队时间」（对齐 task detail 的 summary）。

NULLABLE，不 backfill：老作业入队时刻已不可考（started_at 只能证明"何时
开始跑"），UI 对 NULL 显示「—」。新作业由 create_job 写入。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "project_jobs", "created_at", "created_at REAL")
