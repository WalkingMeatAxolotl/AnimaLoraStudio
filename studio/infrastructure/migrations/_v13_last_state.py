"""v12 → v13: ADR 0006 Addendum 2 — terminal-resume（failed/canceled 可恢复）。

加列（tasks）：

- `last_state_path`：最近一次 epoch 末 auto backup 的 `.pt` 路径
  （auto_epoch_state.pt，覆盖式单文件）
- `last_config_path`：配套 config snapshot 路径（auto_epoch_state.config.json）
- `last_state_epoch` / `last_state_step`：备份点的 epoch / global_step
  （UI "从 epoch N 继续" 提示用）

supervisor 收到 `auto_epoch_backup_written` 事件时写入（此前只存内存 slot
字段，进程 / 机器一死即丢）。落 DB 后 failed（崩溃 / 关机）/ canceled task
的恢复点路径跨重启可查，resume endpoint 据此放宽状态门。

DDL 设计原则：全部 NULLABLE（task 没跑完过一个 epoch 时全 NULL），不需要
backfill；本 Addendum 之前的存量 terminal task 不回填（train task 不写
tasks.output_dir，服务端无法可靠重建路径）。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "last_state_path",
                           "last_state_path TEXT")
    _add_column_if_missing(conn, "tasks", "last_config_path",
                           "last_config_path TEXT")
    _add_column_if_missing(conn, "tasks", "last_state_epoch",
                           "last_state_epoch INTEGER")
    _add_column_if_missing(conn, "tasks", "last_state_step",
                           "last_state_step INTEGER")
