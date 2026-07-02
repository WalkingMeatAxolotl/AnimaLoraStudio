"""Schema 迁移：按顺序应用 SQL 升级，由 PRAGMA user_version 跟踪进度。

`db.init_db()` 先 executescript 基础 SCHEMA（v1，定义 tasks 表），然后调用
`apply_all()` 把 user_version 推到最新。新增迁移就往 MIGRATIONS 末尾加一个
回调，user_version 自动 +1。

约定：
- 任何 ALTER TABLE 必须容忍「列已存在」（IF NOT EXISTS 不适用于 ADD COLUMN，
  所以用 try/except 兜一下）；这样老 DB 升上来不会因为重复 ADD 而失败。
- 通常不允许向后改写已有列；只能加列 / 加表 / 加索引。
- **v9 例外**：ADR-0007 PR-5 destructive 删 stage 列（recreate-table 模式），
  在 ADR-0007 §后果 显式记录。
"""
from __future__ import annotations

import sqlite3
from typing import Callable

from ._v2_projects import migrate as _migrate_v2
from ._v3_monitor_state import migrate as _migrate_v3
from ._v4_task_config_path import migrate as _migrate_v4
from ._v5_task_type import migrate as _migrate_v5
from ._v6_pause_resume import migrate as _migrate_v6
from ._v7_version_trigger_word import migrate as _migrate_v7
from ._v8_version_status_phase import migrate as _migrate_v8
from ._v9_drop_legacy_stage import migrate as _migrate_v9
from ._v10_request_trace import migrate as _migrate_v10
from ._v11_preprocessing_phase import migrate as _migrate_v11
from ._v12_project_archived import migrate as _migrate_v12
from ._v13_last_state import migrate as _migrate_v13
from ._v14_generate_meta import migrate as _migrate_v14

Migration = Callable[[sqlite3.Connection], None]

# 索引位置即版本号（1-based）。v1 = 基础 SCHEMA（不在此列表）。
MIGRATIONS: list[Migration] = [
    _migrate_v2,  # v2: projects / versions / project_jobs + tasks 扩字段
    _migrate_v3,  # v3: tasks.monitor_state_path（PP6.1 per-version monitor）
    _migrate_v4,  # v4: tasks.config_path（PP6.3 私有 config 路径）
    _migrate_v5,  # v5: tasks.task_type（PR-9 区分 train / reg_ai / generate）
    _migrate_v6,  # v6: tasks.paused_* 列 + queue_settings 表（ADR 0006 PR-2）
    _migrate_v7,  # v7: versions.trigger_word（触发词字段）
    _migrate_v8,  # v8: versions.status / phase / last_failure_reason（ADR-0007）
    _migrate_v9,  # v9: 删 projects.stage + versions.stage（ADR-0007 PR-5 destructive）
    _migrate_v10, # v10: tasks.request_trace_id（ADR-0009 PR-1 C6 trace_id 跨进程贯穿）
    _migrate_v11, # v11: versions.phase 加 preprocessing 值（ADR-0010 配套，回填 curating+train 非空 → preprocessing）
    _migrate_v12, # v12: projects.archived_at（项目归档软隐藏）
    _migrate_v13, # v13: tasks.last_state_* 列（ADR 0006 Addendum 2 terminal-resume）
    _migrate_v14, # v14: tasks.generate_params / generate_cover（0.17 P-I forward-write，前端暂不读）
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
