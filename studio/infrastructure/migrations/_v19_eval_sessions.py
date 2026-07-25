"""v18 → v19: EvalSession 数据模型（issue #465 的根治）。

0.21 及以前，一次评估被拆成 `(checkpoint 数 + baseline) × (1 出图 + N 指标)` 个普通
作业，各自一个 tasks 行 + 一个 `studio_data/tasks/<id>/` 日志目录 —— 200 个 checkpoint
就是 603 个。运行状态还散在三处（tasks 行 / run.json / metrics.json），多个指标 worker
并发改写同一个 metrics.json。

这里把评估提升为一等领域对象：

    eval_sessions        一次完整评估，拥有不可变 EvalPlan 和整体生命周期
    eval_candidates      本次评估的一个被测对象（某个 checkpoint 或 baseline）
    eval_metric_results  某个候选在某个指标上的结果

一个 Session 对应**一个** `eval_session` 类型的 tasks 行；`1+M` 个执行阶段跑在这一个
worker 进程内部，阶段状态落这几张表 —— DB 成为运行状态的唯一真相，文件只存 artifacts
和导出（report.json 可从 DB 重新生成）。

旧的 eval_samples / eval_clip / eval_dino / eval_tag / eval_ccip 作业行保留只读：不迁移
旧结果（旧 metrics.json 仍可读）。**但升级瞬间残留的 pending / running 行必须收掉** ——
它们的 worker 模块（`studio.workers.eval_*_worker`）已随本次改造删除，supervisor 派发
过去只会立刻失败重试。同 _v18 冻结旧 project_jobs 的处理方式。

存量日志目录由启动期一次性清理处理（services/eval_cleanup.py 的代际判据）。
"""
from __future__ import annotations

import sqlite3
import time

# 旧模型的 eval 子作业 kind。这里写死而不 import db.LEGACY_EVAL_TASK_TYPES —— 迁移是
# 历史快照，值域应该锁在写下它的那一刻，不能跟着后续常量变动漂移。
_LEGACY_EVAL_KINDS = (
    "eval_samples", "eval_clip", "eval_dino", "eval_tag", "eval_ccip",
)


def migrate(conn: sqlite3.Connection) -> None:
    placeholders = ",".join("?" for _ in _LEGACY_EVAL_KINDS)
    conn.execute(
        f"UPDATE tasks SET status = 'canceled', finished_at = ?, "
        f"error_msg = 'superseded by EvalSession (#465); re-run evaluation to get results' "
        f"WHERE task_type IN ({placeholders}) AND status IN ('pending', 'running')",
        (time.time(), *_LEGACY_EVAL_KINDS),
    )
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS eval_sessions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id        INTEGER,
            parent_task_id INTEGER,
            project_id     INTEGER,
            version_id     INTEGER,
            trigger        TEXT NOT NULL DEFAULT 'manual',
            status         TEXT NOT NULL DEFAULT 'pending',
            stage          TEXT,
            plan_json      TEXT NOT NULL DEFAULT '{}',
            created_at     REAL NOT NULL DEFAULT 0,
            started_at     REAL,
            finished_at    REAL,
            error          TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_eval_sessions_task
            ON eval_sessions(task_id);
        CREATE INDEX IF NOT EXISTS idx_eval_sessions_parent
            ON eval_sessions(parent_task_id);
        CREATE INDEX IF NOT EXISTS idx_eval_sessions_version
            ON eval_sessions(version_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS eval_candidates (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       INTEGER NOT NULL,
            role             TEXT NOT NULL DEFAULT 'checkpoint',
            checkpoint_path  TEXT NOT NULL DEFAULT '',
            checkpoint_digest TEXT,
            epoch            INTEGER,
            step             INTEGER,
            ordinal          INTEGER NOT NULL DEFAULT 0,
            status           TEXT NOT NULL DEFAULT 'pending',
            samples_done     INTEGER NOT NULL DEFAULT 0,
            samples_total    INTEGER NOT NULL DEFAULT 0,
            -- 首版桥梁：Session worker 内部仍复用现有 eval_samples run（run.json +
            -- images/）来出图和算指标，这里记住该 candidate 对应的 run_id。等 Phase 2
            -- 提取出 GenerationEngine、出图直接落 samples/<candidate_id>/ 之后可以退役。
            run_id           TEXT,
            error            TEXT,
            FOREIGN KEY(session_id) REFERENCES eval_sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_eval_candidates_session
            ON eval_candidates(session_id, ordinal);

        CREATE TABLE IF NOT EXISTS eval_metric_results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER NOT NULL,
            metric_key   TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            value        REAL,
            model_ref    TEXT,
            sample_count INTEGER,
            reason       TEXT,
            details_json TEXT,
            FOREIGN KEY(candidate_id) REFERENCES eval_candidates(id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_metric_results_unique
            ON eval_metric_results(candidate_id, metric_key);
        """
    )
