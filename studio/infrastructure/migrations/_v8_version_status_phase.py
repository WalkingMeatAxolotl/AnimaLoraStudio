"""v7 → v8: versions 加 status / phase / last_failure_reason — ADR-0007 §11.3-B。

把 master `versions.stage` 一个 enum 字段拆成两个正交字段：

- `status` (5 enum): preparing / training / completed / failed / canceled
- `phase`  (5 enum, 仅 status=preparing 时有意义):
  curating / tagging / editing / regularizing / ready
- `last_failure_reason` TEXT: 训练失败原因（UI 派生用，可空）

本迁移只 add 字段并按映射表回填，**不删** `versions.stage` 列（删除走 v9）。
迁移期间 backend 由 PR-3 双写新旧字段，保持老 frontend 兼容。

映射表（ADR-0007 §11.3-B）：

  master versions.stage → 新 status / phase
  ────────────────────────────────────────────────────────────
  curating         → status=preparing, phase=curating
  tagging          → status=preparing, phase=tagging
  regularizing     → status=preparing, phase=regularizing
  ready            → status=preparing, phase=ready
  training         → 看 latest task:
                       done     → completed
                       failed   → failed
                       canceled → canceled
                       running / pending / paused → training
                       task 不存在 → preparing + phase=ready (脏数据 fallback)
  done             → status=completed, phase=ready
  未知 stage       → preparing + phase=ready (fallback)

phase 在 status != preparing 时无业务意义，但字段必填以保 schema 简单 —
统一落 ready 表示"已走完准备阶段"。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


# stage → (status, phase) 静态映射（training 例外，需要 task lookup）
_STATIC_STAGE_MAP: dict[str, tuple[str, str]] = {
    "curating":     ("preparing", "curating"),
    "tagging":      ("preparing", "tagging"),
    "regularizing": ("preparing", "regularizing"),
    "ready":        ("preparing", "ready"),
    "done":         ("completed", "ready"),
}

# 老 stage='training' 时，按 latest task.status 派生
_TASK_STATUS_MAP: dict[str, str] = {
    "done":     "completed",
    "failed":   "failed",
    "canceled": "canceled",
    "pending":  "training",
    "running":  "training",
    "paused":   "training",
}


def migrate(conn: sqlite3.Connection) -> None:
    _add_columns(conn)
    _backfill(conn)


def _add_columns(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "versions", "status",
        "status TEXT NOT NULL DEFAULT 'preparing'",
    )
    _add_column_if_missing(
        conn, "versions", "phase",
        "phase TEXT NOT NULL DEFAULT 'curating'",
    )
    _add_column_if_missing(
        conn, "versions", "last_failure_reason",
        "last_failure_reason TEXT",
    )


def _backfill(conn: sqlite3.Connection) -> None:
    """按 ADR §11.3-B 迁移表把现有 versions.stage 翻译成 status + phase。"""
    rows = conn.execute("SELECT id, stage FROM versions").fetchall()
    for vid, stage in rows:
        if stage in _STATIC_STAGE_MAP:
            status, phase = _STATIC_STAGE_MAP[stage]
        elif stage == "training":
            status, phase = _derive_from_latest_task(conn, vid)
        else:
            # 未知 stage（理论上不应出现）→ fallback
            status, phase = "preparing", "ready"
        conn.execute(
            "UPDATE versions SET status = ?, phase = ? WHERE id = ?",
            (status, phase, vid),
        )
    conn.commit()


def _derive_from_latest_task(
    conn: sqlite3.Connection, version_id: int
) -> tuple[str, str]:
    """training stage 时按 latest task 推 status；phase 落 'ready'（终态无意义）。"""
    row = conn.execute(
        "SELECT status FROM tasks "
        "WHERE version_id = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (version_id,),
    ).fetchone()
    if row is None:
        # 脏数据 fallback：stage=training 但无关联 task
        return ("preparing", "ready")
    task_status = str(row[0]) if row[0] else ""
    new_status = _TASK_STATUS_MAP.get(task_status, "preparing")
    return (new_status, "ready")
