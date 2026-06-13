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

回填（存量 terminal task）：Addendum 2 起 auto backup 改落 task 档案
`studio_data/tasks/<id>/state/`，但本 migration 之前跑的 task 写的是旧布局
`<version>/output/state/task_<id>/auto_epoch_state.pt`。对挂在项目上的
failed/canceled task 按 version 目录约定探测旧位置，探到就回填 last_state_*
—— resume 链路只认 DB 里记录的实际路径，新旧布局同样工作。

探测不到的维持 NULL（不可一键 resume，预期内）：Addendum 1 之前的 task
根本没有 auto backup；version 已删的恢复点已随目录消失；游离 task（无
project/version，直接按全局 preset 入队）的 output_dir 只在 yaml 里、可能
已被用户改过，按写时值不可靠，放弃回填。epoch/step 列不回填（要 torch.load
才拿得到，只是 UI 提示字段，不值得）。

DDL 设计原则：全部 NULLABLE，不需要 backfill 即可工作。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def _backfill_from_legacy_layout(conn: sqlite3.Connection) -> None:
    """探测旧布局回填 last_state_*。幂等：只动 last_state_path IS NULL 的行。"""
    # Lazy import：避免 migration 模块加载时强依赖 services 层；测试 monkeypatch
    # services.projects.PROJECTS_DIR 后跑 migrate 会拿到正确路径（同 _v11 pattern）。
    from ...services.projects import projects as _projects

    rows = conn.execute(
        "SELECT t.id, p.id, p.slug, v.label "
        "FROM tasks t "
        "JOIN projects p ON t.project_id = p.id "
        "JOIN versions v ON t.version_id = v.id "
        "WHERE t.status IN ('failed', 'canceled') "
        "AND t.last_state_path IS NULL"
    ).fetchall()
    for tid, pid, slug, label in rows:
        try:
            vdir = _projects.project_dir(int(pid), str(slug)) / "versions" / str(label)
        except (TypeError, ValueError):
            continue
        state_dir = vdir / "output" / "state" / f"task_{int(tid)}"
        pt = state_dir / "auto_epoch_state.pt"
        if not pt.is_file():
            continue
        cfg = state_dir / "auto_epoch_state.config.json"
        conn.execute(
            "UPDATE tasks SET last_state_path = ?, last_config_path = ? "
            "WHERE id = ?",
            (str(pt), str(cfg) if cfg.is_file() else None, int(tid)),
        )
    conn.commit()


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "last_state_path",
                           "last_state_path TEXT")
    _add_column_if_missing(conn, "tasks", "last_config_path",
                           "last_config_path TEXT")
    _add_column_if_missing(conn, "tasks", "last_state_epoch",
                           "last_state_epoch INTEGER")
    _add_column_if_missing(conn, "tasks", "last_state_step",
                           "last_state_step INTEGER")
    _backfill_from_legacy_layout(conn)
