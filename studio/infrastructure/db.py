"""任务队列的 SQLite 持久化。

只保存任务索引；config 仍以 YAML 文件为权威源（task.config_name 指向
studio_data/configs/{config_name}.yaml）。
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .paths import STUDIO_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    config_name  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    priority     INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    pid          INTEGER,
    exit_code    INTEGER,
    output_dir   TEXT,
    error_msg    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_queue
    ON tasks(status, priority DESC, created_at ASC);
"""

VALID_STATUSES = {"pending", "running", "done", "failed", "canceled", "paused"}
# `paused` 不进 terminal —— task 已暂停但可被 resume，不算结束态。
TERMINAL_STATUSES = {"done", "failed", "canceled"}
# 队列页分区用的两个有序状态组（0.17 P-A/P-E）：live = 进行中+等待，history = 已结束。
# 用 tuple 保序，供 SQL `status IN (...)` + 分页。
LIVE_STATUSES = ("running", "paused", "pending")
HISTORY_STATUSES = ("done", "failed", "canceled")


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    """打开连接；调用方负责关闭（建议用 `with connection_for(...)`）。"""
    db_path = path or STUDIO_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def connection_for(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(path: Optional[Path] = None) -> None:
    """建基础表 + 把 schema 升级到最新版本（PRAGMA user_version 跟踪）。"""
    from .migrations import apply_all

    with connection_for(path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
        apply_all(conn)


# ---------------------------------------------------------------------------
# DAO
# ---------------------------------------------------------------------------


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row else None


def create_task(
    conn: sqlite3.Connection,
    *,
    name: str,
    config_name: str,
    priority: int = 0,
) -> int:
    # ADR-0009 PR-1 C6: 入 task 时存当前 ContextVar trace_id（HTTP 请求那一刻
    # 由 TraceIdMiddleware 已 bind）。无则用 bg-{uuid} 标后台触发（CLI / 测试 /
    # supervisor 直接拉起）。supervisor dispatcher 后续读这个列 → env 注入
    # worker 子进程让 worker log 跟用户请求 trace_id 对得上。
    from .logging import get_trace_id, new_trace_id
    request_trace_id = get_trace_id() or f"bg-{new_trace_id()}"
    cur = conn.execute(
        "INSERT INTO tasks(name, config_name, status, priority, created_at, request_trace_id) "
        "VALUES (?, ?, 'pending', ?, ?, ?)",
        (name, config_name, priority, time.time(), request_trace_id),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_task(conn: sqlite3.Connection, task_id: int) -> Optional[dict[str, Any]]:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_dict(row)


def filter_out_task_types(
    items: list[dict[str, Any]], excluded: tuple[str, ...]
) -> list[dict[str, Any]]:
    """commit 15：从 task 列表里剔掉指定 task_type（默认 task_type='train' 兼容）。"""
    return [
        t for t in items
        if (t.get("task_type") or "train") not in excluded
    ]


def list_tasks(
    conn: sqlite3.Connection, status: Optional[str] = None
) -> list[dict[str, Any]]:
    if status:
        sql = (
            "SELECT * FROM tasks WHERE status = ? "
            "ORDER BY priority DESC, created_at ASC"
        )
        params: tuple = (status,)
    else:
        sql = "SELECT * FROM tasks ORDER BY priority DESC, created_at ASC"
        params = ()
    return [dict(r) for r in conn.execute(sql, params)]


def _escape_like(s: str) -> str:
    """转义 LIKE 元字符（\\ % _），配合 `ESCAPE '\\'` 让搜索按字面匹配。"""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_task_filter(
    *,
    statuses: tuple[str, ...],
    exclude_types: tuple[str, ...],
    q: Optional[str],
) -> tuple[str, list[Any]]:
    """构造 WHERE 子句 + 参数（不含 ORDER/LIMIT），供 count_tasks / list_tasks_page 共用。

    - statuses：`status IN (...)`（保序 tuple）
    - exclude_types：`COALESCE(task_type,'train') NOT IN (...)`——老行 NULL 兜底成
      'train' 不会被误排除（0.17 P-F 前默认排掉 generate/reg_ai，保证分页 total 准）
    - q：name / config_name 子串（LIKE + ESCAPE，元字符转义）
    """
    clauses: list[str] = []
    params: list[Any] = []
    if statuses:
        clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
        params.extend(statuses)
    if exclude_types:
        clauses.append(
            f"COALESCE(task_type, 'train') NOT IN ({','.join('?' for _ in exclude_types)})"
        )
        params.extend(exclude_types)
    if q:
        like = f"%{_escape_like(q)}%"
        clauses.append("(name LIKE ? ESCAPE '\\' OR config_name LIKE ? ESCAPE '\\')")
        params.extend([like, like])
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def count_tasks(
    conn: sqlite3.Connection,
    *,
    statuses: tuple[str, ...],
    exclude_types: tuple[str, ...] = (),
    q: Optional[str] = None,
) -> int:
    """匹配 statuses（+ 可选 exclude_types / q）的 task 总数，供分页 total。"""
    where, params = _build_task_filter(
        statuses=statuses, exclude_types=exclude_types, q=q
    )
    row = conn.execute(f"SELECT COUNT(*) AS n FROM tasks{where}", params).fetchone()
    return int(row["n"]) if row else 0


def list_tasks_page(
    conn: sqlite3.Connection,
    *,
    statuses: tuple[str, ...],
    exclude_types: tuple[str, ...] = (),
    q: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """按 statuses（+ 可选 exclude_types / q）取 task，`id DESC`（近的在前）。

    limit=None 不分页（live 组全量）；给 limit 则 `LIMIT ? OFFSET ?`（history 组）。
    """
    where, params = _build_task_filter(
        statuses=statuses, exclude_types=exclude_types, q=q
    )
    sql = f"SELECT * FROM tasks{where} ORDER BY id DESC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params = [*params, int(limit), int(offset)]
    return [dict(r) for r in conn.execute(sql, params)]


def next_pending(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM tasks WHERE status = 'pending' "
        "ORDER BY priority DESC, created_at ASC LIMIT 1"
    ).fetchone()
    return _row_to_dict(row)


def update_task(
    conn: sqlite3.Connection, task_id: int, **fields: Any
) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    params = list(fields.values()) + [task_id]
    conn.execute(f"UPDATE tasks SET {cols} WHERE id = ?", params)
    conn.commit()


def delete_task(conn: sqlite3.Connection, task_id: int) -> int:
    cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    return cur.rowcount


def reorder(
    conn: sqlite3.Connection, ordered_ids: list[int]
) -> None:
    """按给定 id 顺序重写 priority（首位最高）。仅影响 pending 任务。"""
    base = len(ordered_ids)
    for i, tid in enumerate(ordered_ids):
        conn.execute(
            "UPDATE tasks SET priority = ? WHERE id = ? AND status = 'pending'",
            (base - i, tid),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# queue_settings —— kv 表，跨重启保留。ADR 0006 PR-2 引入。
# ---------------------------------------------------------------------------

_QUEUE_HELD_KEY = "queue.held"


def get_queue_held(conn: sqlite3.Connection) -> bool:
    """队列挂起开关（ADR §3.2）。默认 False（未挂起）。"""
    row = conn.execute(
        "SELECT value FROM queue_settings WHERE key = ?", (_QUEUE_HELD_KEY,)
    ).fetchone()
    if row is None:
        return False
    return str(row[0]).lower() == "true"


def set_queue_held(conn: sqlite3.Connection, held: bool) -> None:
    """写挂起开关。值序列化成字面量 "true" / "false"。"""
    conn.execute(
        "INSERT INTO queue_settings(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (_QUEUE_HELD_KEY, "true" if held else "false"),
    )
    conn.commit()
