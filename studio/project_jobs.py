"""project_jobs DAO（pp2 起）。

`project_jobs` 是非训练异步任务（download / tag / reg_build）的统一调度
台账。supervisor 像调度 `tasks` 一样调度它们；前端通过 SSE 看进度。

字段（参见 migrations/_v2_projects.py）：
    id, project_id, version_id?, kind, params(JSON), status,
    started_at, finished_at, pid, log_path, error_msg
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from .paths import STUDIO_DATA

JOB_LOGS_DIR = STUDIO_DATA / "jobs"

VALID_KINDS: frozenset[str] = frozenset({"download", "tag", "reg_build"})
VALID_STATUSES: frozenset[str] = frozenset({
    "pending", "running", "done", "failed", "canceled"
})
TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "canceled"})


class JobError(Exception):
    """project_jobs 业务错误。"""


def log_path_for(job_id: int) -> Path:
    JOB_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return JOB_LOGS_DIR / f"{job_id}.log"


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if not row:
        return None
    out = dict(row)
    if isinstance(out.get("params"), str):
        try:
            out["params_decoded"] = json.loads(out["params"])
        except Exception:
            out["params_decoded"] = None
    return out


def create_job(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    kind: str,
    params: dict[str, Any],
    version_id: Optional[int] = None,
) -> dict[str, Any]:
    if kind not in VALID_KINDS:
        raise JobError(f"非法 kind: {kind!r}")
    cur = conn.execute(
        "INSERT INTO project_jobs(project_id, version_id, kind, params, status) "
        "VALUES (?, ?, ?, ?, 'pending')",
        (project_id, version_id, kind, json.dumps(params)),
    )
    conn.commit()
    jid = int(cur.lastrowid)
    log = log_path_for(jid)
    conn.execute(
        "UPDATE project_jobs SET log_path = ? WHERE id = ?",
        (str(log), jid),
    )
    conn.commit()
    return _row_to_dict(_row(conn, jid)) or {}


def get_job(conn: sqlite3.Connection, jid: int) -> Optional[dict[str, Any]]:
    return _row_to_dict(_row(conn, jid))


def _row(conn: sqlite3.Connection, jid: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM project_jobs WHERE id = ?", (jid,)
    ).fetchone()


def list_jobs(
    conn: sqlite3.Connection,
    *,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
    kind: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM project_jobs WHERE 1=1"
    params: list[Any] = []
    if project_id is not None:
        sql += " AND project_id = ?"
        params.append(project_id)
    if version_id is not None:
        sql += " AND version_id = ?"
        params.append(version_id)
    if kind is not None:
        sql += " AND kind = ?"
        params.append(kind)
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY id DESC"
    return [_row_to_dict(r) or {} for r in conn.execute(sql, params)]


def latest_for(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    kind: str,
    version_id: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """取该项目（+ 可选 version）下最近一条指定 kind 的 job。"""
    sql = (
        "SELECT * FROM project_jobs WHERE project_id = ? AND kind = ?"
    )
    params: list[Any] = [project_id, kind]
    if version_id is not None:
        sql += " AND version_id = ?"
        params.append(version_id)
    sql += " ORDER BY id DESC LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    return _row_to_dict(row)


def next_pending(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM project_jobs WHERE status = 'pending' "
        "ORDER BY id ASC LIMIT 1"
    ).fetchone()
    return _row_to_dict(row)


def mark_running(
    conn: sqlite3.Connection, jid: int, *, pid: Optional[int] = None
) -> None:
    conn.execute(
        "UPDATE project_jobs SET status = 'running', started_at = ?, pid = ? "
        "WHERE id = ?",
        (time.time(), pid, jid),
    )
    conn.commit()


def mark_done(conn: sqlite3.Connection, jid: int) -> None:
    conn.execute(
        "UPDATE project_jobs SET status = 'done', finished_at = ? WHERE id = ?",
        (time.time(), jid),
    )
    conn.commit()


def mark_failed(conn: sqlite3.Connection, jid: int, error_msg: str) -> None:
    conn.execute(
        "UPDATE project_jobs SET status = 'failed', finished_at = ?, "
        "error_msg = ? WHERE id = ?",
        (time.time(), error_msg, jid),
    )
    conn.commit()


def mark_canceled(conn: sqlite3.Connection, jid: int) -> None:
    conn.execute(
        "UPDATE project_jobs SET status = 'canceled', finished_at = ? "
        "WHERE id = ?",
        (time.time(), jid),
    )
    conn.commit()


def update_status(
    conn: sqlite3.Connection,
    jid: int,
    status: str,
    *,
    error_msg: Optional[str] = None,
) -> None:
    if status not in VALID_STATUSES:
        raise JobError(f"非法 status: {status!r}")
    if status == "running":
        mark_running(conn, jid)
    elif status == "done":
        mark_done(conn, jid)
    elif status == "failed":
        mark_failed(conn, jid, error_msg or "unknown")
    elif status == "canceled":
        mark_canceled(conn, jid)
    elif status == "pending":
        conn.execute(
            "UPDATE project_jobs SET status = 'pending', "
            "started_at = NULL, finished_at = NULL, pid = NULL, error_msg = NULL "
            "WHERE id = ?",
            (jid,),
        )
        conn.commit()


def cleanup_orphan_running(conn: sqlite3.Connection) -> int:
    """启动时把残留的 running 状态 job 标 failed（孤儿子进程已死）。"""
    cur = conn.execute(
        "UPDATE project_jobs SET status = 'failed', finished_at = ?, "
        "error_msg = 'supervisor restart; orphan job' "
        "WHERE status = 'running'",
        (time.time(),),
    )
    conn.commit()
    return cur.rowcount
