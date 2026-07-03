"""数据作业 DAO —— R-3 起换底到 tasks 统一台账。

接口保持 pp2 以来的形状（create_job / get_job / list_jobs / mark_* / …），
调用方（workers / eval 服务 / API 路由 / supervisor）零改动；底下全部读写
tasks 表（`task_type` = job kind，`params` JSON 为 _v17 列）。

旧 `project_jobs` 表冻结为只读遗留：_v18 把残留 pending/running 标 canceled；
Q-R2 拍板不展示、不迁移旧行（ID 重映射会污染 eval run 引用与 log 路径）。

行形状兼容：返回 dict 在 tasks 行基础上注入
    kind     = task_type
    log_path = tasks/<id>/run.log（paths.task_log_path，worker stdout 落点）
params_decoded 由 db._row_to_dict 统一提供（同旧 DAO 约定）。
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from ...infrastructure import db
from ...infrastructure.paths import STUDIO_DATA, task_log_path

# 旧 project_jobs 独立日志目录（studio_data/jobs/<id>.log）。R-3 起新作业日志
# 走 tasks/<id>/run.log；此常量仅供遗留只读与老测试 fixture 引用。
JOB_LOGS_DIR = STUDIO_DATA / "jobs"

# kind 全集（权威在 db.JOB_TASK_TYPES，保序 tuple 供 SQL IN；frozenset 供校验）。
JOB_TASK_TYPES: tuple[str, ...] = db.JOB_TASK_TYPES
VALID_KINDS: frozenset[str] = frozenset(JOB_TASK_TYPES)
VALID_STATUSES: frozenset[str] = frozenset({
    "pending", "running", "done", "failed", "canceled"
})
TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "canceled"})
# 0.17 P-G 数据作业只读区：live / history 两段（有序 tuple 供 SQL IN + 分页）。
LIVE_STATUSES: tuple[str, ...] = ("running", "pending")
HISTORY_STATUSES: tuple[str, ...] = ("done", "failed", "canceled")


from studio.domain.errors import DomainError


class JobError(DomainError):
    """数据作业业务错误。

    PR-2 C3 加 DomainError base — handler 自动翻 dual-write envelope。
    """
    default_code = "job.error"


def log_path_for(job_id: int) -> Path:
    """作业日志路径。R-3 起 = tasks/<id>/run.log（与 GPU 任务同款布局）。"""
    return task_log_path(job_id)


def as_job(task: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """tasks 行 → 作业 dict（注入 kind / log_path 兼容字段），就地改并返回。"""
    if not task:
        return None
    task["kind"] = task.get("task_type") or "train"
    task["log_path"] = str(task_log_path(int(task["id"])))
    return task


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
    jid = db.create_task(
        conn,
        name=kind,
        config_name=kind,
        task_type=kind,
        params=params,
        project_id=project_id,
        version_id=version_id,
    )
    return as_job(db.get_task(conn, jid)) or {}


def get_job(conn: sqlite3.Connection, jid: int) -> Optional[dict[str, Any]]:
    task = db.get_task(conn, jid)
    if not task or (task.get("task_type") or "train") not in VALID_KINDS:
        return None
    return as_job(task)


def list_jobs(
    conn: sqlite3.Connection,
    *,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
    kind: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict[str, Any]]:
    if kind is not None:
        type_clause = "task_type = ?"
        params: list[Any] = [kind]
    else:
        type_clause = f"task_type IN ({','.join('?' for _ in JOB_TASK_TYPES)})"
        params = list(JOB_TASK_TYPES)
    sql = f"SELECT * FROM tasks WHERE {type_clause}"
    if project_id is not None:
        sql += " AND project_id = ?"
        params.append(project_id)
    if version_id is not None:
        sql += " AND version_id = ?"
        params.append(version_id)
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY id DESC"
    return [as_job(db._row_to_dict(r)) or {} for r in conn.execute(sql, params)]


def list_pending_fifo(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """supervisor 派发用：pending 数据作业按 `priority DESC, created_at ASC`
    （与 GPU 任务同一 FIFO 语义，D-R3 平级；dispatch_order 硬编码已删）。"""
    placeholders = ",".join("?" for _ in JOB_TASK_TYPES)
    rows = conn.execute(
        f"SELECT * FROM tasks WHERE status = 'pending' "
        f"AND task_type IN ({placeholders}) "
        "ORDER BY priority DESC, created_at ASC",
        JOB_TASK_TYPES,
    )
    return [as_job(db._row_to_dict(r)) or {} for r in rows]


def _escape_like(q: str) -> str:
    """LIKE 元字符转义（% _ \\），同 db._build_task_filter 约定。"""
    return q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _jobs_filter(
    statuses: tuple[str, ...], kind: Optional[str], q: Optional[str] = None,
) -> tuple[str, list[Any]]:
    placeholders = ",".join("?" for _ in statuses)
    where = f" WHERE status IN ({placeholders})"
    params: list[Any] = list(statuses)
    if kind:
        where += " AND task_type = ?"
        params.append(kind)
    else:
        where += f" AND task_type IN ({','.join('?' for _ in JOB_TASK_TYPES)})"
        params.extend(JOB_TASK_TYPES)
    if q:
        # jobs 自身没有 name —— 按所属项目的 title / slug 搜（下沉 SQL 保 total 准）。
        like = f"%{_escape_like(q)}%"
        where += (
            " AND project_id IN (SELECT id FROM projects "
            "WHERE title LIKE ? ESCAPE '\\' OR slug LIKE ? ESCAPE '\\')"
        )
        params.extend([like, like])
    return where, params


def count_jobs(
    conn: sqlite3.Connection,
    *,
    statuses: tuple[str, ...],
    kind: Optional[str] = None,
    q: Optional[str] = None,
) -> int:
    """0.17 P-G — 按状态组（+ 可选 kind / 项目名搜索）计数，供 history 分页 total。"""
    where, params = _jobs_filter(statuses, kind, q)
    row = conn.execute(f"SELECT COUNT(*) FROM tasks{where}", params).fetchone()
    return int(row[0])


def list_jobs_page(
    conn: sqlite3.Connection,
    *,
    statuses: tuple[str, ...],
    kind: Optional[str] = None,
    q: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """0.17 P-G — 按状态组取 job，`id DESC`；limit=None 不分页（live 组全量）。"""
    where, params = _jobs_filter(statuses, kind, q)
    sql = f"SELECT * FROM tasks{where} ORDER BY id DESC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params = [*params, int(limit), int(offset)]
    return [as_job(db._row_to_dict(r)) or {} for r in conn.execute(sql, params)]


def count_active(
    conn: sqlite3.Connection, *, version_id: int, kind: str
) -> int:
    """该 version 下指定 kind 的 pending/running 数（phase 门禁用）。"""
    row = conn.execute(
        "SELECT COUNT(*) FROM tasks "
        "WHERE version_id = ? AND task_type = ? AND status IN ('pending', 'running')",
        (version_id, kind),
    ).fetchone()
    return int(row[0])


def latest_for(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    kind: str,
    version_id: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """取该项目（+ 可选 version）下最近一条指定 kind 的作业。"""
    sql = "SELECT * FROM tasks WHERE project_id = ? AND task_type = ?"
    params: list[Any] = [project_id, kind]
    if version_id is not None:
        sql += " AND version_id = ?"
        params.append(version_id)
    sql += " ORDER BY id DESC LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    return as_job(db._row_to_dict(row))


def mark_running(
    conn: sqlite3.Connection, jid: int, *, pid: Optional[int] = None
) -> None:
    db.update_task(conn, jid, status="running", started_at=time.time(), pid=pid)


def mark_done(conn: sqlite3.Connection, jid: int) -> None:
    db.update_task(conn, jid, status="done", finished_at=time.time())


def mark_failed(conn: sqlite3.Connection, jid: int, error_msg: str) -> None:
    db.update_task(
        conn, jid, status="failed", finished_at=time.time(), error_msg=error_msg
    )


def mark_canceled(conn: sqlite3.Connection, jid: int) -> None:
    db.update_task(conn, jid, status="canceled", finished_at=time.time())


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
        db.update_task(
            conn, jid, status="pending",
            started_at=None, finished_at=None, pid=None, error_msg=None,
        )


def cleanup_orphan_running(conn: sqlite3.Connection) -> int:
    """启动清孤儿。作业已并入 tasks，由 supervisor 的 task 孤儿收割统一处理；
    这里只兜旧 project_jobs 遗留表（_v18 已冻结，此处幂等再保险）。"""
    cur = conn.execute(
        "UPDATE project_jobs SET status = 'failed', finished_at = ?, "
        "error_msg = 'supervisor restart; orphan job' "
        "WHERE status = 'running'",
        (time.time(),),
    )
    conn.commit()
    return cur.rowcount
