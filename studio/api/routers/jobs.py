"""project_jobs (download/tag/reg_build) 读取 + 取消（PR-6 commit 2 从 server.py 抽出）。

4 routes：
    GET  /api/jobs               全局只读列表（0.17 P-G 数据作业区；group=live|history）
    GET  /api/jobs/{jid}         job DB 行
    GET  /api/jobs/{jid}/log     job 日志（可选 tail=N）
    POST /api/jobs/{jid}/cancel  取消 pending / running job（异步 SIGTERM）

注：`/api/projects/{pid}/versions/{vid}/jobs/latest`（hydrate 用）不在本 router 范围
—— 它在 /api/projects/ 路径树下，归 PR-6.5 projects router。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..deps import _supervisor
from ... import db
from ...domain.errors import ConflictError, NotFoundError, ValidationError
from ...services.projects import jobs as project_jobs

router = APIRouter()


# 0.17 P-G — history 分页 page_size 上限（同 /api/queue）。
_MAX_PAGE_SIZE = 100


@router.get("/api/jobs")
def list_jobs_endpoint(
    group: str = "live",
    page: int = 1,
    page_size: int = 20,
    kind: str | None = None,
) -> dict[str, Any]:
    """0.17 P-G 数据作业只读区的数据源。

    - `group=live`：running + pending，不分页，`{items}`。
    - `group=history`：done/failed/canceled，分页 `{items, total, page, page_size}`。
    - `kind`：按 VALID_KINDS 单值过滤（下沉 SQL 保 total 准）。

    只读呈现——不参与 reorder / pause，真正合并进 tasks 留 0.18（设计 D4）。
    """
    if group not in ("live", "history"):
        raise ValidationError(
            f"Unsupported jobs group: {group}",
            code="job.group_invalid", details={"group": group}, http_status=400,
        )
    if kind is not None and kind not in project_jobs.VALID_KINDS:
        raise ValidationError(
            f"Unsupported job kind filter: {kind}",
            code="job.kind_filter_invalid", details={"kind": kind}, http_status=400,
        )

    if group == "live":
        with db.connection_for() as conn:
            items = project_jobs.list_jobs_page(
                conn, statuses=project_jobs.LIVE_STATUSES, kind=kind,
            )
        return {"items": items}

    page = max(1, page)
    page_size = min(_MAX_PAGE_SIZE, max(1, page_size))
    offset = (page - 1) * page_size
    with db.connection_for() as conn:
        total = project_jobs.count_jobs(
            conn, statuses=project_jobs.HISTORY_STATUSES, kind=kind,
        )
        items = project_jobs.list_jobs_page(
            conn, statuses=project_jobs.HISTORY_STATUSES, kind=kind,
            limit=page_size, offset=offset,
        )
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/api/jobs/{jid}")
def get_job_endpoint(jid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, jid)
    if not job:
        raise NotFoundError("Job not found", code="job.not_found", details={"job_id": jid})
    return job


@router.get("/api/jobs/{jid}/log")
def get_job_log(jid: int, tail: int = 0) -> dict[str, Any]:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, jid)
    if not job:
        raise NotFoundError("Job not found", code="job.not_found", details={"job_id": jid})
    log_path = Path(job.get("log_path") or "")
    if not log_path.exists():
        return {"job_id": jid, "content": "", "size": 0}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if tail and tail > 0:
        text = "\n".join(text.splitlines()[-tail:])
    return {
        "job_id": jid,
        "content": text,
        "size": len(text.encode("utf-8")),
    }


@router.post("/api/jobs/{jid}/cancel")
def cancel_job_endpoint(jid: int) -> dict[str, Any]:
    sup = _supervisor()
    ok = sup.cancel_job(jid)
    if not ok:
        with db.connection_for() as conn:
            job = project_jobs.get_job(conn, jid)
        if not job:
            raise NotFoundError("Job not found", code="job.not_found", details={"job_id": jid})
        if job["status"] in project_jobs.TERMINAL_STATUSES:
            raise ValidationError(
                "This job has already finished",
                code="job.already_finished", http_status=400,
            )
        raise ConflictError(
            "Cannot cancel this job in its current state", code="job.cancel_rejected",
        )
    return {"job_id": jid, "canceled": True}
