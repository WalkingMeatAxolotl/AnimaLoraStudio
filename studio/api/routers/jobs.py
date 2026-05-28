"""project_jobs (download/tag/reg_build) 读取 + 取消（PR-6 commit 2 从 server.py 抽出）。

3 routes：
    GET  /api/jobs/{jid}         job DB 行
    GET  /api/jobs/{jid}/log     job 日志（可选 tail=N）
    POST /api/jobs/{jid}/cancel  取消 pending / running job（异步 SIGTERM）

注：`/api/projects/{pid}/versions/{vid}/jobs/latest`（hydrate 用）不在本 router 范围
—— 它在 /api/projects/ 路径树下，归 PR-6.5 projects router。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..deps import _supervisor
from ... import db, project_jobs

router = APIRouter()


@router.get("/api/jobs/{jid}")
def get_job_endpoint(jid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, jid)
    if not job:
        raise HTTPException(404, f"job 不存在: id={jid}")
    return job


@router.get("/api/jobs/{jid}/log")
def get_job_log(jid: int, tail: int = 0) -> dict[str, Any]:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, jid)
    if not job:
        raise HTTPException(404, f"job 不存在: id={jid}")
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
            raise HTTPException(404, f"job 不存在: id={jid}")
        if job["status"] in project_jobs.TERMINAL_STATUSES:
            raise HTTPException(400, f"job 已 {job['status']}")
        raise HTTPException(409, "cancel rejected (state mismatch)")
    return {"job_id": jid, "canceled": True}
