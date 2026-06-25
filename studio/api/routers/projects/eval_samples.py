"""Version eval sample run endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...schemas.projects import EvalRunRequest
from ._shared import _publish_job_state, _version_dir_or_404
from .... import db
from ....infrastructure.paths import task_eval_dir
from ....services import eval_auto, eval_samples
from ....services.projects import jobs as project_jobs

router = APIRouter()


def _tail_log(job: dict[str, Any] | None, *, lines: int = 80) -> str:
    if not job:
        return ""
    path = Path(job.get("log_path") or "")
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return "\n".join(text.splitlines()[-lines:])


@router.get("/api/projects/{pid}/versions/{vid}/eval/samples")
def list_eval_sample_runs_endpoint(
    pid: int,
    vid: int,
    task_id: int | None = None,
) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = task_eval_dir(task_id) if task_id else None
    with db.connection_for() as conn:
        job = project_jobs.latest_for(
            conn, project_id=pid, version_id=vid, kind=eval_samples.JOB_KIND
        )
    try:
        runs = eval_samples.list_runs(vdir, eval_root)
    except eval_samples.EvalSamplesError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"runs": runs, "latest_job": job, "log_tail": _tail_log(job)}


@router.post("/api/projects/{pid}/versions/{vid}/eval/run")
def run_task_eval_endpoint(
    pid: int, vid: int, body: EvalRunRequest
) -> dict[str, Any]:
    """Manually evaluate a finished task over an explicit checkpoint set.

    Results are written task-scoped (`tasks/<task_id>/eval/`) so they show up in
    that task's eval page; the auto-eval Settings gate does not apply.
    """
    _version_dir_or_404(pid, vid)
    if not body.checkpoints:
        raise HTTPException(400, "checkpoints 不能为空")
    with db.connection_for() as conn:
        task = db.get_task(conn, int(body.task_id))
        if not task:
            raise HTTPException(404, f"task {body.task_id} 不存在")
        if int(task.get("project_id") or 0) != pid or int(task.get("version_id") or 0) != vid:
            raise HTTPException(400, "task 不属于该 project/version")
        try:
            queued = eval_auto.queue_manual_task_eval(
                conn, task, list(body.checkpoints), max_items=body.max_items
            )
        except eval_samples.EvalSamplesError as exc:
            raise HTTPException(400, str(exc)) from exc
    for job, _run in queued:
        _publish_job_state(job)
    return {
        "queued": len(queued),
        "jobs": [job for job, _ in queued],
        "runs": [run for _, run in queued],
    }


@router.get("/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}")
def get_eval_sample_run_endpoint(
    pid: int,
    vid: int,
    run_id: str,
    task_id: int | None = None,
) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = task_eval_dir(task_id) if task_id else None
    try:
        run = eval_samples.load_run(vdir, run_id, eval_root)
    except eval_samples.EvalSamplesError as exc:
        raise HTTPException(400, str(exc)) from exc
    if run is None:
        raise HTTPException(404, f"eval sample run 不存在: {run_id}")
    return {"run": run}


@router.get("/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}/images/{filename}")
def get_eval_sample_image_endpoint(
    pid: int,
    vid: int,
    run_id: str,
    filename: str,
    task_id: int | None = None,
) -> Any:
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = task_eval_dir(task_id) if task_id else None
    try:
        path = eval_samples.sample_image_path(vdir, run_id, filename, eval_root)
    except eval_samples.EvalSamplesError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
