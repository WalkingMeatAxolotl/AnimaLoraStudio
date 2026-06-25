"""Version eval sample run endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...schemas.projects import EvalSamplesStart
from ._shared import _publish_job_state, _version_dir_or_404
from .... import db
from ....infrastructure.paths import task_eval_dir
from ....services import eval_samples
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


@router.post("/api/projects/{pid}/versions/{vid}/eval/samples")
def start_eval_sample_run_endpoint(
    pid: int, vid: int, body: EvalSamplesStart
) -> dict[str, Any]:
    p, v, vdir = _version_dir_or_404(pid, vid)
    try:
        with db.connection_for() as conn:
            job, run = eval_samples.start_job(
                conn,
                p,
                v,
                vdir,
                checkpoint_path=body.checkpoint_path,
                max_items=body.max_items,
            )
    except eval_samples.EvalSamplesError as exc:
        raise HTTPException(400, str(exc)) from exc
    except eval_samples.eval_manifest.EvalManifestError as exc:
        raise HTTPException(400, str(exc)) from exc
    _publish_job_state(job)
    return {"job": job, "run": run}


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
