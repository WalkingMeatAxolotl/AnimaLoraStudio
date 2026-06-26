"""Version eval metric result endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ...schemas.projects import EvalClipStart, EvalDinoStart
from ._shared import _publish_job_state, _version_dir_or_404
from .... import db, secrets
from ....infrastructure.paths import task_eval_dir
from ....services import eval_clip, eval_dino, eval_metrics, eval_samples
from ....services.projects import jobs as project_jobs

router = APIRouter()

# 训练后 / 手动评估的 job kind（inline 训练时评估无 job，进训练日志）。
_EVAL_JOB_KINDS = ("eval_samples", "eval_clip", "eval_dino")


@router.get("/api/projects/{pid}/versions/{vid}/eval/metrics")
def list_eval_metric_results_endpoint(
    pid: int,
    vid: int,
    task_id: int | None = None,
) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = task_eval_dir(task_id) if task_id else None
    try:
        results = eval_metrics.list_results(vdir, eval_root)
    except (eval_metrics.EvalMetricsError, eval_samples.EvalSamplesError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "metric_specs": eval_metrics.metric_specs(),
        "cache": eval_metrics.cache_layout(vdir, eval_root),
        "results": results,
    }


@router.get("/api/projects/{pid}/versions/{vid}/eval/jobs")
def list_task_eval_jobs_endpoint(
    pid: int, vid: int, task_id: int,
) -> dict[str, Any]:
    """列某 task 的训练后/手动评估 job（eval_samples/clip/dino），给前端按
    run_id 关联 checkpoint 行 + 取原始日志（job_log_appended / GET /api/jobs/{id}/log）。

    job_log_appended 事件不带 task_id，刷新会丢 live 关联，所以靠这个端点重新发现。
    inline 训练时评估无 job（进训练日志），不在此列。
    """
    with db.connection_for() as conn:
        rows = project_jobs.list_jobs(conn, project_id=pid, version_id=vid)
    out: list[dict[str, Any]] = []
    for j in rows:
        if j.get("kind") not in _EVAL_JOB_KINDS:
            continue
        params = j.get("params_decoded") or {}
        if int(params.get("task_id") or 0) != task_id:
            continue
        out.append({
            "id": j.get("id"),
            "kind": j.get("kind"),
            "status": j.get("status"),
            "run_id": params.get("run_id"),
            "checkpoint_path": params.get("checkpoint_path"),
        })
    return {"jobs": out}


@router.get("/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}/metrics")
def get_eval_metric_result_endpoint(
    pid: int,
    vid: int,
    run_id: str,
    task_id: int | None = None,
) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = task_eval_dir(task_id) if task_id else None
    try:
        result = eval_metrics.load_result(vdir, run_id, eval_root)
    except (eval_metrics.EvalMetricsError, eval_samples.EvalSamplesError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if result is None:
        raise HTTPException(404, f"eval sample run 不存在: {run_id}")
    return {"metric_specs": eval_metrics.metric_specs(), "result": result}


@router.post("/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}/metrics/clip")
def start_eval_clip_metrics_endpoint(
    pid: int, vid: int, run_id: str, body: EvalClipStart
) -> dict[str, Any]:
    p, v, vdir = _version_dir_or_404(pid, vid)
    cfg = secrets.load().eval_metrics
    try:
        with db.connection_for() as conn:
            job, result = eval_clip.start_job(
                conn,
                p,
                v,
                vdir,
                run_id,
                model_name=body.model_name or cfg.clip_model_name,
            )
    except (
        eval_clip.EvalClipError,
        eval_metrics.EvalMetricsError,
        eval_samples.EvalSamplesError,
    ) as exc:
        raise HTTPException(400, str(exc)) from exc
    _publish_job_state(job)
    return {"job": job, "result": result}


@router.post("/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}/metrics/dino")
def start_eval_dino_metrics_endpoint(
    pid: int, vid: int, run_id: str, body: EvalDinoStart
) -> dict[str, Any]:
    p, v, vdir = _version_dir_or_404(pid, vid)
    cfg = secrets.load().eval_metrics
    try:
        with db.connection_for() as conn:
            job, result = eval_dino.start_job(
                conn,
                p,
                v,
                vdir,
                run_id,
                model_name=body.model_name or cfg.dino_model_name,
            )
    except (
        eval_dino.EvalDinoError,
        eval_metrics.EvalMetricsError,
        eval_samples.EvalSamplesError,
    ) as exc:
        raise HTTPException(400, str(exc)) from exc
    _publish_job_state(job)
    return {"job": job, "result": result}
