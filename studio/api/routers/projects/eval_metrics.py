"""Version eval metric result endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ...schemas.projects import EvalClipStart, EvalDinoStart
from ._shared import _publish_job_state, _version_dir_or_404
from ...deps import _supervisor
from .... import db, secrets
from ....infrastructure.paths import task_eval_dir
from ....services import eval_clip, eval_dino, eval_metrics, eval_samples
from ....services.projects import jobs as project_jobs

router = APIRouter()

# 训练后 / 手动评估的 job kind（inline 训练时评估无 job，进训练日志）。
_EVAL_JOB_KINDS = ("eval_samples", "eval_clip", "eval_dino", "eval_tag", "eval_ccip")


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


@router.delete("/api/projects/{pid}/versions/{vid}/eval/runs")
def clear_task_eval_endpoint(pid: int, vid: int, task_id: int) -> dict[str, Any]:
    """清空某 task 的全部评估结果（run + 出图 + 指标）并把该 task 的评估 job 全部
    转 canceled。

    用于「删掉现有评估、重新跑」：下次「运行评估」从干净状态开始。已完成的指标文件
    一并删除；**该 task 的所有评估 job**（含已完成的）都标 canceled —— 否则历史 job
    会一直留在 DB、堆进合并日志（评估日志抽屉），还会让状态被老 failed job 带歪。
    pending/running 的走 supervisor 取消（SIGTERM），已结束的直接改 DB 状态。
    """
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = task_eval_dir(task_id)
    try:
        sup = _supervisor()
    except Exception:
        sup = None  # 服务启动中 / 无 supervisor → 退化为直接改 DB 状态
    with db.connection_for() as conn:
        rows = project_jobs.list_jobs(conn, project_id=pid, version_id=vid)
    targets = [
        j for j in rows
        if j.get("kind") in _EVAL_JOB_KINDS
        and int((j.get("params_decoded") or {}).get("task_id") or 0) == task_id
    ]
    canceled = 0
    for j in targets:
        jid = int(j["id"])
        was_active = j.get("status") not in ("done", "failed", "canceled")
        # 活跃 job 优先走 supervisor（running→SIGTERM 置 cancel_pending，退出标 canceled
        # 而非 failed）。拿不到 supervisor 或已结束 job → 直接改 DB 状态（把历史 job 从
        # 合并日志里移除）。已是 canceled 的跳过。
        if not (sup is not None and was_active and sup.cancel_job(jid)):
            if j.get("status") != "canceled":
                with db.connection_for() as conn:
                    project_jobs.mark_canceled(conn, jid)
        if was_active:
            canceled += 1
    removed = eval_samples.delete_all_runs(vdir, eval_root)
    return {"removed_runs": removed, "canceled_jobs": canceled}


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
