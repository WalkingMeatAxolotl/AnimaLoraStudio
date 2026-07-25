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
from ....services import eval_auto, eval_samples, eval_session
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


def _eval_root_for(task_id: int | None, session_id: int | None) -> Path | None:
    """出图产物的根目录。

    新模型（EvalSession）落 `eval/sessions/<sid>/samples/`；`task_id` 是 0.21 及以前
    的**存量**布局 `tasks/<训练 task id>/eval/`。两者都要能读 —— 老项目的历史结果不该
    因为换模型而看不到。
    """
    if session_id:
        return eval_session.samples_root(session_id)
    return task_eval_dir(task_id) if task_id else None


@router.get("/api/projects/{pid}/versions/{vid}/eval/samples")
def list_eval_sample_runs_endpoint(
    pid: int,
    vid: int,
    task_id: int | None = None,
    session_id: int | None = None,
) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = _eval_root_for(task_id, session_id)
    with db.connection_for() as conn:
        job = project_jobs.latest_for(
            conn, project_id=pid, version_id=vid, kind=eval_session.TASK_TYPE
        )
    try:
        runs = eval_samples.list_runs(vdir, eval_root)
    except eval_samples.EvalSamplesError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"runs": runs, "latest_job": job, "log_tail": _tail_log(job)}


@router.get("/api/projects/{pid}/versions/{vid}/eval/scale")
def eval_scale_endpoint(
    pid: int, vid: int, selected: int | None = None,
) -> dict[str, Any]:
    """评估规模预估 —— 出图数与阶段数（issue #465 的成本可见性）。

    `selected` = 手动选中的 checkpoint 数；省略则按 version 的 checkpoint 策略算
    （训练后自动评估会评几个）。作业数恒为 1。
    """
    project, version, vdir = _version_dir_or_404(pid, vid)
    return eval_auto.eval_scale(project, version, vdir, selected_count=selected)


@router.post("/api/projects/{pid}/versions/{vid}/eval/run")
def run_task_eval_endpoint(
    pid: int, vid: int, body: EvalRunRequest
) -> dict[str, Any]:
    """手动评估一个已完成 task 的指定 checkpoint 集 —— 建**一个** EvalSession。

    不看自动评估开关（用户明确点了按钮）。上一轮的 Session 不动：历史全部留档，
    评估页默认显示最新那次。
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
            session = eval_auto.queue_manual_task_eval(
                conn, task, list(body.checkpoints)
            )
        except (eval_samples.EvalSamplesError, eval_session.EvalSessionError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if session is None:
            raise HTTPException(400, "没有可评估的 checkpoint（路径无效或不在 output/ 下）")
        # 走 project_jobs 而非 db.get_task：`kind` / `log_path` 是 as_job 注入的兼容
        # 字段，_publish_job_state 要用。
        eval_task = project_jobs.get_job(conn, int(session["task_id"]))
    if eval_task:
        _publish_job_state(eval_task)
    return {"session": session}


@router.get("/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}")
def get_eval_sample_run_endpoint(
    pid: int,
    vid: int,
    run_id: str,
    task_id: int | None = None,
    session_id: int | None = None,
) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = _eval_root_for(task_id, session_id)
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
    session_id: int | None = None,
) -> Any:
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = _eval_root_for(task_id, session_id)
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
