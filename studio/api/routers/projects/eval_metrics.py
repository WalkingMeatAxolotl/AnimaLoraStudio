"""评估结果读侧端点。

一次评估 = 一个 EvalSession = 一个作业（#465），所以**没有**「合并多个子作业日志」的
端点了 —— 日志直接走统一的 `/api/logs/{task_id}`。历史 Session 全部保留，`/eval/sessions`
列出来供切换，`/eval/metrics` 默认给最新一次。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ._shared import _version_dir_or_404
from ...deps import _supervisor
from .... import db
from ....infrastructure.paths import task_eval_dir
from ....services import eval_metrics, eval_samples, eval_session

router = APIRouter()


@router.get("/api/projects/{pid}/versions/{vid}/eval/sessions")
def list_eval_sessions_endpoint(
    pid: int, vid: int, task_id: int | None = None, limit: int = 50,
) -> dict[str, Any]:
    """列该 version（可选：某训练 task）的评估 Session，最新在前。

    历史全部保留（一次评估一条），前端用它做「看哪一次评估」的切换。
    """
    _version_dir_or_404(pid, vid)
    with db.connection_for() as conn:
        sessions = [
            eval_session.reconcile_with_task(conn, s)
            for s in eval_session.list_sessions(
                conn, project_id=pid, version_id=vid,
                parent_task_id=task_id, limit=max(1, min(limit, 200)),
            )
        ]
    # plan 整体可能很大（200 个候选 + 验证集清单），列表里只给摘要
    for s in sessions:
        plan = s.pop("plan", None) or {}
        s.pop("plan_json", None)
        s["candidate_count"] = len(plan.get("candidates") or []) + (
            1 if (plan.get("baseline") or {}).get("enabled") else 0
        )
        s["metric_keys"] = (plan.get("metrics") or {}).get("keys") or []
        s["validation_images"] = (plan.get("reference_manifest") or {}).get("count", 0)
    return {"sessions": sessions}


@router.get("/api/projects/{pid}/versions/{vid}/eval/metrics")
def list_eval_metric_results_endpoint(
    pid: int,
    vid: int,
    task_id: int | None = None,
    session_id: int | None = None,
) -> dict[str, Any]:
    """评估结果。

    优先读 EvalSession（#465 起的数据模型）：`session_id` 指定哪一次，省略则取该
    task / version 最新一次。没有任何 Session 时回落到**存量**文件结果（0.21 及以前
    的 run.json / metrics.json 仍可读），这样老项目的历史指标不会因为换模型而消失。
    """
    _, _, vdir = _version_dir_or_404(pid, vid)
    with db.connection_for() as conn:
        session = None
        if session_id:
            session = eval_session.get_session(conn, session_id)
            if session is None:
                raise HTTPException(404, f"eval session 不存在: {session_id}")
            if int(session.get("version_id") or 0) != vid:
                raise HTTPException(400, "eval session 不属于该 version")
        else:
            latest = eval_session.list_sessions(
                conn, project_id=pid, version_id=vid,
                parent_task_id=task_id, limit=1,
            )
            session = latest[0] if latest else None
        if session is not None:
            # worker 被 kill 时 Session 会停在 running（最后一次写没发生）；拿 task
            # 的终态兜底，否则面板永远转圈、按钮也点不动
            session = eval_session.reconcile_with_task(conn, session)
            sid = int(session["id"])
            results = eval_session.session_results(conn, sid) or []
            return {
                "metric_specs": eval_metrics.metric_specs(),
                "cache": eval_metrics.cache_layout(
                    vdir, eval_session.samples_root(sid)
                ),
                "results": results,
                "session": {
                    k: v for k, v in session.items() if k not in ("plan", "plan_json")
                },
            }

    # 存量回落：没有 Session（老项目 / 从没跑过新版评估）
    eval_root = task_eval_dir(task_id) if task_id else None
    try:
        results = eval_metrics.list_results(vdir, eval_root)
    except (eval_metrics.EvalMetricsError, eval_samples.EvalSamplesError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "metric_specs": eval_metrics.metric_specs(),
        "cache": eval_metrics.cache_layout(vdir, eval_root),
        "results": results,
        "session": None,
        "legacy": True,
    }


@router.get("/api/projects/{pid}/versions/{vid}/eval/sessions/{sid}/grid")
def eval_session_grid_endpoint(pid: int, vid: int, sid: int) -> dict[str, Any]:
    """出图的 checkpoint × prompt 矩阵（给前端复用测试页的 XY 网格）。

    评估本来就为每个候选 × 每张验证图出了图，这里把它们排成矩阵让用户能顺手肉眼比 ——
    省掉去测试页重跑一次 XY。baseline 在第一列（纯底模对照，测试页做不到）。
    """
    _, _, vdir = _version_dir_or_404(pid, vid)
    with db.connection_for() as conn:
        session = eval_session.get_session(conn, sid)
        if session is None or int(session.get("version_id") or 0) != vid:
            raise HTTPException(404, f"eval session 不存在: {sid}")
        grid = eval_session.sample_grid(conn, sid, vdir)
    if grid is None:
        raise HTTPException(404, f"eval session 不存在: {sid}")
    return grid


@router.post("/api/projects/{pid}/versions/{vid}/eval/sessions/{sid}/cancel")
def cancel_eval_session_endpoint(pid: int, vid: int, sid: int) -> dict[str, Any]:
    """中断一次评估：取消 Session 的 task（已算出的结果保留）。"""
    _version_dir_or_404(pid, vid)
    with db.connection_for() as conn:
        session = eval_session.get_session(conn, sid)
        if session is None or int(session.get("version_id") or 0) != vid:
            raise HTTPException(404, f"eval session 不存在: {sid}")
        task_id = int(session.get("task_id") or 0)
    # Session 的 task 走统一队列取消（异步 SIGTERM）；worker 收到信号后把当前阶段标
    # canceled，已算完的候选结果留在库里。
    if task_id:
        _supervisor().cancel(task_id)
    return {"canceled": sid, "task_id": task_id or None}


@router.post("/api/projects/{pid}/versions/{vid}/eval/sessions/{sid}/retry")
def retry_eval_session_endpoint(pid: int, vid: int, sid: int) -> dict[str, Any]:
    """重跑一次失败 / 被中断的评估。

    复用 worker 的断点续跑：已出完图的候选跳过出图、已算完的指标跳过重算，所以补的
    只是没跑完的那部分。每次重试是队列里一条新的作业行（每次尝试都留档）。
    """
    _version_dir_or_404(pid, vid)
    with db.connection_for() as conn:
        session = eval_session.get_session(conn, sid)
        if session is None or int(session.get("version_id") or 0) != vid:
            raise HTTPException(404, f"eval session 不存在: {sid}")
        session = eval_session.reconcile_with_task(conn, session)
        try:
            session = eval_session.retry_session(conn, sid)
        except eval_session.EvalSessionError as exc:
            raise HTTPException(400, str(exc)) from exc
    return {"session": {k: v for k, v in session.items() if k not in ("plan", "plan_json")}}


@router.delete("/api/projects/{pid}/versions/{vid}/eval/sessions/{sid}")
def delete_eval_session_endpoint(pid: int, vid: int, sid: int) -> dict[str, Any]:
    """删一次评估的记录和产物。checkpoint 是引用，不受影响。"""
    _version_dir_or_404(pid, vid)
    with db.connection_for() as conn:
        session = eval_session.get_session(conn, sid)
        if session is None or int(session.get("version_id") or 0) != vid:
            raise HTTPException(404, f"eval session 不存在: {sid}")
        # 先兜底对齐 —— 否则 worker 被 kill 留下的僵尸 running 连删都删不掉
        session = eval_session.reconcile_with_task(conn, session)
        if session.get("status") in (
            eval_session.STATUS_PENDING, eval_session.STATUS_RUNNING
        ):
            raise HTTPException(400, "评估还在进行中，先中断再删除")
        eval_session.delete_session(conn, sid)
    return {"deleted": sid}


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
