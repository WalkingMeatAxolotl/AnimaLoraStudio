"""健康检查 / 系统状态 / 训练监控状态读取（PR-5 从 server.py 抽出）。

3 routes：
    GET /api/health         健康检查（含 app version）
    GET /api/system/stats   Topbar 系统资源（CPU / RAM / GPU / VRAM）冷启动用
    GET /api/state          per-task monitor_state.json，前端监控页冷启拉一次后走 SSE 增量
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import responses as _responses
from ... import db
from ...services import system_stats

router = APIRouter()


_TASK_STATE_SELECT = """
    SELECT
        t.monitor_state_path,
        t.id AS task_id,
        t.project_id,
        t.version_id,
        p.slug AS project_slug,
        v.label AS version_label
    FROM tasks t
    LEFT JOIN projects p ON p.id = t.project_id
    LEFT JOIN versions v ON v.id = t.version_id
"""


@router.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    return {"status": "ok", "version": request.app.version}


@router.get("/api/system/stats")
def get_system_stats() -> dict[str, Any]:
    """Topbar 系统资源小组件用 (CPU/RAM/GPU/VRAM)。前端按 2-3s 轮询。"""
    return system_stats.stats_to_json(system_stats.collect_stats())


@router.get("/api/state")
def get_state(task_id: Optional[int] = None, max_points: int = 0) -> JSONResponse:
    """读取训练监控 state.json（PP6.1 改造 — per-task）。

    `task_id` 给了 → 查 tasks.monitor_state_path 对应文件；没有 / 文件缺失 →
    返回 EMPTY_STATE，不报错。
    `task_id` 没给 → 优先 running 的 task；没 running 时回退到**最近一次**
    （done / failed / canceled）带 monitor_state_path 的 task，让监控页结束
    后还能看到上一次训练的曲线。都没有再返回 EMPTY_STATE。

    `max_points`（默认 0 = 不降采样）— PR #37 引入时默认 1000，PR (此处)
    改成默认 0：cold start 是一次性 HTTP，10k+ 步训练用户经常碰到，宁可
    payload 大一点也要给完整历史。想降采样的 caller 显式传 max_points=N
    （留着给未来 thumbnail / 预览之类的轻量场景）。

    旧的全局 `monitor_data/state.json` 路径已退役（PP6.1）。
    """
    target_path: Optional[Path] = None
    target_task: dict[str, Any] | None = None
    if task_id is not None:
        with db.connection_for() as conn:
            row = conn.execute(
                _TASK_STATE_SELECT + " WHERE t.id = ?",
                (task_id,),
            ).fetchone()
        if row and row["monitor_state_path"]:
            target_path = Path(row["monitor_state_path"])
            target_task = dict(row)
    else:
        # 没给 task_id：先找 running 的 task；没 running 回退到最近的完成任务
        with db.connection_for() as conn:
            row = conn.execute(
                _TASK_STATE_SELECT +
                " WHERE t.status = 'running' AND t.monitor_state_path IS NOT NULL "
                "ORDER BY t.started_at DESC LIMIT 1"
            ).fetchone()
            if not (row and row["monitor_state_path"]):
                row = conn.execute(
                    _TASK_STATE_SELECT +
                    " WHERE t.monitor_state_path IS NOT NULL "
                    "ORDER BY COALESCE(t.finished_at, t.started_at, t.created_at) DESC "
                    "LIMIT 1"
                ).fetchone()
        if row and row["monitor_state_path"]:
            target_path = Path(row["monitor_state_path"])
            target_task = dict(row)

    if target_path is None or not target_path.exists():
        return JSONResponse(_responses.EMPTY_STATE)
    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"failed to read state: {exc}")

    # 服务端下采样 losses / lr_history（samples cap 50 已经在 train_monitor 端做了）
    if max_points and max_points > 0:
        from runtime.train_monitor import _downsample_uniform
        if isinstance(data.get("losses"), list):
            data["losses"] = _downsample_uniform(data["losses"], max_points)
        if isinstance(data.get("lr_history"), list):
            data["lr_history"] = _downsample_uniform(data["lr_history"], max_points)
        if isinstance(data.get("optimizer_metrics_history"), list):
            data["optimizer_metrics_history"] = _downsample_uniform(data["optimizer_metrics_history"], max_points)

    _attach_eval_context(data, target_task)
    return JSONResponse(data)


def _attach_eval_context(data: Any, task: dict[str, Any] | None) -> None:
    if not isinstance(data, dict) or not task:
        return
    project_id = task.get("project_id")
    version_id = task.get("version_id")
    if project_id is None or version_id is None:
        return
    data.setdefault("task_id", task.get("task_id"))
    data.setdefault("project_id", project_id)
    data.setdefault("version_id", version_id)
    if task.get("project_slug") is not None:
        data.setdefault("project_slug", task.get("project_slug"))
    if task.get("version_label") is not None:
        data.setdefault("version_label", task.get("version_label"))
