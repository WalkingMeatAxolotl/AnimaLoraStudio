"""诊断包下载（docs/design/logging-target-state.md §3.6）。

1 route：
    GET /api/diagnostics/bundle?task_id=N   zip：env.json + task.json + task/run.log +
                                            config 快照 + monitor 快照 + 时间窗内的
                                            studio.log 片段；不带 task_id = env +
                                            studio.log 尾部。内容与脱敏规则见
                                            services/diagnostics.py。
入口：任务详情页「诊断包」、设置 → 系统 → 日志「导出诊断包」。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response

from ...domain.errors import NotFoundError
from ...services import diagnostics
from .installs import env_summary

router = APIRouter()


def _extra_env() -> dict[str, Any]:
    try:
        return {"env_summary": env_summary()}
    except Exception as e:  # noqa: BLE001
        return {"env_summary_error": str(e)}


@router.get("/api/diagnostics/bundle")
def diagnostics_bundle(task_id: int | None = Query(None, ge=1)) -> Response:
    try:
        data, name = diagnostics.build_bundle(task_id, extra_env=_extra_env())
    except LookupError:
        raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
