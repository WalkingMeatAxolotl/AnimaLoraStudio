"""Queue 数据 import / export + task snapshot（PR-6 commit 6 从 server.py 抽出）。

3 routes：
    GET  /api/queue/export                       JSON 文件下载（默认全部 / ?ids= 指定）
    POST /api/queue/import                       从 payload 重建队列（preset 名查重）
    GET  /api/queue/{task_id}/snapshot/config    ADR-0007 §11.7 task 启动 freeze 的 config

注：export / import 路径必须放在 `/api/queue/{task_id}` 之前（FastAPI 按
定义顺序匹配 path —— "export" / "import" 字符串否则会被当 task_id 走整数解析
报错）。本文件 router 顺序按这个约束排，include 时跟随 router 自己顺序。
"""
from __future__ import annotations

import json as _json
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from ...schemas.queue import ImportRequest
from .... import db
from ....domain.errors import NotFoundError, ValidationError
from ....services.presets import io as presets_io
from ....services import queue_io, task_snapshot
from ....infrastructure.event_bus import bus

router = APIRouter()


@router.get("/api/queue/export")
def export_queue(ids: str = "") -> Response:
    """`?ids=1,2,3` 指定导出的任务，缺省导出全部。

    响应带 `Content-Disposition: attachment` —— 前端 <a download> 直链就能触发
    浏览器原生下载（和 train.zip / outputs.zip 一套范式）。导出/失败 publish
    queue_export_ready / _failed SSE，前端用来清 app-side spinner + 弹 toast。
    body 仍是合法 JSON，tests / 程序化调用方拿 resp.json() 不受影响。
    """
    if ids.strip():
        try:
            id_list = [int(x) for x in ids.split(",") if x.strip()]
        except ValueError as exc:
            raise ValidationError(
                "The selected task IDs are not valid",
                code="queue.export_ids_invalid", http_status=400,
            ) from exc
    else:
        with db.connection_for() as conn:
            id_list = [t["id"] for t in db.list_tasks(conn)]
    try:
        payload = queue_io.export_tasks(id_list)
    except Exception as exc:
        bus.publish({"type": "queue_export_failed", "error": str(exc)})
        raise

    body = _json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    filename = f"queue_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    bus.publish({"type": "queue_export_ready"})
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/api/queue/import")
def import_queue(body: ImportRequest) -> dict[str, Any]:
    try:
        return queue_io.import_tasks(body.payload)
    except (ValueError, presets_io.PresetError) as exc:
        raise ValidationError(
            f"Could not import the queue: {exc}",
            code="queue.import_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc


@router.get("/api/queue/{task_id}/snapshot/config")
def get_task_snapshot_config(task_id: int) -> dict[str, Any]:
    """ADR-0007 §11.7：返回 task 启动时冻结的 config。

    返回 ``{"yaml": str, "config": dict}``。task 不存在 / 无 snapshot → 404。
    UI [关联配置] tab 用此 + 触发 "套用此配置" 路由跳转到 ⑦ 训练 phase + prefill。
    """
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
    data = task_snapshot.read_snapshot_config(task_id)
    if data is None:
        raise NotFoundError(
            "No saved configuration for this task",
            code="task.snapshot_not_found", details={"task_id": task_id},
        )
    return data
