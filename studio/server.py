"""AnimaStudio 守护服务（FastAPI）。

P1 范围（本文件目前实现）：
    - GET  /                   返回兼容的 monitor_smooth.html（旧监控）
    - GET  /api/health         健康检查
    - GET  /api/state          读取 monitor_data/state.json
    - GET  /samples/{name}     代理采样图（output/samples/）
    - GET  /studio/...         React 应用（构建后挂载，可缺省）

后续阶段会扩展（参见 plan）：
    - P2: /api/schema, /api/configs/*
    - P3: /api/queue/*, /api/events (SSE), /api/logs/{id}
    - P4: /api/datasets

启动：
    python -m studio.server [--host 127.0.0.1] [--port 8765] [--reload]
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import configs_io, db
from .event_bus import bus
from .paths import (
    LEGACY_MONITOR_HTML,
    LOGS_DIR,
    MONITOR_STATE_FILE,
    OUTPUT_DIR,
    STUDIO_DB,
    USER_CONFIGS_DIR,
    WEB_DIST,
    ensure_dirs,
)
from .schema import GROUP_ORDER, TrainingConfig
from .supervisor import Supervisor

ensure_dirs()
db.init_db()


@asynccontextmanager
async def _lifespan(app_: FastAPI) -> AsyncIterator[None]:
    """启动绑定 event bus 到当前 loop 并起 supervisor；关闭时停 supervisor。"""
    bus.attach_loop(asyncio.get_running_loop())
    sup = Supervisor(on_event=bus.publish)
    sup.start()
    app_.state.supervisor = sup
    try:
        yield
    finally:
        sup.stop()


app = FastAPI(title="AnimaStudio", version="0.1.0", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

EMPTY_STATE: dict[str, Any] = {
    "losses": [],
    "lr_history": [],
    "epoch": 0,
    "step": 0,
    "total_steps": 0,
    "speed": 0.0,
    "samples": [],
    "start_time": None,
    "config": {},
}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": app.version}


@app.get("/api/state")
def get_state() -> JSONResponse:
    """读取训练侧写出的 monitor_data/state.json。
    训练未启动 / 文件缺失时返回空状态，不报错。"""
    if not MONITOR_STATE_FILE.exists():
        return JSONResponse(EMPTY_STATE)
    try:
        data = json.loads(MONITOR_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"failed to read state: {exc}")
    return JSONResponse(data)


# ---------------------------------------------------------------------------
# /api/schema, /api/configs/*
# ---------------------------------------------------------------------------


class DuplicateRequest(BaseModel):
    new_name: str


@app.get("/api/schema")
def get_schema() -> dict[str, Any]:
    """返回 TrainingConfig 的 JSON Schema + 分组顺序，前端据此渲染表单。"""
    return {
        "schema": TrainingConfig.model_json_schema(),
        "groups": [{"key": k, "label": label} for k, label in GROUP_ORDER],
    }


@app.get("/api/configs")
def list_configs_endpoint() -> dict[str, Any]:
    return {"items": configs_io.list_configs()}


@app.get("/api/configs/{name}")
def get_config(name: str) -> dict[str, Any]:
    try:
        return configs_io.read_config(name)
    except configs_io.ConfigError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc


@app.put("/api/configs/{name}")
def put_config(name: str, body: dict[str, Any]) -> dict[str, str]:
    try:
        path = configs_io.write_config(name, body)
    except configs_io.ConfigError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc
    return {"name": name, "path": str(path)}


@app.delete("/api/configs/{name}")
def delete_config_endpoint(name: str) -> dict[str, str]:
    try:
        configs_io.delete_config(name)
    except configs_io.ConfigError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc
    return {"deleted": name}


@app.post("/api/configs/{name}/duplicate")
def duplicate_config_endpoint(name: str, body: DuplicateRequest) -> dict[str, str]:
    try:
        path = configs_io.duplicate_config(name, body.new_name)
    except configs_io.ConfigError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc
    return {"name": body.new_name, "path": str(path)}


def _err_code(exc: configs_io.ConfigError) -> int:
    """ConfigError → HTTP 状态码：'不存在' → 404，名字非法 → 400，其它 → 422。"""
    msg = str(exc)
    if "不存在" in msg:
        return 404
    if "非法配置名" in msg or "已存在" in msg:
        return 400
    return 422


# ---------------------------------------------------------------------------
# /api/queue, /api/logs, /api/events  (P3)
# ---------------------------------------------------------------------------


class EnqueueRequest(BaseModel):
    config_name: str
    name: Optional[str] = None
    priority: int = 0


class ReorderRequest(BaseModel):
    ordered_ids: list[int]


def _supervisor() -> Supervisor:
    sup: Optional[Supervisor] = getattr(app.state, "supervisor", None)
    if sup is None:
        raise HTTPException(503, "supervisor not running")
    return sup


@app.get("/api/queue")
def list_queue(status: Optional[str] = None) -> dict[str, Any]:
    if status and status not in db.VALID_STATUSES:
        raise HTTPException(400, f"unknown status: {status}")
    with db.connection_for() as conn:
        items = db.list_tasks(conn, status=status)
    return {"items": items}


@app.post("/api/queue")
def enqueue(body: EnqueueRequest) -> dict[str, Any]:
    cfg_path = USER_CONFIGS_DIR / f"{body.config_name}.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, f"config not found: {body.config_name}")
    name = body.name or body.config_name
    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name=name, config_name=body.config_name, priority=body.priority
        )
        task = db.get_task(conn, task_id)
    bus.publish(
        {"type": "task_state_changed", "task_id": task_id, "status": "pending"}
    )
    return task or {"id": task_id}


@app.get("/api/queue/{task_id}")
def get_queue_item(task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    return task


@app.post("/api/queue/{task_id}/cancel")
def cancel_task(task_id: int) -> dict[str, Any]:
    if not _supervisor().cancel(task_id):
        # 可能任务已结束 / 不在 supervisor 控制
        with db.connection_for() as conn:
            task = db.get_task(conn, task_id)
        if not task:
            raise HTTPException(404)
        if task["status"] in db.TERMINAL_STATUSES:
            raise HTTPException(400, f"task already {task['status']}")
        raise HTTPException(409, "cancel rejected (state mismatch)")
    return {"task_id": task_id, "canceled": True}


@app.post("/api/queue/{task_id}/retry")
def retry_task(task_id: int) -> dict[str, Any]:
    """已结束任务重新入队：复制 config_name 创建新 task。"""
    with db.connection_for() as conn:
        original = db.get_task(conn, task_id)
        if not original:
            raise HTTPException(404)
        if original["status"] not in db.TERMINAL_STATUSES:
            raise HTTPException(400, "only terminal tasks can be retried")
        new_id = db.create_task(
            conn,
            name=original["name"],
            config_name=original["config_name"],
            priority=original["priority"],
        )
        new_task = db.get_task(conn, new_id)
    bus.publish(
        {"type": "task_state_changed", "task_id": new_id, "status": "pending"}
    )
    return new_task or {"id": new_id}


@app.delete("/api/queue/{task_id}")
def delete_queue_item(task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
        if not task:
            raise HTTPException(404)
        if task["status"] not in db.TERMINAL_STATUSES:
            raise HTTPException(400, "only terminal tasks can be deleted")
        db.delete_task(conn, task_id)
    return {"deleted": task_id}


@app.post("/api/queue/reorder")
def reorder_queue(body: ReorderRequest) -> dict[str, Any]:
    with db.connection_for() as conn:
        db.reorder(conn, body.ordered_ids)
    return {"reordered": len(body.ordered_ids)}


@app.get("/api/logs/{task_id}")
def get_log(task_id: int) -> dict[str, Any]:
    p = LOGS_DIR / f"{task_id}.log"
    if not p.exists():
        return {"task_id": task_id, "content": "", "size": 0}
    text = p.read_text(encoding="utf-8", errors="replace")
    return {"task_id": task_id, "content": text, "size": len(text.encode("utf-8"))}


@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    """SSE：广播任务状态变化事件给所有订阅者。"""
    queue = await bus.subscribe()

    async def gen() -> AsyncIterator[bytes]:
        try:
            yield b": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(evt)}\n\n".encode("utf-8")
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# /samples
# ---------------------------------------------------------------------------


@app.get("/samples/{filename}")
def get_sample(filename: str) -> FileResponse:
    # 简单防越权
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")
    path = OUTPUT_DIR / "samples" / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


# ---------------------------------------------------------------------------
# 静态资源
# ---------------------------------------------------------------------------

# React 应用：构建后通过 /studio 访问。开发期请用 `npm run dev` 起 5173。
if WEB_DIST.exists():
    app.mount("/studio", StaticFiles(directory=str(WEB_DIST), html=True), name="studio")


@app.get("/", response_model=None)
def root() -> FileResponse | JSONResponse:
    """根路径返回兼容的旧监控页（保持现有体验不变）。
    P3 之后切换为 React 应用，此路由会重定向到 /studio。"""
    if LEGACY_MONITOR_HTML.exists():
        return FileResponse(LEGACY_MONITOR_HTML)
    return JSONResponse(
        {
            "message": "AnimaStudio is running. Build the React app at studio/web/ "
            "(npm install && npm run build) to enable the new UI."
        }
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AnimaStudio daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--reload", action="store_true", help="dev mode (auto-reload on edit)"
    )
    args = parser.parse_args()

    print(f"[AnimaStudio] http://{args.host}:{args.port}")
    uvicorn.run(
        "studio.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
