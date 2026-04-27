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
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import (
    browse,
    datasets,
    db,
    presets_io,
    project_jobs,
    projects,
    queue_io,
    secrets,
    versions,
)
from .event_bus import bus
from .services import downloader
from .paths import (
    LEGACY_MONITOR_HTML,
    LOGS_DIR,
    MONITOR_STATE_FILE,
    OUTPUT_DIR,
    REPO_ROOT,
    STUDIO_DB,
    USER_PRESETS_DIR,
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
# /api/schema, /api/presets/*  ( + 旧 /api/configs/* 308 redirect)
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


@app.get("/api/presets")
def list_presets_endpoint() -> dict[str, Any]:
    return {"items": presets_io.list_presets()}


@app.get("/api/presets/{name}")
def get_preset(name: str) -> dict[str, Any]:
    try:
        return presets_io.read_preset(name)
    except presets_io.PresetError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc


@app.put("/api/presets/{name}")
def put_preset(name: str, body: dict[str, Any]) -> dict[str, str]:
    try:
        path = presets_io.write_preset(name, body)
    except presets_io.PresetError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc
    return {"name": name, "path": str(path)}


@app.delete("/api/presets/{name}")
def delete_preset_endpoint(name: str) -> dict[str, str]:
    try:
        presets_io.delete_preset(name)
    except presets_io.PresetError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc
    return {"deleted": name}


@app.post("/api/presets/{name}/duplicate")
def duplicate_preset_endpoint(name: str, body: DuplicateRequest) -> dict[str, str]:
    try:
        path = presets_io.duplicate_preset(name, body.new_name)
    except presets_io.PresetError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc
    return {"name": body.new_name, "path": str(path)}


def _err_code(exc: presets_io.PresetError) -> int:
    """PresetError → HTTP 状态码：'不存在' → 404，名字非法/已存在 → 400，其它 → 422。"""
    msg = str(exc)
    if "不存在" in msg:
        return 404
    if "非法预设名" in msg or "已存在" in msg:
        return 400
    return 422


# 旧 /api/configs/* 端点保留为 308 redirect（保护任何外部脚本）。
# 308 保持 method + body，所以 PUT/POST/DELETE 都能透明转发。
@app.api_route(
    "/api/configs",
    methods=["GET", "POST", "PUT", "DELETE"],
    include_in_schema=False,
)
def _configs_root_redirect(request: Request) -> RedirectResponse:
    qs = ("?" + request.url.query) if request.url.query else ""
    return RedirectResponse(url=f"/api/presets{qs}", status_code=308)


@app.api_route(
    "/api/configs/{rest:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
    include_in_schema=False,
)
def _configs_redirect(rest: str, request: Request) -> RedirectResponse:
    qs = ("?" + request.url.query) if request.url.query else ""
    return RedirectResponse(url=f"/api/presets/{rest}{qs}", status_code=308)


# ---------------------------------------------------------------------------
# /api/secrets  (PP0 全局凭证 / 服务配置)
# ---------------------------------------------------------------------------


@app.get("/api/secrets")
def get_secrets() -> dict[str, Any]:
    return secrets.to_masked_dict(secrets.load())


@app.put("/api/secrets")
def put_secrets(body: dict[str, Any]) -> dict[str, Any]:
    new = secrets.update(body)
    return secrets.to_masked_dict(new)


# ---------------------------------------------------------------------------
# /api/projects + /api/projects/{pid}/versions  (PP1)
# ---------------------------------------------------------------------------


class ProjectCreate(BaseModel):
    title: str
    slug: Optional[str] = None
    note: Optional[str] = None
    initial_version_label: Optional[str] = "v1"


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    note: Optional[str] = None
    stage: Optional[str] = None
    active_version_id: Optional[int] = None


class VersionCreate(BaseModel):
    label: str
    fork_from_version_id: Optional[int] = None
    note: Optional[str] = None


class VersionUpdate(BaseModel):
    note: Optional[str] = None
    stage: Optional[str] = None
    config_name: Optional[str] = None


def _project_payload(p: dict[str, Any]) -> dict[str, Any]:
    """对外详情 payload：项目本身 + versions[] 含 stats + download stats。"""
    out = dict(p)
    out.update(projects.stats_for_project(p))
    with db.connection_for() as conn:
        vs = versions.list_versions(conn, p["id"])
    out["versions"] = [
        {**v, "stats": versions.stats_for_version(p, v)} for v in vs
    ]
    return out


def _publish_project_state(p: dict[str, Any]) -> None:
    bus.publish({
        "type": "project_state_changed",
        "project_id": p["id"],
        "stage": p["stage"],
    })


def _publish_version_state(v: dict[str, Any]) -> None:
    bus.publish({
        "type": "version_state_changed",
        "project_id": v["project_id"],
        "version_id": v["id"],
        "stage": v["stage"],
    })


def _project_err_code(exc: Exception) -> int:
    msg = str(exc)
    if "不存在" in msg:
        return 404
    if "已存在" in msg or "非法" in msg or "不能为空" in msg:
        return 400
    return 422


@app.get("/api/projects")
def list_projects_endpoint() -> dict[str, Any]:
    with db.connection_for() as conn:
        rows = projects.list_projects(conn)
    return {"items": projects.projects_with_stats(rows)}


@app.post("/api/projects")
def create_project_endpoint(body: ProjectCreate) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            p = projects.create_project(
                conn, title=body.title, slug=body.slug, note=body.note
            )
        except projects.ProjectError as exc:
            raise HTTPException(_project_err_code(exc), str(exc)) from exc
        if body.initial_version_label:
            try:
                versions.create_version(
                    conn, project_id=p["id"], label=body.initial_version_label
                )
            except versions.VersionError as exc:
                # 项目已建好；版本失败给前端但保留项目
                raise HTTPException(_project_err_code(exc), str(exc)) from exc
        p = projects.get_project(conn, p["id"])
    assert p is not None
    _publish_project_state(p)
    return _project_payload(p)


@app.get("/api/projects/{pid}")
def get_project_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    return _project_payload(p)


@app.patch("/api/projects/{pid}")
def patch_project_endpoint(pid: int, body: ProjectUpdate) -> dict[str, Any]:
    fields = body.model_dump(exclude_unset=True)
    with db.connection_for() as conn:
        try:
            p = projects.update_project(conn, pid, **fields)
        except projects.ProjectError as exc:
            raise HTTPException(_project_err_code(exc), str(exc)) from exc
    _publish_project_state(p)
    return _project_payload(p)


@app.delete("/api/projects/{pid}")
def delete_project_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            projects.soft_delete_project(conn, pid)
        except projects.ProjectError as exc:
            raise HTTPException(_project_err_code(exc), str(exc)) from exc
    return {"deleted": pid}


@app.post("/api/projects/_trash/empty")
def empty_trash_endpoint() -> dict[str, Any]:
    return {"removed": projects.empty_trash()}


# Versions ------------------------------------------------------------------


@app.get("/api/projects/{pid}/versions")
def list_versions_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        vs = versions.list_versions(conn, pid)
        p = projects.get_project(conn, pid)
    assert p is not None
    return {
        "items": [
            {**v, "stats": versions.stats_for_version(p, v)} for v in vs
        ]
    }


@app.post("/api/projects/{pid}/versions")
def create_version_endpoint(pid: int, body: VersionCreate) -> dict[str, Any]:
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        try:
            v = versions.create_version(
                conn,
                project_id=pid,
                label=body.label,
                fork_from_version_id=body.fork_from_version_id,
                note=body.note,
            )
        except versions.VersionError as exc:
            raise HTTPException(_project_err_code(exc), str(exc)) from exc
    _publish_version_state(v)
    return v


@app.get("/api/projects/{pid}/versions/{vid}")
def get_version_endpoint(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        p = projects.get_project(conn, pid)
    if not v or v["project_id"] != pid:
        raise HTTPException(404, f"版本不存在: id={vid}")
    assert p is not None
    return {**v, "stats": versions.stats_for_version(p, v)}


@app.patch("/api/projects/{pid}/versions/{vid}")
def patch_version_endpoint(
    pid: int, vid: int, body: VersionUpdate
) -> dict[str, Any]:
    fields = body.model_dump(exclude_unset=True)
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        try:
            v = versions.update_version(conn, vid, **fields)
        except versions.VersionError as exc:
            raise HTTPException(_project_err_code(exc), str(exc)) from exc
    _publish_version_state(v)
    return v


@app.delete("/api/projects/{pid}/versions/{vid}")
def delete_version_endpoint(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        versions.delete_version(conn, vid)
    return {"deleted": vid}


@app.post("/api/projects/{pid}/versions/{vid}/activate")
def activate_version_endpoint(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        v = versions.activate_version(conn, vid)
        p = projects.get_project(conn, pid)
    assert p is not None
    _publish_project_state(p)
    return _project_payload(p)


# ---------------------------------------------------------------------------
# /api/projects/{pid}/download + /api/projects/{pid}/files + /api/jobs/*  (PP2)
# ---------------------------------------------------------------------------


class DownloadRequest(BaseModel):
    tag: str
    count: int = 20
    api_source: str = "gelbooru"


class EstimateRequest(BaseModel):
    tag: str
    api_source: str = "gelbooru"


def _publish_job_state(job: dict[str, Any]) -> None:
    bus.publish({
        "type": "job_state_changed",
        "job_id": job["id"],
        "project_id": job["project_id"],
        "version_id": job.get("version_id"),
        "kind": job["kind"],
        "status": job["status"],
    })


@app.post("/api/projects/{pid}/download/estimate")
def estimate_download(pid: int, body: EstimateRequest) -> dict[str, Any]:
    """先调 booru 的 count API 估算命中数，再让用户决定 count。

    返回 -1 表示未知（API 不支持精确计数）；前端按「下载全部」处理。
    """
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise HTTPException(400, f"不支持的 api_source: {body.api_source}")
    if not body.tag.strip():
        raise HTTPException(400, "tag 不能为空")
    if not secrets.has_credentials_for(body.api_source):
        raise HTTPException(
            400,
            f"未配置 {body.api_source} 凭据，请先到「设置」页填写",
        )
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
    sec = secrets.load()
    if body.api_source == "danbooru":
        opts = downloader.DownloadOptions(
            tag=body.tag.strip(),
            count=1,
            api_source="danbooru",
            username=sec.danbooru.username,
            api_key=sec.danbooru.api_key,
            exclude_tags=list(sec.download.exclude_tags),
        )
    else:
        opts = downloader.DownloadOptions(
            tag=body.tag.strip(),
            count=1,
            api_source="gelbooru",
            user_id=sec.gelbooru.user_id,
            api_key=sec.gelbooru.api_key,
            exclude_tags=list(sec.download.exclude_tags),
        )
    count = downloader.estimate(opts)
    return {
        "tag": body.tag.strip(),
        "api_source": body.api_source,
        "exclude_tags": list(sec.download.exclude_tags),
        "effective_query": opts.effective_tag_query(),
        "count": count,
    }


@app.post("/api/projects/{pid}/download")
def start_download(pid: int, body: DownloadRequest) -> dict[str, Any]:
    if not body.tag.strip():
        raise HTTPException(400, "tag 不能为空")
    if body.count < 1:
        raise HTTPException(400, "count 必须 >= 1")
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise HTTPException(400, f"不支持的 api_source: {body.api_source}")
    if not secrets.has_credentials_for(body.api_source):
        raise HTTPException(
            400,
            f"未配置 {body.api_source} 凭据，请先到「设置」页填写",
        )

    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            kind="download",
            params={
                "tag": body.tag.strip(),
                "count": body.count,
                "api_source": body.api_source,
            },
        )
        # 推进项目 stage → downloading
        p = projects.advance_stage(conn, pid, "downloading")
    _publish_job_state(job)
    _publish_project_state(p)
    return job


@app.get("/api/projects/{pid}/download/status")
def download_status(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        job = project_jobs.latest_for(conn, project_id=pid, kind="download")
    if not job:
        return {"job": None, "log_tail": ""}
    log_path = Path(job.get("log_path") or "")
    tail = ""
    if log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(text.splitlines()[-50:])
        except Exception:
            tail = ""
    return {"job": job, "log_tail": tail}


@app.get("/api/projects/{pid}/files")
def list_files(pid: int, bucket: str = "download") -> dict[str, Any]:
    if bucket != "download":
        raise HTTPException(
            400, f"PP2 仅支持 bucket=download（PP3 会加 train/reg/samples）"
        )
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    items: list[dict[str, Any]] = []
    if pdir.exists():
        for f in sorted(pdir.iterdir()):
            if f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS:
                items.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "has_meta": f.with_suffix(".booru.txt").exists(),
                })
    return {"items": items, "count": len(items)}


@app.get("/api/projects/{pid}/thumb")
def project_thumb(pid: int, bucket: str = "download", name: str = "") -> FileResponse:
    """直接 serve 原图（不缩；前端 CSS 缩）。PP3 加缩略缓存。"""
    if bucket != "download":
        raise HTTPException(400, "PP2 仅支持 bucket=download")
    if "/" in name or "\\" in name or ".." in name or not name:
        raise HTTPException(400, "invalid name")
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    f = projects.project_dir(p["id"], p["slug"]) / "download" / name
    if not f.exists() or f.suffix.lower() not in datasets.IMAGE_EXTS:
        raise HTTPException(404)
    return FileResponse(f)


# /api/jobs/* —————————————————————————————————————————————————————————


@app.get("/api/jobs/{jid}")
def get_job_endpoint(jid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, jid)
    if not job:
        raise HTTPException(404, f"job 不存在: id={jid}")
    return job


@app.get("/api/jobs/{jid}/log")
def get_job_log(jid: int, tail: int = 0) -> dict[str, Any]:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, jid)
    if not job:
        raise HTTPException(404, f"job 不存在: id={jid}")
    log_path = Path(job.get("log_path") or "")
    if not log_path.exists():
        return {"job_id": jid, "content": "", "size": 0}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if tail and tail > 0:
        text = "\n".join(text.splitlines()[-tail:])
    return {
        "job_id": jid,
        "content": text,
        "size": len(text.encode("utf-8")),
    }


@app.post("/api/jobs/{jid}/cancel")
def cancel_job_endpoint(jid: int) -> dict[str, Any]:
    sup = _supervisor()
    ok = sup.cancel_job(jid)
    if not ok:
        with db.connection_for() as conn:
            job = project_jobs.get_job(conn, jid)
        if not job:
            raise HTTPException(404, f"job 不存在: id={jid}")
        if job["status"] in project_jobs.TERMINAL_STATUSES:
            raise HTTPException(400, f"job 已 {job['status']}")
        raise HTTPException(409, "cancel rejected (state mismatch)")
    return {"job_id": jid, "canceled": True}


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


# 导入 / 导出必须放在 /api/queue/{task_id} 之前，否则 "export" / "import" 会
# 被当成 task_id 走整数解析。
class ImportRequest(BaseModel):
    payload: dict[str, Any]


@app.get("/api/queue/export")
def export_queue(ids: str = "") -> dict[str, Any]:
    """`?ids=1,2,3` 指定导出的任务，缺省导出全部。"""
    if ids.strip():
        try:
            id_list = [int(x) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "ids must be comma-separated integers")
    else:
        with db.connection_for() as conn:
            id_list = [t["id"] for t in db.list_tasks(conn)]
    return queue_io.export_tasks(id_list)


@app.post("/api/queue/import")
def import_queue(body: ImportRequest) -> dict[str, Any]:
    try:
        return queue_io.import_tasks(body.payload)
    except (ValueError, presets_io.PresetError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/queue")
def list_queue(status: Optional[str] = None) -> dict[str, Any]:
    if status and status not in db.VALID_STATUSES:
        raise HTTPException(400, f"unknown status: {status}")
    with db.connection_for() as conn:
        items = db.list_tasks(conn, status=status)
    return {"items": items}


@app.post("/api/queue")
def enqueue(body: EnqueueRequest) -> dict[str, Any]:
    cfg_path = USER_PRESETS_DIR / f"{body.config_name}.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, f"preset not found: {body.config_name}")
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


# ---------------------------------------------------------------------------
# /api/datasets  (P4)
# ---------------------------------------------------------------------------


@app.get("/api/datasets")
def get_datasets(path: str = "") -> dict[str, Any]:
    """扫描数据集目录。`?path=` 指定根目录；缺省 = repo_root/dataset。"""
    root = Path(path) if path else REPO_ROOT / "dataset"
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    return datasets.scan_dataset_root(root)


@app.get("/api/browse")
def browse_dir(path: str = "") -> dict[str, Any]:
    """目录浏览（给前端 path picker 用）。缺省 = REPO_ROOT。"""
    target = Path(path) if path else REPO_ROOT
    if not target.is_absolute():
        target = (REPO_ROOT / target).resolve()
    try:
        return browse.list_dir(target)
    except browse.BrowseError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/datasets/thumbnail")
def get_dataset_thumbnail(folder: str, name: str) -> FileResponse:
    """返回 dataset 缩略图（实际是原图，前端用 CSS 缩放）。"""
    if ".." in folder or ".." in name or "\\" in name or "/" in name:
        raise HTTPException(400, "invalid path component")
    p = (Path(folder) / name).resolve()
    # 保证落在 repo 内（防止任意磁盘读取）
    try:
        p.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise HTTPException(403, "thumbnail path outside repo")
    if not p.exists() or p.suffix.lower() not in datasets.IMAGE_EXTS:
        raise HTTPException(404)
    return FileResponse(p)


# ---------------------------------------------------------------------------


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
