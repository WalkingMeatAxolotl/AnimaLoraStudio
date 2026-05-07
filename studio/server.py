"""AnimaStudio 守护服务（FastAPI）。

P1 范围（本文件目前实现）：
    - GET  /                   302 跳转到 /studio/（旧监控页搬到 /monitor_smooth.html）
    - GET  /api/health         健康检查
    - GET  /api/state          读取 task 的 per-task monitor state
    - GET  /samples/{name}     代理采样图（按 task_id 解析到 version 目录）
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
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import (
    browse,
    curation,
    datasets,
    db,
    presets_io,
    project_jobs,
    projects,
    queue_io,
    secrets,
    thumb_cache,
    versions,
)
from .event_bus import bus
from .services import (
    caption_snapshot,
    downloader,
    presets as preset_flow,
    model_downloader,
    onnxruntime_setup,
    reg_builder,
    tagedit,
    train_io,
    uploads as uploads_svc,
    version_config,
)
from .services.tagger import VALID_TAGGER_NAMES, get_tagger
from .paths import (
    GENERATE_CONFIGS_DIR,
    GENERATE_JOBS_DIR,
    LEGACY_MONITOR_HTML,
    LOGS_DIR,
    OUTPUT_DIR,
    REPO_ROOT,
    STUDIO_DB,
    USER_PRESETS_DIR,
    WEB_DIST,
    ensure_dirs,
)
from .schema import GROUP_ORDER, GenerateConfig, RegAiConfig, TrainingConfig
from .supervisor import Supervisor

ensure_dirs()
db.init_db()

logger = logging.getLogger(__name__)


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
    "total_epochs": 0,
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
def get_state(task_id: Optional[int] = None) -> JSONResponse:
    """读取训练监控 state.json（PP6.1 改造 — per-task）。

    `task_id` 给了 → 查 tasks.monitor_state_path 对应文件；没有 / 文件缺失 →
    返回 EMPTY_STATE，不报错。
    `task_id` 没给 → 优先 running 的 task；没 running 时回退到**最近一次**
    （done / failed / canceled）带 monitor_state_path 的 task，让监控页结束
    后还能看到上一次训练的曲线。都没有再返回 EMPTY_STATE。

    旧的全局 `monitor_data/state.json` 路径已退役（PP6.1）。
    """
    target_path: Optional[Path] = None
    if task_id is not None:
        with db.connection_for() as conn:
            row = conn.execute(
                "SELECT monitor_state_path FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if row and row["monitor_state_path"]:
            target_path = Path(row["monitor_state_path"])
    else:
        # 没给 task_id：先找 running 的 task；没 running 回退到最近的完成任务
        with db.connection_for() as conn:
            row = conn.execute(
                "SELECT monitor_state_path FROM tasks WHERE status = 'running' "
                "AND monitor_state_path IS NOT NULL "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if not (row and row["monitor_state_path"]):
                row = conn.execute(
                    "SELECT monitor_state_path FROM tasks "
                    "WHERE monitor_state_path IS NOT NULL "
                    "ORDER BY COALESCE(finished_at, started_at, created_at) DESC "
                    "LIMIT 1"
                ).fetchone()
        if row and row["monitor_state_path"]:
            target_path = Path(row["monitor_state_path"])

    if target_path is None or not target_path.exists():
        return JSONResponse(EMPTY_STATE)
    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
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
        "groups": [
            {"key": k, "label": label, "default_collapsed": dc}
            for k, label, dc in GROUP_ORDER
        ],
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
# /api/models — 下载训练所需主模型 / VAE / tokenizer（PP7 第一刀）
# ---------------------------------------------------------------------------


class ModelDownloadRequest(BaseModel):
    model_id: str           # "anima_main" | "anima_vae" | "qwen3" | "t5_tokenizer"
    variant: Optional[str] = None  # 仅 anima_main 用，其他忽略


@app.get("/api/models/catalog")
def get_models_catalog() -> dict[str, Any]:
    """前端设置页 Models 区块用：列已知模型 + 各自磁盘状态 + 当前下载状态。"""
    return model_downloader.build_catalog()


@app.post("/api/models/download")
def start_model_download(body: ModelDownloadRequest) -> dict[str, Any]:
    """启动后台下载，立即返回 status key；前端通过 SSE
    (`model_download_changed`) 或轮询 catalog 看进度。"""
    try:
        key = model_downloader.trigger(body.model_id, body.variant)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    snap = model_downloader.get_status_snapshot()
    return {"key": key, "status": snap.get(key, {}).get("status", "running")}


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


# Train export / import (PP7) -----------------------------------------------


@app.get("/api/projects/{pid}/versions/{vid}/train.zip")
def export_version_train_zip(
    pid: int, vid: int, background: BackgroundTasks
) -> FileResponse:
    """打包 version 的 train/ + manifest.json 为 zip 一次性下载。

    实现：写到临时文件再 FileResponse；响应发完后 BackgroundTasks 清理。
    与 outputs.zip 一致用 ZIP_STORED（PNG/jpg 已压缩，再压只是浪费 CPU）。
    """
    import tempfile

    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        p = projects.get_project(conn, pid)
        assert p is not None

        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            train_io.export_train(conn, vid, tmp_path)
        except train_io.TrainIOError as exc:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(400, str(exc)) from exc
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    background.add_task(lambda: tmp_path.unlink(missing_ok=True))
    archive_name = f"{p['slug']}-{v['label']}.train.zip"
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=archive_name,
        background=background,
    )


@app.post("/api/projects/import-train")
async def import_train_zip(file: UploadFile = File(...)) -> dict[str, Any]:
    """上传训练集 zip → 新建 project + v1（stage=tagging），返回新项目。"""
    import tempfile

    if not file.filename:
        raise HTTPException(400, "缺少上传文件")
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        # UploadFile 内部本就是 SpooledTemporaryFile，大文件会落临时盘
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.close()
        tmp_path = Path(tmp.name)
        with db.connection_for() as conn:
            try:
                result = train_io.import_train(conn, tmp_path)
            except train_io.TrainIOError as exc:
                raise HTTPException(400, str(exc)) from exc
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass

    p = result["project"]
    _publish_project_state(p)
    _publish_version_state(result["version"])
    return {
        "project": _project_payload(p),
        "version": result["version"],
        "stats": result["stats"],
    }


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


@app.post("/api/projects/{pid}/upload")
async def upload_local_files(
    pid: int, files: list[UploadFile] = File(...)
) -> dict[str, Any]:
    """本地上传：单图（jpg/png）或 zip 包（自动解压）→ project 的 download/。

    与 booru 下载共用同一份「全量备份」目录；上传不走 job 系统，端点同步处理
    并返回 added / skipped 列表。任一文件成功即把项目 stage 推到 downloading。
    """
    if not files:
        raise HTTPException(400, "没有上传文件")
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"

    # 全量读入内存交给 service 解析；FastAPI 的 UploadFile 内部本就是 SpooledTemporaryFile，
    # 大文件会落临时盘，所以这里 read() 不会立即吃光内存。
    pairs: list[tuple[str, io.BytesIO]] = []
    for f in files:
        data = await f.read()
        pairs.append((f.filename or "", io.BytesIO(data)))
    result = uploads_svc.accept_many(pairs, pdir)

    if result.added:
        with db.connection_for() as conn:
            updated = projects.advance_stage(conn, pid, "downloading")
        _publish_project_state(updated)
    return result.as_dict()


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


class DeleteFilesRequest(BaseModel):
    names: list[str]


@app.post("/api/projects/{pid}/files/delete")
def delete_project_files(
    pid: int, body: DeleteFilesRequest
) -> dict[str, Any]:
    """从项目 `download/` 删除指定文件（含同名 caption metadata）。

    metadata 命名约定：
    - booru 下载会写 `{stem}.booru.txt`
    - tag/caption 流程可能写 `{stem}.txt` 或 `{stem}.json`
    都一并清理；不存在的扩展静默跳过。
    """
    if not body.names:
        return {"deleted": [], "missing": []}
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    if not pdir.exists():
        return {"deleted": [], "missing": list(body.names)}

    META_EXTS = (".booru.txt", ".txt", ".json")
    deleted: list[str] = []
    missing: list[str] = []
    for name in body.names:
        if (
            not name
            or "/" in name
            or "\\" in name
            or ".." in name
        ):
            raise HTTPException(400, f"invalid name: {name!r}")
        f = pdir / name
        if not f.exists() or not f.is_file():
            missing.append(name)
            continue
        try:
            f.unlink()
        except OSError as exc:
            raise HTTPException(500, f"删除失败 {name}: {exc}") from exc
        # 清理同 stem 的 metadata（best-effort，失败仅日志）
        stem = f.stem
        for ext in META_EXTS:
            m = pdir / f"{stem}{ext}"
            if m.exists():
                try:
                    m.unlink()
                except OSError as exc:
                    logger.warning("删 metadata 失败 %s: %s", m, exc)
        deleted.append(name)
    return {"deleted": deleted, "missing": missing}


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


def _thumb_response(src: Path, size: int) -> FileResponse:
    """统一 thumb 响应：弱 etag（基于 src mtime+size）+ no-cache 强制重验。

    早先用 `Cache-Control: public, max-age=86400` 会让浏览器记住所有响应 24h，
    包括重启过渡期的失败响应；用户视角就是「重启后图片加载不了」。改用 etag +
    no-cache 后，浏览器每次发条件请求，命中走 304 几 ms，错过响应不再阻塞。
    """
    out = thumb_cache.get_or_make_thumb(src, size)
    try:
        mtime_ns = out.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    etag = f'W/"{mtime_ns}-{size}"'
    return FileResponse(
        out,
        headers={
            "Cache-Control": "no-cache, must-revalidate",
            "ETag": etag,
        },
    )


@app.get("/api/projects/{pid}/thumb")
def project_thumb(
    pid: int,
    bucket: str = "download",
    name: str = "",
    size: int = 256,
) -> FileResponse:
    """缩略图：默认 256px JPEG（缓存）；size=0 → 原图。

    缓存路径：`studio_data/thumb_cache/{sha1(src+mtime+size)}.jpg`。
    源文件 mtime 变化会自动 invalidate（hash 变）。

    Cache 策略见 `_thumb_response` —— 不让浏览器长缓存，避免重启过渡期失败响应
    把图片锁死 24h。
    """
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
        logger.info("thumb 404: pid=%s bucket=%s name=%s -> %s", pid, bucket, name, f)
        raise HTTPException(404)
    return _thumb_response(f, size)


# /api/jobs/* —————————————————————————————————————————————————————————


@app.get("/api/jobs/{jid}")
def get_job_endpoint(jid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, jid)
    if not job:
        raise HTTPException(404, f"job 不存在: id={jid}")
    return job


_HYDRATABLE_JOB_KINDS = {"download", "tag", "reg_build"}


@app.get("/api/projects/{pid}/versions/{vid}/jobs/latest")
def get_latest_version_job(pid: int, vid: int, kind: str) -> dict[str, Any]:
    """页面刷新 hydrate 用：返回该 version 下指定 kind 的最近一条 job + 全量日志。

    Tagging / Regularization 页之前只在本会话 startBuild 后才知道 jid，刷新一下
    就丢了；这里给个起点让前端 mount 时锁回 jid + 回放历史日志，SSE 继续接力。
    `job` 可能是 running / pending / 已完成；前端按 status 决定要不要继续等事件。
    """
    if kind not in _HYDRATABLE_JOB_KINDS:
        raise HTTPException(400, f"unknown kind: {kind}")
    with db.connection_for() as conn:
        job = project_jobs.latest_for(
            conn, project_id=pid, kind=kind, version_id=vid
        )
    if not job:
        return {"job": None, "log": ""}
    log_path = Path(job.get("log_path") or "")
    log = ""
    if log_path.exists():
        try:
            log = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            log = ""
    return {"job": job, "log": log}


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
# /api/projects/{pid}/versions/{vid}/curation  (PP3)
# ---------------------------------------------------------------------------


class CopyRequest(BaseModel):
    files: list[str]
    dest_folder: str


class RemoveRequest(BaseModel):
    folder: str
    files: list[str]


class FolderOp(BaseModel):
    op: str  # "create" | "rename" | "delete"
    name: str
    new_name: Optional[str] = None


def _curation_err_code(exc: curation.CurationError) -> int:
    msg = str(exc)
    if "不存在" in msg:
        return 404
    if "已存在" in msg or "非法" in msg:
        return 400
    return 422


def _maybe_advance_after_train_change(conn, pid: int, vid: int) -> None:
    """copy/remove 后视情况推进 stage：train 有图 → curating → tagging 提示位。"""
    if curation.has_train_images(conn, pid, vid):
        v = versions.get_version(conn, vid)
        if v and v["stage"] == "curating":
            updated = versions.advance_stage(conn, vid, "tagging")
            _publish_version_state(updated)
        p = projects.get_project(conn, pid)
        if p and p["stage"] in ("created", "downloading", "curating"):
            updated_p = projects.advance_stage(conn, pid, "tagging")
            _publish_project_state(updated_p)


@app.get("/api/projects/{pid}/versions/{vid}/curation")
def get_curation(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            return curation.curation_view(conn, pid, vid)
        except curation.CurationError as exc:
            raise HTTPException(_curation_err_code(exc), str(exc)) from exc


@app.post("/api/projects/{pid}/versions/{vid}/curation/copy")
def copy_to_train(
    pid: int, vid: int, body: CopyRequest
) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            result = curation.copy_to_train(
                conn, pid, vid, body.files, body.dest_folder
            )
        except curation.CurationError as exc:
            raise HTTPException(_curation_err_code(exc), str(exc)) from exc
        _maybe_advance_after_train_change(conn, pid, vid)
    return result


@app.post("/api/projects/{pid}/versions/{vid}/curation/remove")
def remove_from_train(
    pid: int, vid: int, body: RemoveRequest
) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            result = curation.remove_from_train(
                conn, pid, vid, body.folder, body.files
            )
        except curation.CurationError as exc:
            raise HTTPException(_curation_err_code(exc), str(exc)) from exc
    return result


@app.post("/api/projects/{pid}/versions/{vid}/curation/folder")
def folder_op(
    pid: int, vid: int, body: FolderOp
) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            if body.op == "create":
                p = curation.create_folder(conn, pid, vid, body.name)
                return {"path": str(p)}
            if body.op == "rename":
                if not body.new_name:
                    raise HTTPException(400, "rename 需要 new_name")
                p = curation.rename_folder(
                    conn, pid, vid, body.name, body.new_name
                )
                return {"path": str(p)}
            if body.op == "delete":
                curation.delete_folder(conn, pid, vid, body.name)
                return {"deleted": body.name}
            raise HTTPException(400, f"unknown op: {body.op}")
        except curation.CurationError as exc:
            raise HTTPException(_curation_err_code(exc), str(exc)) from exc


# ---------------------------------------------------------------------------
# /api/tagger/{name}/check + /api/projects/{pid}/versions/{vid}/tag
# /api/projects/{pid}/versions/{vid}/captions/*  (PP4)
# ---------------------------------------------------------------------------


class Wd14Overrides(BaseModel):
    """打标页对 wd14 设置的「本次任务覆盖」—— 仅在 worker 进程内生效，
    不写回 secrets.json。"""
    threshold_general: Optional[float] = None
    threshold_character: Optional[float] = None
    model_id: Optional[str] = None
    local_dir: Optional[str] = None
    blacklist_tags: Optional[list[str]] = None


class TagJobRequest(BaseModel):
    tagger: str = "wd14"
    output_format: str = "txt"                # "txt" | "json"
    wd14_overrides: Optional[Wd14Overrides] = None


class CaptionEdit(BaseModel):
    tags: list[str]


class CommitItem(BaseModel):
    folder: str
    name: str
    tags: list[str]


class CommitRequest(BaseModel):
    items: list[CommitItem]


class BatchOp(BaseModel):
    op: str                                   # add|remove|replace|dedupe|stats
    scope: dict[str, Any]                     # {kind, folder?, names?}
    tags: Optional[list[str]] = None          # add/remove
    old: Optional[str] = None                 # replace
    new: Optional[str] = None                 # replace
    position: Optional[str] = "back"          # add: front|back
    top: int = 50                             # stats


# WD14 runtime / GPU 装包 (PP8) ---------------------------------------------


@app.get("/api/wd14/runtime")
def wd14_runtime() -> dict[str, Any]:
    """返回 onnxruntime 当前装的是哪个包 + 可用 EP + nvidia-smi 检测结果。"""
    rt = onnxruntime_setup.current_runtime()
    return {**rt, "cuda_detect": onnxruntime_setup.detect_cuda()}


class WD14InstallRequest(BaseModel):
    target: str = "auto"  # "auto" | "gpu" | "cpu"


@app.post("/api/wd14/install")
def wd14_install(body: WD14InstallRequest) -> dict[str, Any]:
    """切换 onnxruntime 包：先 uninstall 两个互斥包，再装目标。

    同步 pip install，几分钟级；前端按钮要带 loading。
    onnxruntime 是 C extension，装完后**必须重启 Studio** 才能切换 EP（pip 卸装
    重装不能热替换已 import 的 .pyd/.so）。返回 `restart_required=True` 让前端
    显式提示。
    """
    if body.target not in ("auto", "gpu", "cpu"):
        raise HTTPException(400, "target must be auto|gpu|cpu")
    try:
        res = onnxruntime_setup.install_runtime(body.target)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    stdout = res.pop("stdout", "")
    tail = "\n".join(stdout.splitlines()[-30:])
    # 同时返回当前进程视角（providers 仍是旧的，UI 用来对比）
    rt = onnxruntime_setup.current_runtime()
    return {
        **res,
        **rt,
        "cuda_detect": onnxruntime_setup.detect_cuda(),
        "stdout_tail": tail,
    }


@app.get("/api/tagger/{name}/check")
def check_tagger(name: str) -> dict[str, Any]:
    if name not in VALID_TAGGER_NAMES:
        raise HTTPException(400, f"unknown tagger: {name}")
    try:
        t = get_tagger(name)
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "ok": False, "msg": str(exc)}
    ok, msg = t.is_available()
    return {
        "name": name,
        "ok": ok,
        "msg": msg,
        "requires_service": getattr(t, "requires_service", False),
    }


def _version_train_dir_or_404(pid: int, vid: int):
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        p = projects.get_project(conn, pid)
    assert p is not None
    return p, v, versions.version_dir(p["id"], p["slug"], v["label"]) / "train"


@app.post("/api/projects/{pid}/versions/{vid}/tag")
def start_tag(pid: int, vid: int, body: TagJobRequest) -> dict[str, Any]:
    if body.tagger not in VALID_TAGGER_NAMES:
        raise HTTPException(400, f"unknown tagger: {body.tagger}")
    if body.output_format not in {"txt", "json"}:
        raise HTTPException(400, "output_format must be txt|json")
    _, v, _ = _version_train_dir_or_404(pid, vid)

    params: dict[str, Any] = {
        "tagger": body.tagger,
        "version_id": vid,
        "output_format": body.output_format,
    }
    if body.tagger == "wd14" and body.wd14_overrides is not None:
        # 仅保留用户实际填写的字段；空 dict 也不写
        ov = body.wd14_overrides.model_dump(exclude_none=True)
        if ov:
            params["wd14_overrides"] = ov

    with db.connection_for() as conn:
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            version_id=vid,
            kind="tag",
            params=params,
        )
        # 推 stage：tagging
        if v["stage"] in ("curating",):
            updated = versions.advance_stage(conn, vid, "tagging")
            _publish_version_state(updated)
        p = projects.get_project(conn, pid)
        if p and p["stage"] in ("created", "downloading", "curating"):
            up = projects.advance_stage(conn, pid, "tagging")
            _publish_project_state(up)
    _publish_job_state(job)
    return job


@app.get("/api/projects/{pid}/versions/{vid}/captions")
def list_captions_endpoint(
    pid: int, vid: int, folder: Optional[str] = None, full: bool = False
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    if folder is None:
        return {"folder": None, "items": tagedit.list_all_captions(train, full=full)}
    if not folder or "/" in folder or "\\" in folder or ".." in folder:
        raise HTTPException(400, "invalid folder")
    return {
        "folder": folder,
        "items": tagedit.list_captions_in_folder(train, folder, full=full),
    }


@app.get("/api/projects/{pid}/versions/{vid}/captions/{folder}/{filename}")
def get_caption_endpoint(
    pid: int, vid: int, folder: str, filename: str
) -> dict[str, Any]:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")
    if "/" in folder or "\\" in folder or ".." in folder:
        raise HTTPException(400, "invalid folder")
    _, _, train = _version_train_dir_or_404(pid, vid)
    try:
        return tagedit.read_one(train, folder, filename)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.put("/api/projects/{pid}/versions/{vid}/captions/{folder}/{filename}")
def put_caption_endpoint(
    pid: int, vid: int, folder: str, filename: str, body: CaptionEdit
) -> dict[str, Any]:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")
    if "/" in folder or "\\" in folder or ".." in folder:
        raise HTTPException(400, "invalid folder")
    _, _, train = _version_train_dir_or_404(pid, vid)
    try:
        return tagedit.write_one(train, folder, filename, body.tags)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------------------
# Caption snapshots（PP4 拆分后新增）
# ---------------------------------------------------------------------------


def _version_dir_or_404(pid: int, vid: int):
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        p = projects.get_project(conn, pid)
    assert p is not None
    return p, v, versions.version_dir(p["id"], p["slug"], v["label"])


@app.post("/api/projects/{pid}/versions/{vid}/captions/snapshot")
def create_caption_snapshot(pid: int, vid: int) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    return caption_snapshot.create_snapshot(vdir)


@app.get("/api/projects/{pid}/versions/{vid}/captions/snapshots")
def list_caption_snapshots(pid: int, vid: int) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    return {"items": caption_snapshot.list_snapshots(vdir)}


@app.post("/api/projects/{pid}/versions/{vid}/captions/snapshots/{sid}/restore")
def restore_caption_snapshot(pid: int, vid: int, sid: str) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    try:
        return caption_snapshot.restore_snapshot(vdir, sid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except caption_snapshot.SnapshotError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/projects/{pid}/versions/{vid}/captions/snapshots/{sid}")
def delete_caption_snapshot(pid: int, vid: int, sid: str) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    try:
        caption_snapshot.delete_snapshot(vdir, sid)
        return {"deleted": sid}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except caption_snapshot.SnapshotError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/projects/{pid}/versions/{vid}/captions/commit")
def commit_captions(pid: int, vid: int, body: CommitRequest) -> dict[str, Any]:
    """一次性写入多个 caption；写之前自动生成快照作还原点。"""
    _, _, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    snap = caption_snapshot.create_snapshot(vdir)
    written = 0
    skipped: list[str] = []
    for it in body.items:
        if "/" in it.folder or "\\" in it.folder or ".." in it.folder:
            skipped.append(f"{it.folder}/{it.name}")
            continue
        if "/" in it.name or "\\" in it.name or ".." in it.name:
            skipped.append(f"{it.folder}/{it.name}")
            continue
        img = train / it.folder / it.name
        if not img.exists():
            skipped.append(f"{it.folder}/{it.name}")
            continue
        tagedit.write_tags(img, it.tags)
        written += 1
    return {"snapshot": snap, "written": written, "skipped": skipped}


@app.post("/api/projects/{pid}/versions/{vid}/captions/batch")
def batch_caption_endpoint(
    pid: int, vid: int, body: BatchOp
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    op = body.op
    scope = body.scope
    if op == "add":
        n = tagedit.add_tags(
            scope, train, body.tags or [],
            position="front" if body.position == "front" else "back",
        )
        return {"op": op, "affected": n}
    if op == "remove":
        return {"op": op, "affected": tagedit.remove_tags(scope, train, body.tags or [])}
    if op == "replace":
        if not body.old or not body.new:
            raise HTTPException(400, "replace 需要 old 和 new")
        return {"op": op, "affected": tagedit.replace_tag(scope, train, body.old, body.new)}
    if op == "dedupe":
        return {"op": op, "affected": tagedit.dedupe(scope, train)}
    if op == "stats":
        return {"op": op, "items": tagedit.stats(scope, train, top=max(1, body.top))}
    raise HTTPException(400, f"unknown op: {op}")


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/reg  (PP5)
# ---------------------------------------------------------------------------


class RegBuildRequest(BaseModel):
    # 目标数量永远 = train 总数（与源脚本一致），UI 不暴露
    excluded_tags: list[str] = []
    auto_tag: bool = True
    api_source: str = "gelbooru"
    incremental: bool = False  # PP5.1：补足 — 不清空已有图，只补缺口
    # PP5.5 进阶配置（默认值与源脚本一致）
    skip_similar: bool = True
    aspect_ratio_filter_enabled: bool = False
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    postprocess_method: str = "smart"  # smart | stretch | crop
    postprocess_max_crop_ratio: float = 0.1


def _reg_dir(vdir: Path) -> Path:
    """reg 根目录 — 子目录直接镜像 train 子文件夹（与源脚本一致，无 1_general 中间层）。"""
    return vdir / "reg"


@app.get("/api/projects/{pid}/versions/{vid}/reg/preview-tags")
def reg_preview_tags(pid: int, vid: int, top: int = 20) -> dict[str, Any]:
    """返回 train 的 tag 频率 top N（不真生成 reg）。给 UI「排除 tag」勾选用。"""
    _, _, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    items = reg_builder.preview_train_tag_distribution(train, top=max(1, top))
    return {"items": [{"tag": t, "count": c} for t, c in items]}


@app.get("/api/projects/{pid}/versions/{vid}/reg")
def get_reg_status(pid: int, vid: int) -> dict[str, Any]:
    """返回 reg 集状态（meta + 图片数 + 文件名列表）。"""
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    if not rdir.exists():
        return {"exists": False, "meta": None, "image_count": 0, "files": []}
    images: list[str] = []
    for f in sorted(rdir.rglob("*")):
        if f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS:
            try:
                rel = f.relative_to(rdir).as_posix()
            except ValueError:
                continue
            images.append(rel)
    meta = reg_builder.read_meta(rdir)
    meta_dict = None
    if meta is not None:
        from dataclasses import asdict as _asdict
        meta_dict = _asdict(meta)
    return {
        "exists": bool(images) or meta is not None,
        "meta": meta_dict,
        "image_count": len(images),
        "files": images,
    }


@app.post("/api/projects/{pid}/versions/{vid}/reg/build")
def start_reg_build(pid: int, vid: int, body: RegBuildRequest) -> dict[str, Any]:
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise HTTPException(400, "api_source must be gelbooru|danbooru")
    if body.postprocess_method not in {"smart", "stretch", "crop"}:
        raise HTTPException(400, "postprocess_method must be smart|stretch|crop")
    if not (0.05 <= body.postprocess_max_crop_ratio <= 0.5):
        raise HTTPException(400, "postprocess_max_crop_ratio must be 0.05–0.5")
    if body.aspect_ratio_filter_enabled and not (
        0.0 < body.min_aspect_ratio < body.max_aspect_ratio
    ):
        raise HTTPException(400, "min_aspect_ratio must be < max_aspect_ratio (both > 0)")
    _, v, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    has_image = train.exists() and any(
        f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
        for f in train.rglob("*")
    )
    if not has_image:
        raise HTTPException(400, "train 还没有图片，先去 ① 整理 / ② 下载")

    with db.connection_for() as conn:
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            version_id=vid,
            kind="reg_build",
            params={
                "version_id": vid,
                "excluded_tags": list(body.excluded_tags),
                "auto_tag": bool(body.auto_tag),
                "api_source": body.api_source,
                "incremental": bool(body.incremental),
                "skip_similar": bool(body.skip_similar),
                "aspect_ratio_filter_enabled": bool(body.aspect_ratio_filter_enabled),
                "min_aspect_ratio": float(body.min_aspect_ratio),
                "max_aspect_ratio": float(body.max_aspect_ratio),
                "postprocess_method": body.postprocess_method,
                "postprocess_max_crop_ratio": float(body.postprocess_max_crop_ratio),
            },
        )
        if v["stage"] in ("curating", "tagging"):
            updated = versions.advance_stage(conn, vid, "regularizing")
            _publish_version_state(updated)
    _publish_job_state(job)
    return job


@app.get("/api/projects/{pid}/versions/{vid}/reg/caption")
def get_reg_caption(pid: int, vid: int, path: str) -> dict[str, Any]:
    """读 reg 集中单张图的 caption。`path` 是相对 reg/ 的路径（含子文件夹）。"""
    if not path or ".." in path or path.startswith("/") or path.startswith("\\"):
        raise HTTPException(400, "invalid path")
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    img = (rdir / path).resolve()
    try:
        img.relative_to(rdir.resolve())
    except ValueError:
        raise HTTPException(400, "path outside reg dir")
    if not img.exists() or img.suffix.lower() not in datasets.IMAGE_EXTS:
        raise HTTPException(404, "image not found")
    return {"path": path, "tags": tagedit.read_tags(img)}


@app.delete("/api/projects/{pid}/versions/{vid}/reg")
def delete_reg(pid: int, vid: int) -> dict[str, Any]:
    """清空 reg/ 内容（含 meta.json + 所有子文件夹），保留空目录本身。

    `versions.create_version` 总会建空 reg/；判定「存在」= 有 meta 或图片。
    """
    import shutil as _shutil
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    has_content = rdir.exists() and (
        (rdir / "meta.json").exists()
        or any(
            f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
            for f in rdir.rglob("*")
        )
    )
    if not has_content:
        return {"deleted": False, "reason": "reg empty"}
    try:
        for child in rdir.iterdir():
            if child.is_dir():
                _shutil.rmtree(child)
            else:
                child.unlink()
    except OSError as exc:
        raise HTTPException(500, f"删除失败: {exc}") from exc
    return {"deleted": True}


class RegAiRequest(BaseModel):
    excluded_tags: list[str] = []
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 4.0
    sampler_name: str = "er_sde"
    scheduler: str = "simple"
    seed: int = 0
    lora_configs: list[_LoraEntry] = []
    incremental: bool = False
    mixed_precision: str = "bf16"
    xformers: bool = False


@app.post("/api/projects/{pid}/versions/{vid}/reg/generate-ai")
def reg_generate_ai(pid: int, vid: int, body: RegAiRequest) -> dict[str, Any]:
    """逐图 AI 正则图生成：用每张 train 图的 tag 作 prompt，生成对应正则图。"""
    model_paths = _resolve_model_paths()
    _, _, vdir = _version_dir_or_404(pid, vid)
    train_dir = vdir / "train"
    if not train_dir.exists() or not any(
        f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
        for f in train_dir.rglob("*")
    ):
        raise HTTPException(400, "train 还没有图片，先去 ① 整理 / ② 下载")

    rdir = _reg_dir(vdir)
    rdir.mkdir(parents=True, exist_ok=True)

    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name=f"reg-ai p{pid}v{vid}", config_name="reg_ai", priority=0
        )
        db.update_task(conn, task_id, task_type="reg_ai")

    job_dir = GENERATE_JOBS_DIR / str(task_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    cfg = RegAiConfig(
        **model_paths,
        train_dir=str(train_dir),
        reg_dir=str(rdir),
        excluded_tags=list(body.excluded_tags),
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        steps=body.steps,
        cfg_scale=body.cfg_scale,
        sampler_name=body.sampler_name,
        scheduler=body.scheduler,
        seed=body.seed,
        lora_configs=[lc.model_dump() for lc in body.lora_configs],
        incremental=body.incremental,
        mixed_precision=body.mixed_precision,
        xformers=body.xformers,
    )

    cfg_path = GENERATE_CONFIGS_DIR / f"reg_ai_{task_id}.json"
    cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    with db.connection_for() as conn:
        db.update_task(conn, task_id, config_path=str(cfg_path))
        task = db.get_task(conn, task_id)

    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    return task or {"id": task_id}


@app.get("/api/projects/{pid}/versions/{vid}/reg/generate-ai/{task_id}")
def get_reg_ai_task(pid: int, vid: int, task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task or task.get("task_type") != "reg_ai":
        raise HTTPException(404)
    return task


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/config  (PP6.2 训练配置 — version 私有)
# ---------------------------------------------------------------------------


class FromPresetRequest(BaseModel):
    name: str  # 全局 preset 名


class SaveAsPresetRequest(BaseModel):
    name: str
    overwrite: bool = False


def _project_and_version_or_404(
    pid: int, vid: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    with db.connection_for() as conn:
        try:
            return version_config.get_project_and_version(conn, pid, vid)
        except version_config.VersionConfigError as exc:
            raise HTTPException(404, str(exc)) from exc


@app.get("/api/projects/{pid}/versions/{vid}/config")
def get_version_config_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """读 version 私有 config；不存在返回 has_config=false / config=null。"""
    project, ver = _project_and_version_or_404(pid, vid)
    if not version_config.has_version_config(project, ver):
        return {
            "has_config": False,
            "config": None,
            "project_specific_fields": sorted(version_config.PROJECT_SPECIFIC_FIELDS),
        }
    try:
        cfg = version_config.read_version_config(project, ver)
    except version_config.VersionConfigError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "has_config": True,
        "config": cfg,
        "project_specific_fields": sorted(version_config.PROJECT_SPECIFIC_FIELDS),
    }


@app.put("/api/projects/{pid}/versions/{vid}/config")
def put_version_config_endpoint(
    pid: int, vid: int, body: dict[str, Any]
) -> dict[str, Any]:
    """直接写 version 私有 config（全量替换）。

    PP10.4：项目特定字段（data_dir / output_dir / output_name 等）**不**强制
    覆盖。fork_preset 时已经预填好；用户在 Train 页可以自由改（例如
    `resume_lora` 接续训练、自定义 output_name）。改坏了再换一次预设回到
    默认。
    """
    project, ver = _project_and_version_or_404(pid, vid)
    try:
        version_config.write_version_config(
            project, ver, body, force_project_overrides=False
        )
        cfg = version_config.read_version_config(project, ver)
    except version_config.VersionConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"has_config": True, "config": cfg}


@app.post("/api/projects/{pid}/versions/{vid}/config/from_preset")
def fork_preset_for_version_endpoint(
    pid: int, vid: int, body: FromPresetRequest
) -> dict[str, Any]:
    """从全局 preset 复制一份进 version 私有 config（应用项目特定字段）。"""
    project, ver = _project_and_version_or_404(pid, vid)
    try:
        cfg = preset_flow.fork_preset_for_version(body.name, project, ver)
    except presets_io.PresetError as exc:
        raise HTTPException(_err_code(exc), str(exc)) from exc
    except version_config.VersionConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    # 同步 versions.config_name = 来源 preset 名（informational only）
    with db.connection_for() as conn:
        versions.update_version(conn, vid, config_name=body.name)
    return {"has_config": True, "config": cfg, "from_preset": body.name}


@app.post("/api/projects/{pid}/versions/{vid}/config/save_as_preset")
def save_version_config_as_preset_endpoint(
    pid: int, vid: int, body: SaveAsPresetRequest
) -> dict[str, Any]:
    """version 私有 config → 全局 preset（清掉项目特定字段）。"""
    project, ver = _project_and_version_or_404(pid, vid)
    try:
        cfg = preset_flow.save_version_config_as_preset(
            project, ver, body.name, overwrite=body.overwrite
        )
    except presets_io.PresetError as exc:
        raise HTTPException(_err_code(exc), str(exc)) from exc
    except version_config.VersionConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"saved_preset": body.name, "config": cfg}


@app.post("/api/projects/{pid}/versions/{vid}/queue")
def enqueue_version_training(pid: int, vid: int) -> dict[str, Any]:
    """PP6.3 — 把 version 入队训练。

    校验：
    - version 已配置训练参数（version_config 存在）
    - 该 version 没有 active task（pending / running）
    """
    project, ver = _project_and_version_or_404(pid, vid)
    if not version_config.has_version_config(project, ver):
        raise HTTPException(
            400, "请先在 ⑥ 训练页选预设并保存配置后再入队"
        )
    cfg_path = version_config.version_config_path(project, ver)

    with db.connection_for() as conn:
        # 该 version 当前是否已有 active task
        active = conn.execute(
            "SELECT id, status FROM tasks "
            "WHERE version_id = ? AND status IN ('pending', 'running') "
            "LIMIT 1",
            (vid,),
        ).fetchone()
        if active:
            raise HTTPException(
                409,
                f"该版本已有 active task #{active['id']}（{active['status']}），"
                "请等其完成或取消",
            )

        # 创建 task
        slug = project["slug"]
        label = ver["label"]
        task_name = f"{slug}_{label}"
        config_name = ver["config_name"] or f"proj_{pid}_{label}"  # informational
        cur = conn.execute(
            "INSERT INTO tasks(name, config_name, status, priority, created_at, "
            "project_id, version_id, config_path) "
            "VALUES (?, ?, 'pending', 0, ?, ?, ?, ?)",
            (task_name, config_name, time.time(), pid, vid, str(cfg_path)),
        )
        tid = int(cur.lastrowid)
        conn.commit()

        # 推 stage：configured → training（task 启动后由 supervisor 推 done）
        if ver["stage"] in ("ready", "configured", "regularizing", "tagging", "curating"):
            updated = versions.advance_stage(conn, vid, "training")
            _publish_version_state(updated)
        if project["stage"] in ("created", "downloading", "curating", "tagging", "regularizing"):
            up = projects.advance_stage(conn, pid, "training")
            _publish_project_state(up)

        task = db.get_task(conn, tid)
    bus.publish({
        "type": "task_state_changed",
        "task_id": tid,
        "status": "pending",
    })
    return task or {}


# version 级缩略图：bucket = train | reg | samples（PP3 加 train，reg/samples 留作 PP4-5）
@app.get("/api/projects/{pid}/versions/{vid}/thumb")
def version_thumb(
    pid: int,
    vid: int,
    bucket: str = "train",
    folder: str = "",
    name: str = "",
    size: int = 256,
) -> FileResponse:
    if bucket not in {"train", "reg", "samples"}:
        raise HTTPException(400, f"非法 bucket: {bucket}")
    if "/" in name or "\\" in name or ".." in name or not name:
        raise HTTPException(400, "invalid name")
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        p = projects.get_project(conn, pid)
    if not v or not p or v["project_id"] != pid:
        raise HTTPException(404, "版本不存在")
    vdir = versions.version_dir(p["id"], p["slug"], v["label"]) / bucket
    if bucket in {"train", "reg"}:
        if not folder or "/" in folder or "\\" in folder or ".." in folder:
            raise HTTPException(400, "invalid folder")
        f = vdir / folder / name
    else:
        f = vdir / name
    if not f.exists() or f.suffix.lower() not in datasets.IMAGE_EXTS:
        logger.info(
            "version thumb 404: pid=%s vid=%s bucket=%s folder=%s name=%s -> %s",
            pid, vid, bucket, folder, name, f,
        )
        raise HTTPException(404)
    return _thumb_response(f, size)


# ---------------------------------------------------------------------------
# /api/generate  — 独立图片生成（复用 anima_generate.py 推理链路）
# ---------------------------------------------------------------------------


class _LoraEntry(BaseModel):
    path: str
    scale: float = 1.0


class GenerateRequest(BaseModel):
    prompts: list[str] = ["newest, safe, 1girl, masterpiece, best quality"]
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 4.0
    sampler_name: str = "er_sde"
    scheduler: str = "simple"
    count: int = 1
    seed: int = 0
    lora_configs: list[_LoraEntry] = []
    mixed_precision: str = "bf16"
    xformers: bool = False


def _resolve_model_paths() -> dict[str, str]:
    """从 secrets 解析默认模型路径（与 version_config 逻辑对齐）。"""
    from .services.model_downloader import models_root
    root = models_root()
    return {
        "transformer_path": str(root / "diffusion_models" / "anima-preview3-base.safetensors"),
        "vae_path": str(root / "vae" / "qwen_image_vae.safetensors"),
        "text_encoder_path": str(root / "text_encoders"),
        "t5_tokenizer_path": str(root / "t5_tokenizer"),
    }


@app.post("/api/generate")
def enqueue_generate(body: GenerateRequest) -> dict[str, Any]:
    """创建独立图片生成任务并入队。"""
    model_paths = _resolve_model_paths()

    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name="generate", config_name="generate", priority=0
        )
        db.update_task(conn, task_id, task_type="generate")

    job_dir = GENERATE_JOBS_DIR / str(task_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    cfg = GenerateConfig(
        **model_paths,
        output_dir=str(job_dir),
        prompts=body.prompts,
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        steps=body.steps,
        cfg_scale=body.cfg_scale,
        sampler_name=body.sampler_name,
        scheduler=body.scheduler,
        count=body.count,
        seed=body.seed,
        lora_configs=[lc.model_dump() for lc in body.lora_configs],
        mixed_precision=body.mixed_precision,
        xformers=body.xformers,
    )

    cfg_path = GENERATE_CONFIGS_DIR / f"gen_{task_id}.json"
    cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    with db.connection_for() as conn:
        db.update_task(conn, task_id, config_path=str(cfg_path))
        task = db.get_task(conn, task_id)

    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    return task or {"id": task_id}


@app.get("/api/generate")
def list_generate_tasks(status: Optional[str] = None) -> dict[str, Any]:
    """列出所有生成任务。"""
    if status and status not in db.VALID_STATUSES:
        raise HTTPException(400, f"unknown status: {status}")
    with db.connection_for() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE task_type = 'generate' "
            + ("AND status = ? " if status else "")
            + "ORDER BY created_at DESC",
            (status,) if status else (),
        ).fetchall()
    return {"items": [dict(r) for r in rows]}


@app.get("/api/generate/{task_id}")
def get_generate_task(task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task or task.get("task_type") != "generate":
        raise HTTPException(404)
    return task


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
    """已结束任务重新入队：复制完整训练上下文创建新 task。

    需要复制的字段（PP6.1+ 引入；老的 retry 只复制 name/config_name/priority
    会让 supervisor 走老降级路径用全局 preset 而不是 version 私有 config，
    导致重试参数与原任务不一致）：
    - config_path：version 私有 config 的绝对路径
    - project_id / version_id：用于 monitor_state_path 解析与 stage 推进

    不复制：status / pid / *_at / exit_code / error_msg / monitor_state_path
    （都是「上次跑」的产物；新任务从 pending 开始，supervisor 会重新解析）。
    """
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
        copy_fields: dict[str, Any] = {}
        for k in ("config_path", "project_id", "version_id"):
            if original.get(k) is not None:
                copy_fields[k] = original[k]
        if copy_fields:
            db.update_task(conn, new_id, **copy_fields)
        new_task = db.get_task(conn, new_id)
    bus.publish(
        {"type": "task_state_changed", "task_id": new_id, "status": "pending"}
    )
    return new_task or {"id": new_id}


# ---------------------------------------------------------------------------
# /api/queue/{task_id}/outputs — 查看 / 下载训练产物
# ---------------------------------------------------------------------------


_LOCALHOST_HOSTS = {"127.0.0.1", "::1", "localhost"}
_LORA_EXTS = {".safetensors", ".ckpt", ".pt", ".bin"}


def _task_output_dir(task: dict[str, Any]) -> Optional[Path]:
    """根据 task 推断 output 目录：versions/{label}/output。

    没 project_id / version_id 的老任务（PP1 之前）→ 返回 None；调用方应该
    处理为「无 output 目录」。
    """
    pid = task.get("project_id")
    vid = task.get("version_id")
    if not (pid and vid):
        return None
    with db.connection_for() as conn:
        v = versions.get_version(conn, int(vid))
        p = projects.get_project(conn, int(pid))
    if not v or not p or v["project_id"] != pid:
        return None
    return versions.version_dir(int(pid), p["slug"], v["label"]) / "output"


def _is_loopback(request: Request) -> bool:
    client = request.client
    return bool(client and client.host in _LOCALHOST_HOSTS)


@app.get("/api/queue/{task_id}/outputs")
def list_task_outputs(task_id: int, request: Request) -> dict[str, Any]:
    """列出 task 关联 version 的 output 目录里所有文件。

    `supports_open_folder` 仅在请求来自 loopback（同机浏览器）时为 True；
    云端部署时永远为 False，避免前端显示一个无意义按钮。
    """
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    out_dir = _task_output_dir(task)
    files: list[dict[str, Any]] = []
    if out_dir and out_dir.exists():
        for f in sorted(out_dir.iterdir(), key=lambda p: p.name):
            if not f.is_file():
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            files.append({
                "name": f.name,
                "size": st.st_size,
                "mtime": st.st_mtime,
                "is_lora": f.suffix.lower() in _LORA_EXTS,
            })
    return {
        "task_id": task_id,
        "output_dir": str(out_dir) if out_dir else None,
        "exists": bool(out_dir and out_dir.exists()),
        "supports_open_folder": _is_loopback(request),
        "files": files,
    }


@app.get("/api/queue/{task_id}/outputs.zip")
def download_task_outputs_zip(
    task_id: int, background: BackgroundTasks
) -> FileResponse:
    """把 output 目录全部文件打包成 zip 一次性下载。

    实现：写到临时文件再 FileResponse；响应发完后用 BackgroundTasks 清理。
    safetensors / pt 几乎都是已压缩二进制，用 ZIP_STORED（不再压缩，CPU 省一倍）。
    """
    import tempfile
    import zipfile
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    out_dir = _task_output_dir(task)
    if not out_dir or not out_dir.exists():
        raise HTTPException(404, "no output dir")
    files = [f for f in sorted(out_dir.iterdir()) if f.is_file()]
    if not files:
        raise HTTPException(404, "empty output dir")

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(
            tmp_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    background.add_task(lambda: tmp_path.unlink(missing_ok=True))

    archive_name = f"task_{task_id}_outputs.zip"
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=archive_name,
        background=background,
    )


@app.get("/api/queue/{task_id}/output/{filename}")
def download_task_output(task_id: int, filename: str) -> FileResponse:
    """下载 output 目录下的指定文件。`Content-Disposition: attachment` 让
    浏览器走「保存」对话框而不是 inline 渲染。"""
    if "/" in filename or "\\" in filename or ".." in filename or not filename:
        raise HTTPException(400, "invalid filename")
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    out_dir = _task_output_dir(task)
    if not out_dir or not out_dir.exists():
        raise HTTPException(404, "no output dir")
    f = out_dir / filename
    if not f.exists() or not f.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(
        f,
        media_type="application/octet-stream",
        filename=filename,  # 让 starlette 自动加 Content-Disposition
    )


@app.post("/api/queue/{task_id}/open-folder")
def open_task_folder(task_id: int, request: Request) -> dict[str, Any]:
    """在 server 主机上用 OS 文件管理器打开 output 目录。

    **仅 loopback 请求允许**：云端部署时浏览器不在 server 那台机，开了用户也
    看不到，反而是远程命令执行入口；这里直接 403 拒绝。
    """
    if not _is_loopback(request):
        raise HTTPException(
            403, "open-folder is only available for local (loopback) requests"
        )
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    out_dir = _task_output_dir(task)
    if not out_dir or not out_dir.exists():
        raise HTTPException(404, "no output dir")
    try:
        import platform
        import subprocess as _sub
        sysname = platform.system()
        if sysname == "Windows":
            os.startfile(str(out_dir))  # type: ignore[attr-defined]
        elif sysname == "Darwin":
            _sub.Popen(["open", str(out_dir)])
        else:
            _sub.Popen(["xdg-open", str(out_dir)])
    except Exception as exc:
        raise HTTPException(500, f"failed to open folder: {exc}") from exc
    return {"opened": str(out_dir)}


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
def get_sample(
    filename: str,
    task_id: Optional[int] = None,
    w: Optional[int] = None,
) -> FileResponse:
    """采样图代理。

    `?task_id=N` 给了 → 在该任务 monitor_state_path 周围按多个候选目录查找：
    - `monitor_state.json` 同级 `samples/`（PP6.1 当初约定）
    - `monitor_state.json` 同级 `output/samples/`（anima_train 实际写法 ——
      sample_dir = output_dir/samples，output_dir 通常是 versions/{label}/output）
    - 同级 `output/<任意子目录>/samples/`（兜底防 anima_train 用别的 output 名）

    没给 task_id → 兜底全局 OUTPUT_DIR/samples/（旧训练直接命令行的兼容）。

    `?w=N` 给了 → 走 thumb_cache 生成 N px 缩略图（用于监控页缩略图条）；
    不给 → 返回原图。两种都走 _thumb_response 的弱 etag + no-cache，浏览器
    304 命中即可，避免「重启窗口期失败响应被永久缓存」问题。
    """
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")

    resolved: Optional[Path] = None
    if task_id is not None:
        with db.connection_for() as conn:
            row = conn.execute(
                "SELECT monitor_state_path FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if not row or not row["monitor_state_path"]:
            raise HTTPException(404)
        monitor_dir = Path(row["monitor_state_path"]).parent
        candidates = [
            monitor_dir / "samples" / filename,
            monitor_dir / "output" / "samples" / filename,
        ]
        # 再扫一层 output/<sub>/samples/ 兜底（用户改 output_dir 名字时仍能找到）
        output_root = monitor_dir / "output"
        if output_root.is_dir():
            for sub in output_root.iterdir():
                if sub.is_dir():
                    candidates.append(sub / "samples" / filename)
        for p in candidates:
            if p.exists():
                resolved = p
                break
        if resolved is None:
            logger.info(
                "sample 404: task_id=%s file=%s tried=%s",
                task_id, filename, [str(p) for p in candidates],
            )
            raise HTTPException(404)
    else:
        path = OUTPUT_DIR / "samples" / filename
        if not path.exists():
            raise HTTPException(404)
        resolved = path

    # w 给了走缩略图；w<=0 或没给 → 原图。复用 thumb_cache，盘上落 .jpg；
    # 浏览器走弱 etag + no-cache，304 命中很轻。
    if w is not None and w > 0:
        return _thumb_response(resolved, w)
    return _thumb_response(resolved, 0)  # size=0 内部直接返回 src，不缩


# ---------------------------------------------------------------------------
# 静态资源
# ---------------------------------------------------------------------------


class SPAStaticFiles(StaticFiles):
    """SPA 路由兜底：未命中实际文件且不像静态资产时，返回 index.html。

    这样直接刷新 `/studio/projects/1/v/1/curate` 这种 react-router 路由
    也能拿到 index.html，让 BrowserRouter 在前端解析路径。
    带文件扩展名的请求（.js/.css/.png 等）保持原 404 行为，避免把缺失的
    资源吞成 200 误导浏览器。
    """

    async def get_response(self, path, scope):  # type: ignore[override]
        from starlette.exceptions import HTTPException as StarletteHTTPException
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            # 末段含 "." → 视为静态资产请求，不兜底
            last = path.rsplit("/", 1)[-1]
            if "." in last:
                raise
            return FileResponse(Path(self.directory) / "index.html")


# React 应用：构建后通过 /studio 访问。开发期请用 `npm run dev` 起 5173。
if WEB_DIST.exists():
    app.mount(
        "/studio",
        SPAStaticFiles(directory=str(WEB_DIST), html=True),
        name="studio",
    )


@app.get("/", response_model=None)
def root() -> RedirectResponse | JSONResponse:
    """根路径 302 跳转到 React 应用 `/studio/`。

    旧监控页仍可通过 `/monitor_smooth.html` 直达（QueueMonitor iframe 用）。
    若前端尚未构建（dist 缺失），返回 JSON 提示。"""
    if WEB_DIST.exists():
        return RedirectResponse(url="/studio/", status_code=302)
    return JSONResponse(
        {
            "message": "AnimaStudio is running. Build the React app at studio/web/ "
            "(npm install && npm run build) to enable the new UI."
        }
    )


@app.get("/monitor_smooth.html", response_model=None, include_in_schema=False)
def monitor_smooth_html() -> FileResponse:
    """直接路径访问 monitor_smooth.html（PP6.1：QueueMonitor iframe 走这里）。"""
    if not LEGACY_MONITOR_HTML.exists():
        raise HTTPException(404, "monitor_smooth.html not found")
    return FileResponse(LEGACY_MONITOR_HTML)


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

    # 真正给用户看的入口是 /studio/（前端 SPA），裸根路径只是兼容旧 monitor。
    print(f"[AnimaStudio] http://{args.host}:{args.port}/studio/")
    uvicorn.run(
        "studio.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
