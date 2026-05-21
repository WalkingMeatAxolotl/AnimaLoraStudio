"""AnimaStudio 守护服务（FastAPI）。

P1 范围（本文件目前实现）：
    - GET  /                   302 跳转到 /studio/
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
import threading
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
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator

from . import (
    __version__,
    browse,
    curation,
    datasets,
    db,
    preprocess as preprocess_svc,
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
    flash_attention_setup,
    onnxruntime_setup,
    pending_install,
    preprocess_manifest,
    release_notes as release_notes_svc,
    torch_setup,
    reg_builder,
    system_stats,
    tagedit,
    train_io,
    updater,
    uploads as uploads_svc,
    version_config,
    xformers_setup,
)
from .services.tagger import VALID_TAGGER_NAMES, get_tagger
from .paths import (
    LOGS_DIR,
    OUTPUT_DIR,
    REPO_ROOT,
    STUDIO_DATA,
    STUDIO_DB,
    USER_PRESETS_DIR,
    WEB_DIST,
    ensure_dirs,
    safe_join,
    validate_path_component,
)
from .schema import (
    GROUP_ORDER,
    AttentionBackend,
    GenerateConfig,
    LoraEntry,
    RegAiConfig,
    TrainingConfig,
    XYMatrixSpec,
    migrate_legacy_attention,
)
from .supervisor import Supervisor

ensure_dirs()
db.init_db()

logger = logging.getLogger(__name__)


def _safe_join_or_400(base: Path, *parts: str) -> Path:
    """safe_join 的 HTTPException 版本。把 ValueError 包成 400。"""
    try:
        return safe_join(base, *parts)
    except ValueError as exc:
        raise HTTPException(400, f"invalid path: {exc}") from exc


def _validate_component_or_400(name: str) -> None:
    """validate_path_component 的 HTTPException 版本（用于不需要 join 的纯名校验）。"""
    try:
        validate_path_component(name)
    except ValueError as exc:
        raise HTTPException(400, f"invalid path: {exc}") from exc


def _install_proactor_disconnect_filter(loop: asyncio.AbstractEventLoop) -> None:
    """吞 Windows + asyncio Proactor 的 cosmetic ConnectionResetError 噪声。

    Python asyncio 在 Windows 上有 [bpo-44291](https://github.com/python/cpython/issues/87691) 类问题：
    远端 TCP 强制断开（用户关 tab / 刷新 / SSE 重连，WinError 10054 / 10053）
    时 `_ProactorBasePipeTransport._call_connection_lost` 走 `socket.shutdown()`
    抛 `ConnectionResetError` / `ConnectionAbortedError`，但 callback 内部
    没 catch 这两个 expected error，asyncio 默认 handler 打 traceback 到
    stderr。server 完全没事，只是日志被刷一行无意义 stack。

    精确过滤：只在 exception 是 ConnectionResetError / ConnectionAbortedError
    且 handle repr 含 `_call_connection_lost` 时静默吞掉；其它 asyncio
    异常仍交给 default handler。仅 Windows 装；其它平台用 SelectorEventLoop
    没这个 bug。
    """
    if os.name != "nt":
        return

    def _filter(loop_: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            handle = context.get("handle")
            if handle and "_call_connection_lost" in repr(handle):
                return
        loop_.default_exception_handler(context)

    loop.set_exception_handler(_filter)


@asynccontextmanager
async def _lifespan(app_: FastAPI) -> AsyncIterator[None]:
    """启动绑定 event bus 到当前 loop 并起 supervisor；关闭时停 supervisor。"""
    # 装 Windows ProactorEventLoop 的 ConnectionResetError 过滤器（详见 helper docstring）
    _install_proactor_disconnect_filter(asyncio.get_running_loop())

    # 测试出图 tempdir 遗留清扫（防 supervisor crash 泄漏 anima_gen_* 目录）
    from .services.inference_core import cleanup_stale_generate_tempdirs
    from .services import generate_cache, model_downloader as _md
    cleanup_stale_generate_tempdirs()

    # TAEFlux（中间步预览）后台下载：跟 server 一起启动；下载失败不阻塞 server。
    # 如果已下载则 noop；下载期间用户能正常用其他功能，预览功能等下载完才生效。
    def _bg_download_taeflux() -> None:
        try:
            if _md.taeflux_available():
                return
            logger.info("background-downloading TAEFlux (~1.6MB)…")
            ok = _md.download_taeflux(on_log=lambda m: logger.info("[taeflux] %s", m))
            if not ok:
                logger.warning("taeflux background download failed; preview disabled until manual install")
        except Exception:
            logger.exception("taeflux background download crashed")
    threading.Thread(target=_bg_download_taeflux, name="taeflux-bg-download", daemon=True).start()

    bus.attach_loop(asyncio.get_running_loop())

    # commit 11：SSE 客户端断连 + 30s 缓冲后清 generate cache。
    # 防刷新/短抖动：用户重连（_on_first_subscribe）取消计时器。
    _disconnect_timer: dict[str, Optional[threading.Timer]] = {"t": None}

    def _on_last_unsubscribe() -> None:
        # 已有 timer 不重置（多个客户端各自 unsubscribe 时，最后一个才是关键）
        if _disconnect_timer["t"] is not None:
            return
        timer = threading.Timer(30.0, _flush_cache)
        timer.daemon = True
        _disconnect_timer["t"] = timer
        timer.start()

    def _on_first_subscribe() -> None:
        timer = _disconnect_timer.get("t")
        if timer is not None:
            timer.cancel()
            _disconnect_timer["t"] = None

    def _flush_cache() -> None:
        n = generate_cache.total_count()
        if n:
            generate_cache.clear_all()
            logger.info("flushed generate cache (%d images) after SSE idle", n)
        _disconnect_timer["t"] = None

    bus.set_connection_callbacks(
        on_first_subscribe=_on_first_subscribe,
        on_last_unsubscribe=_on_last_unsubscribe,
    )

    sup = Supervisor(on_event=bus.publish)
    sup.start()
    app_.state.supervisor = sup

    # PR #37: system stats SSE — 后台 sampler 每 2.5s 采集 + bus.publish。前端
    # 只 mount 时 GET 一次冷启动，避免 cloud 部署被每客户端独立轮询污染。
    def _publish_system_stats(payload: dict[str, Any]) -> None:
        bus.publish({"type": "system_stats_updated", "payload": payload})

    sys_sampler = system_stats.SystemStatsSampler(_publish_system_stats)
    sys_sampler.start()
    app_.state.system_stats_sampler = sys_sampler

    try:
        yield
    finally:
        # 取消可能挂着的 disconnect timer，shutdown 阶段不需要再延迟
        timer = _disconnect_timer.get("t")
        if timer is not None:
            timer.cancel()
        sys_sampler.stop()
        sup.stop()
        # commit 11：lifespan shutdown 清掉所有图 cache（释放内存 + 干净退出）
        generate_cache.clear_all()


app = FastAPI(title="AnimaStudio", version=__version__, lifespan=_lifespan)


# ── gzip ────────────────────────────────────────────────────────────
# 给 JSON / 文本响应自动 gzip 压缩。对 /api/state（10k 步训练 ~500KB → ~100KB）
# 这种大数组高度可压；对小响应 (< 1000B) 跳过避免 framing overhead 反而变大。
#
# 显式排除两类路径：
#   - /api/events：SSE 流。GZipMiddleware 会 buffer chunks 到 minimum_size
#     才发，破坏 SSE 实时性 + 某些 EventSource 实现解析 gzip 流有兼容问题
#   - /samples/*：图片字节（PNG/JPEG/WEBP 已经是压缩格式），gzip 再压净浪费
#     CPU 且 size 略增
_GZIP_SKIP_PREFIXES = ("/api/events", "/samples/")


class _SelectiveGZipMiddleware(GZipMiddleware):
    """GZipMiddleware 的子类，按 path 前缀绕过指定路由。"""

    async def __call__(self, scope, receive, send):  # type: ignore[override]
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if any(path.startswith(p) for p in _GZIP_SKIP_PREFIXES):
                await self.app(scope, receive, send)
                return
        await super().__call__(scope, receive, send)


app.add_middleware(_SelectiveGZipMiddleware, minimum_size=1000)


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


@app.get("/api/system/stats")
def get_system_stats() -> dict[str, Any]:
    """Topbar 系统资源小组件用 (CPU/RAM/GPU/VRAM)。前端按 2-3s 轮询。"""
    return system_stats.stats_to_json(system_stats.collect_stats())


@app.get("/api/state")
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

    # 服务端下采样 losses / lr_history（samples cap 50 已经在 train_monitor 端做了）
    if max_points and max_points > 0:
        from runtime.train_monitor import _downsample_uniform
        if isinstance(data.get("losses"), list):
            data["losses"] = _downsample_uniform(data["losses"], max_points)
        if isinstance(data.get("lr_history"), list):
            data["lr_history"] = _downsample_uniform(data["lr_history"], max_points)

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


@app.get("/api/presets/{name}/download")
def download_preset(name: str) -> FileResponse:
    """端到端文件 I/O：直接返回 `studio_data/presets/{name}.yaml` 原文件。"""
    try:
        path = presets_io.preset_path(name)
    except presets_io.PresetError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"预设不存在: {name}")
    return FileResponse(path, media_type="application/yaml", filename=f"{name}.yaml")


@app.post("/api/presets/import")
async def import_preset(file: UploadFile = File(...)) -> dict[str, Any]:
    """接 .yaml/.yml/.json 上传 → 解析 + schema 校验 → 落盘到 `suggested_name`。

    无冲突 → write_preset 直接写,返回 200 `{name, path}`。
    冲突(`suggested_name.yaml` 已存在)→ 409 + 结构化 detail
    `{message, config, suggested_name}`,前端 ImportConflictDialog 让用户选
    覆盖 / 另存为 / 取消,选定后走 PUT /api/presets/{name} 完成落盘。
    解析/校验失败 → 400/422。
    """
    raw = await file.read()
    try:
        config, suggested = presets_io.parse_preset_bytes(raw, file.filename or "")
    except presets_io.PresetError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc
    if presets_io.preset_path(suggested).exists():
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"预设已存在: {suggested}",
                "config": config,
                "suggested_name": suggested,
            },
        )
    try:
        path = presets_io.write_preset(suggested, config)
    except presets_io.PresetError as exc:
        raise HTTPException(status_code=_err_code(exc), detail=str(exc)) from exc
    return {"name": suggested, "path": str(path)}


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


@app.get("/api/models/path-defaults")
def get_models_path_defaults() -> dict[str, str]:
    """当前 Settings 算出的 4 个模型字段绝对路径。

    给预设页 reset 按钮和「新建预设」初始填充用——这两个场景没有 project
    上下文，拿不到 /api/projects/{pid}/versions/{vid}/config 里的
    project_specific_defaults，所以单独开一个端点。
    """
    return model_downloader.default_paths_for_new_version()


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


class UpscalerSelectRequest(BaseModel):
    label: str   # 预设 key 或 custom 文件名


@app.post("/api/upscalers/select")
def select_upscaler(body: UpscalerSelectRequest) -> dict[str, Any]:
    """切换默认放大器。写入 secrets.models.selected_upscaler。

    接受预设 label 或本地已有的 custom 文件名；非法值（既不在预设也不在
    upscalers/ 目录）返回 400。
    """
    label = body.label.strip()
    if not label:
        raise HTTPException(400, "label 不能为空")
    valid = label in model_downloader.UPSCALER_VARIANTS
    if not valid:
        # custom 文件名：必须已经在磁盘上
        try:
            target = model_downloader.upscaler_target(label)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not target.exists():
            raise HTTPException(
                404, f"放大器不存在: {label}（既非预设也未在 upscalers/ 找到）"
            )
    cur = secrets.load()
    new_models = cur.models.model_copy(update={"selected_upscaler": label})
    new = cur.model_copy(update={"models": new_models})
    secrets.save(new)
    return {"selected": label}


class UpscalerCustomDownloadRequest(BaseModel):
    source: str   # "hf" | "ms"
    repo_id: str  # 例 "Kim2091/UltraSharp" 或 ModelScope 同形式
    filename: str  # 例 "4x-UltraSharp.pth"


@app.post("/api/upscalers/download_custom")
def start_upscaler_custom_download(
    body: UpscalerCustomDownloadRequest,
) -> dict[str, Any]:
    """自定义放大器下载：用户填 HF/MS repo + 文件名，落到 `{upscalers}/{filename}`。

    复用通用 start_download_async；key 形如 `upscaler:custom:foo.pth` 便于前端 SSE
    过滤 + catalog 状态匹配。
    """
    from pathlib import Path as _Path

    if body.source not in ("hf", "ms"):
        raise HTTPException(400, f"未知下载源: {body.source}")
    if not body.repo_id.strip() or not body.filename.strip():
        raise HTTPException(400, "repo_id / filename 不能为空")
    save_name = _Path(body.filename).name
    if not save_name.lower().endswith(model_downloader.UPSCALER_EXTS):
        raise HTTPException(
            400,
            f"仅支持 {model_downloader.UPSCALER_EXTS} 扩展名",
        )
    key = f"upscaler:custom:{save_name}"
    model_downloader.start_download_async(
        key,
        lambda log: model_downloader.download_upscaler_custom(
            body.source, body.repo_id, body.filename, on_log=log
        ),
    )
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
    trigger_word: Optional[str] = None


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
            projects.delete_project(conn, pid)
        except projects.ProjectError as exc:
            raise HTTPException(_project_err_code(exc), str(exc)) from exc
    return {"deleted": pid}


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


@app.get("/api/projects/{pid}/versions/{vid}/lora_ckpts")
def list_version_lora_ckpts(pid: int, vid: int) -> dict[str, Any]:
    """列出 version output/ 下所有 .safetensors（step / epoch / final），
    用于 LoRA picker 第二层（XY ckpt 轴 + 单图模式切 ckpt）。"""
    p, v, vdir = _version_dir_or_404(pid, vid)
    return {"items": versions.list_lora_ckpts(vdir)}


@app.get("/api/projects/{pid}/state_ckpts")
def list_project_state_ckpts(pid: int) -> dict[str, Any]:
    """列出项目所有 versions 的 training_state_step*.pt，按 version 分组。

    给 Train 页 resume_state 字段的「浏览本项目」picker 用：用户看 version
    分组的语义化文件列表，选中后前端把绝对路径写入字段。
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise HTTPException(404, f"项目不存在: id={pid}")
        return {"groups": versions.list_project_state_ckpts(conn, p)}


@app.get("/api/projects/{pid}/lora_ckpts")
def list_project_lora_ckpts(pid: int) -> dict[str, Any]:
    """列出项目所有 versions 的 LoRA ckpt（.safetensors），按 version 分组。

    给 Train 页 resume_lora 字段的「浏览本项目」picker 用。
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise HTTPException(404, f"项目不存在: id={pid}")
        return {"groups": versions.list_project_lora_ckpts(conn, p)}


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

    打包完成 / 失败 publish version_train_zip_ready / _failed —— 前端用 <a>
    直链触发下载（浏览器原生进度条），SSE 事件用于清 app-side "打包中..." 状态
    + 失败时弹 toast。和 outputs.zip 一套范式。
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
            bus.publish({
                "type": "version_train_zip_failed",
                "project_id": pid,
                "version_id": vid,
                "error": str(exc),
            })
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({
                "type": "version_train_zip_failed",
                "project_id": pid,
                "version_id": vid,
                "error": str(exc),
            })
            raise

    bus.publish({
        "type": "version_train_zip_ready",
        "project_id": pid,
        "version_id": vid,
    })
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


# ---------------------------------------------------------------------------
# /api/projects/{pid}/preprocess/* — 预处理阶段（下载与筛选之间）
# 第一阶段只做放大（spandrel + 4x-AnimeSharp）；裁剪 / 涂抹后续 PR。
# ---------------------------------------------------------------------------


class PreprocessStartRequest(BaseModel):
    mode: str = "all"  # all | selected | all_force
    names: Optional[list[str]] = None
    model: str = preprocess_svc.DEFAULT_MODEL
    tile_size: int = preprocess_svc.DEFAULT_TILE_SIZE
    tile_pad: int = preprocess_svc.DEFAULT_TILE_PAD
    device: str = preprocess_svc.DEFAULT_DEVICE
    # target_area=None 走纯 4× 模型；非 None 走智能（够大跳模型 + LANCZOS 缩到目标）
    target_area: Optional[int] = preprocess_svc.DEFAULT_TARGET_AREA


class PreprocessRestoreRequest(BaseModel):
    """还原已处理图：删 manifest entry + 删 preprocess/{name} PNG。

    还原后该图回到「隐式 original」状态——下游 resolver 重新指向 download/。
    见 ADR 0004。
    """
    names: list[str]


# 旧字段名兼容（前端切换期间，PreprocessDeleteRequest = PreprocessRestoreRequest）
PreprocessDeleteRequest = PreprocessRestoreRequest


class CropRect(BaseModel):
    """归一化裁剪矩形 [0..1]^4。x/y = 左上角，w/h = 宽高。"""
    x: float
    y: float
    w: float
    h: float
    label: Optional[str] = None


class PreprocessCropRequest(BaseModel):
    """裁剪 job 输入：源文件名 → 一个或多个归一化矩形。

    源文件名为 preprocess/ 下当前文件名（或 download/ 文件名兜底，若 preprocess/
    没对应）。每个矩形产出一张 PNG：N=1 覆盖 stem.png；N>1 输出 stem_c0.png /
    stem_c1.png / ... 并删除原 stem.png。
    """
    crops: dict[str, list[CropRect]]


@app.post("/api/projects/{pid}/preprocess/start")
def start_preprocess(pid: int, body: PreprocessStartRequest) -> dict[str, Any]:
    """开始预处理 job（当前只放大）。

    mode='all' 增量跳过已处理；'all_force' 全部重跑；'selected' 处理 names。
    返回新建的 job 行。
    """
    if body.mode not in ("all", "selected", "all_force"):
        raise HTTPException(400, f"未知 mode: {body.mode}")
    if body.tile_size <= 0:
        raise HTTPException(400, "tile_size 必须 > 0")
    if body.device not in ("auto", "cuda", "cpu"):
        raise HTTPException(400, f"未知 device: {body.device}")
    # 边界：合理面积区间 256² ~ 4096²（再大就该自己写脚本了），None 表示关闭智能模式
    if body.target_area is not None and (
        body.target_area < 256 * 256 or body.target_area > 4096 * 4096
    ):
        raise HTTPException(400, f"target_area 超出范围: {body.target_area}")

    # 模型权重必须先下载（避免 worker 启起来才报错）。
    # body.model 可以是预设 label 或 custom filename（带扩展名）；
    # upscaler_target 内部做穿越保护 + 扩展名白名单。
    try:
        target = model_downloader.upscaler_target(body.model)
    except ValueError as exc:
        raise HTTPException(400, f"未知放大器: {body.model}") from exc
    if not target.exists():
        raise HTTPException(
            409,
            f"放大器权重未下载: {body.model}（请先到「设置 → 预处理」下载）",
        )

    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        try:
            job = preprocess_svc.start_job(
                conn,
                project_id=pid,
                mode=body.mode,
                names=body.names,
                model=body.model,
                tile_size=body.tile_size,
                tile_pad=body.tile_pad,
                device=body.device,
                target_area=body.target_area,
            )
        except preprocess_svc.PreprocessError as exc:
            raise HTTPException(400, str(exc)) from exc
        # 推进 stage（不阻塞 curate；stepper 自己派生 done/active）
        p = projects.advance_stage(conn, pid, "preprocessing")
    _publish_job_state(job)
    _publish_project_state(p)
    return job


@app.get("/api/projects/{pid}/preprocess/status")
def preprocess_status(pid: int) -> dict[str, Any]:
    """返回最新 preprocess job + 日志尾 + 概要统计。"""
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise HTTPException(404, f"项目不存在: id={pid}")
        job = project_jobs.latest_for(
            conn, project_id=pid, kind=preprocess_svc.PREPROCESS_KIND
        )
    log_tail = ""
    if job:
        log_path = Path(job.get("log_path") or "")
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                log_tail = "\n".join(text.splitlines()[-50:])
            except Exception:
                log_tail = ""
    return {
        "job": job,
        "log_tail": log_tail,
        "summary": preprocess_svc.summary(p),
    }


@app.get("/api/projects/{pid}/preprocess/files")
def list_preprocess_files(pid: int) -> dict[str, Any]:
    """返回 preprocess/ 已处理产物 + download/ 里还没处理的源。"""
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    return {
        "processed": preprocess_svc.list_processed(p),
        "pending": preprocess_svc.list_pending(p),
        "summary": preprocess_svc.summary(p),
    }


@app.get("/api/projects/{pid}/preprocess/crop/workspace")
def list_crop_workspace(pid: int) -> dict[str, Any]:
    """裁剪页工作集：返回所有可裁剪的图 + 像素尺寸。

    包含两类：
    - preprocess/ 里已处理的图（origin 指 download/ 原图）
    - download/ 里未处理的图（裁剪页把"未放大"图当 1× pass-through）

    返回 `{images: [{name, source, w, h, mtime, size, processed}, ...]}`。
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    return {"images": preprocess_svc.list_crop_workspace(p)}


@app.post("/api/projects/{pid}/preprocess/crop")
def start_preprocess_crop(
    pid: int, body: PreprocessCropRequest
) -> dict[str, Any]:
    """开始裁剪 job。

    `crops`: `{源文件名: [{x,y,w,h,label?}], ...}`，每条 rect 归一化 [0..1]。
    源文件名为 preprocess/ 下当前文件名（worker 兜底 download/）。

    返回新建的 job 行。worker 切 PNG + 更新 manifest（多裁剪走 fan-out 命名
    `{stem}_c{n}.png` 并删原 `{stem}.png`）。详见 docs/design/preprocess-crop-design.md。
    """
    if not body.crops:
        raise HTTPException(400, "crops 不能为空")
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        # Pydantic 模型转成 dict 喂业务层（业务层会再做一次校验 + clamp）
        crops_payload: dict[str, list[dict[str, Any]]] = {
            name: [r.model_dump() for r in rects]
            for name, rects in body.crops.items()
        }
        try:
            job = preprocess_svc.start_crop_job(
                conn, project_id=pid, crops=crops_payload
            )
        except preprocess_svc.PreprocessError as exc:
            raise HTTPException(400, str(exc)) from exc
        p = projects.advance_stage(conn, pid, "preprocessing")
    _publish_job_state(job)
    _publish_project_state(p)
    return job


@app.post("/api/projects/{pid}/preprocess/files/restore")
def restore_preprocess_files(
    pid: int, body: PreprocessRestoreRequest
) -> dict[str, Any]:
    """还原指定产物：删 manifest entry + 删 preprocess/{name} PNG。

    还原后图回到「未处理」（隐式 original）状态。下游 resolver 重新指向
    download/{原名}。见 ADR 0004。
    """
    if not body.names:
        return {"restored": [], "missing": []}
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    try:
        res = preprocess_svc.restore_products(p, body.names)
    except preprocess_svc.PreprocessError as exc:
        raise HTTPException(400, str(exc)) from exc
    if res["restored"]:
        _publish_project_state(p)
    return res


@app.get("/api/projects/{pid}/preprocess/thumb")
def preprocess_thumb(
    pid: int, name: str = "", size: int = 256
) -> FileResponse:
    """[Deprecated] preprocess/ 目录的缩略图。

    ADR 0004 之后 `/api/projects/{pid}/thumb?bucket=download&name=<original>`
    自带 manifest resolve，前端走那个就够；此端点保留只为兼容旧 URL（仍按
    传入的 preprocess/{name} 直读，不绕 manifest）。
    """
    if "/" in name or "\\" in name or ".." in name or not name:
        raise HTTPException(400, "invalid name")
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    _, pre = preprocess_svc.project_paths(p)
    f = pre / name
    if not f.exists() or f.suffix.lower() not in datasets.IMAGE_EXTS:
        logger.info("preprocess thumb 404: pid=%s name=%s -> %s", pid, name, f)
        raise HTTPException(404)
    return _thumb_response(f, size)


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
        f = _safe_join_or_400(pdir, name)
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

    `name` 是 download/ 下的**原始文件名**。后端通过
    `preprocess_manifest.resolve()` 决定实际字节路径（见 ADR 0004）：
      - 未处理 → download/{name}
      - 已处理 → preprocess/{stem}.png（用户看到的是"升级后"的图，但 URL 不变）

    前端**不需要**知道有没有预处理过——这个端点已经吃下了差异。

    缓存路径：`studio_data/thumb_cache/{sha1(src+mtime+size)}.jpg`。
    源文件 mtime 变化会自动 invalidate（hash 变）。
    """
    if bucket != "download":
        raise HTTPException(400, "PP2 仅支持 bucket=download")
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    pdir = projects.project_dir(p["id"], p["slug"])
    # path traversal 校验（safe_join 对 download 路径，校验文件名安全性）
    _safe_join_or_400(pdir / "download", name)
    # ADR 0004：resolve 决定真路径。Multi-crop 之后一个 download 可能对应 N 张
    # preprocess 文件 — thumb 端点取第一张展示（按字典序），后续可改用 _c{n}
    # 寻址。kind 字段在新 schema 没有，所以这里也容忍 entry 没 kind。
    preprocess_manifest.ensure_manifest(pdir)
    candidates = preprocess_manifest.resolve_origin(pdir, name)
    f = candidates[0] if candidates else (pdir / "download" / name)
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


class CLTaggerOverrides(BaseModel):
    """打标页对 CLTagger 设置的「本次任务覆盖」—— 仅在 worker 进程内生效。"""
    threshold_general: Optional[float] = None
    threshold_character: Optional[float] = None
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    tag_mapping_path: Optional[str] = None
    local_dir: Optional[str] = None
    add_rating_tag: Optional[bool] = None
    add_model_tag: Optional[bool] = None
    blacklist_tags: Optional[list[str]] = None


class LLMTaggerOverrides(BaseModel):
    """打标页对 LLM tagger 设置的「本次任务覆盖」—— 仅在 worker 进程内生效。

    - `current_preset`：切换 active preset id
    - 其余字段：覆盖 active preset 的同名字段
    - `api_key` 不允许 override（避免出现在 task params/日志）
    """
    current_preset: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    prompt: Optional[str] = None
    output_format: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None
    concurrency: Optional[int] = None
    requests_per_second: Optional[float] = None
    max_requests_per_minute: Optional[int] = None
    max_side: Optional[int] = None
    jpeg_quality: Optional[int] = None
    max_image_mb: Optional[float] = None


class TagJobRequest(BaseModel):
    tagger: str = "wd14"
    output_format: str = "txt"                # "txt" | "json"
    wd14_overrides: Optional[Wd14Overrides] = None
    cltagger_overrides: Optional[CLTaggerOverrides] = None
    llm_overrides: Optional[LLMTaggerOverrides] = None
    # 触发词；空串 / None = 不启用。打标时作为第一个 tag prepend 到 caption；
    # 同时持久化到 version.trigger_word，后续 train 阶段从私有 yaml 读出。
    trigger_word: Optional[str] = None


class LLMModelsRefreshRequest(BaseModel):
    # preset_id 指定要更新的 preset；不传则用当前 current_preset
    preset_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = None


class LLMConnectionTestRequest(BaseModel):
    preset_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    timeout: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


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


# PyTorch 运行时 / 重装（PR-S2）-------------------------------------------


@app.get("/api/torch/status")
def torch_status() -> dict[str, Any]:
    """返回 torch 当前状态 + 驱动检测 + 推荐 cu tag + 误装诊断 flag。

    UI 用 `is_cpu_with_gpu` 决定是否显著提示「检测到 GPU 但装的是 CPU 版」。
    `is_cuda_build_unavailable` 标志驱动 / WSL 问题（不是 pip 能修的，UI 给文档链接）。
    """
    return torch_setup.current_status()


class TorchReinstallRequest(BaseModel):
    target: str = "auto"  # "auto" | "cu128" | "cu126" | "cu124" | "cu118" | "cpu"


@app.post("/api/torch/reinstall")
def torch_reinstall(body: TorchReinstallRequest) -> dict[str, Any]:
    """注册 torch 重装请求；下次 Studio 启动时由 launcher 进程执行。

    为什么不直接装：server 进程已 import 了 torch（flash_attention_setup 等间接拉
    上的），Windows 上 `torch\\_C.cp311-win_amd64.pyd` 被锁，pip uninstall / replace
    会撞 [WinError 5] 拒绝访问。改成写 marker → 用户 Ctrl+C 重启 → cli.py 启动
    时还没 import torch，pip 能正常替换文件。

    返回 `{pending: true, target, tag, message}`，UI 显示「请关闭并重启 Studio」。
    """
    try:
        tag = torch_setup._decide_target_tag(body.target)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    pending_install.register_torch_reinstall(body.target)
    return {
        "pending": True,
        "target": body.target,
        "tag": tag,
        "message": "重装请求已注册。请 Ctrl+C 关闭 Studio 后重新运行 studio.bat / studio.sh —— 启动时会自动安装 torch（~3 GB，5-30 分钟），然后正常起 server。",
    }


# FlashAttention runtime（PR-7b）-----------------------------------------


class FlashAttnInstallRequest(BaseModel):
    url: Optional[str] = None  # None = 自动从 GitHub Releases 选最优


@app.get("/api/flash-attention/status")
def flash_attn_status() -> dict[str, Any]:
    """返回 flash_attn 安装状态 + 当前环境检测 + GitHub 候选 wheel 列表。

    candidates 里 score / tags 等 UI 不需要的字段已剥掉，只保留 url/name/notes/usable。
    候选最多取前 20 个，避免 GitHub 历史 release 一大坨刷屏。
    fetch_error 非 None 表示 GitHub API 请求失败（限流 / 网络 / 国内防火墙）；
    UI 要展示这条让用户能选择手动粘 URL。
    """
    status = flash_attention_setup.current_status()
    env = flash_attention_setup.detect_env()
    candidates, fetch_error = flash_attention_setup.find_candidates(env)
    slim = [
        {"url": c["url"], "name": c["name"], "notes": c["notes"], "usable": c["usable"]}
        for c in candidates[:20]
    ]
    return {**status, "env": env, "candidates": slim, "fetch_error": fetch_error}


@app.post("/api/flash-attention/install")
def flash_attn_install(body: FlashAttnInstallRequest) -> dict[str, Any]:
    """安装 flash_attn wheel；url=null 走 service 的自动匹配。

    同步 pip install（远端 wheel ~150MB），可能几分钟；UI 按钮必须带 loading。
    flash_attn 是 C extension，装完必须重启 Studio 才能切换；返回 restart_required=True。
    """
    try:
        return flash_attention_setup.install(body.url)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


# xformers runtime ------------------------------------------------------


@app.get("/api/xformers/status")
def xformers_status() -> dict[str, Any]:
    """返回 xformers 安装状态。

    比 flash_attention/status 简洁很多 —— xformers 走 PyPI 直装，不需要 GitHub
    候选 wheel 列表 / 环境检测细节（status 里 installed/version 已经够用）。
    """
    return xformers_setup.current_status()


@app.post("/api/xformers/install")
def xformers_install() -> dict[str, Any]:
    """pip install xformers --index-url <torch-cu-index>。

    同步执行；远端 wheel 通常几十到几百 MB，几分钟级。装失败抛 500，message
    含 stderr 末尾（多数失败 = 上游 wheel 没覆盖当前 torch+cu 组合）。

    xformers 是 C extension，装完返回 restart_required=True 让 UI 提示重启。
    """
    try:
        return xformers_setup.install()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


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


def _select_preset(
    tagger_cfg: "secrets.LLMTaggerConfig", preset_id: Optional[str]
) -> "secrets.LLMPresetConfig":
    pid = preset_id or tagger_cfg.current_preset
    for preset in tagger_cfg.presets:
        if preset.id == pid:
            return preset
    return tagger_cfg.active


@app.post("/api/llm-tagger/models/refresh")
def refresh_llm_tagger_models(body: LLMModelsRefreshRequest) -> dict[str, Any]:
    """读取 OpenAI-compatible /models，并保存到指定 preset 的 model_ids。

    `preset_id` 不传时用 current_preset。成功后才落 secrets，避免请求失败时写脏。
    """
    from .services import llm_tagger as llm_tagger_svc

    tagger_cfg = secrets.load().llm_tagger
    target = _select_preset(tagger_cfg, body.preset_id)
    base_url = (body.base_url if body.base_url is not None else target.base_url).strip()
    api_key = (
        target.api_key
        if body.api_key is None or body.api_key == secrets.MASK
        else body.api_key.strip()
    )
    if not base_url:
        raise HTTPException(400, "base_url is required")
    try:
        model_ids = llm_tagger_svc.fetch_openai_compatible_models(
            base_url,
            api_key,
            timeout=body.timeout or target.timeout,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, str(exc)) from exc
    selected = target.model if target.model in model_ids else (model_ids[0] if model_ids else target.model)
    preset_patch: dict[str, Any] = {
        "id": target.id,
        "base_url": base_url,
        "model_ids": model_ids,
        "model": selected,
    }
    if body.api_key not in (None, secrets.MASK):
        preset_patch["api_key"] = api_key
    new = secrets.update({"llm_tagger": {"presets": [preset_patch]}})
    return {
        "items": model_ids,
        "preset_id": target.id,
        "secrets": secrets.to_masked_dict(new),
    }


@app.post("/api/llm-tagger/test")
def test_llm_tagger_connection(body: LLMConnectionTestRequest) -> dict[str, Any]:
    """Run a text-only LLM connectivity test without saving form values.

    Defaults come from the target preset (preset_id or current); body fields
    override on top.
    """
    from .services import llm_tagger as llm_tagger_svc

    tagger_cfg = secrets.load().llm_tagger
    target = _select_preset(tagger_cfg, body.preset_id)
    merged = target.model_dump()
    for key in ("base_url", "model", "endpoint", "timeout", "max_tokens", "temperature"):
        value = getattr(body, key)
        if value is not None:
            merged[key] = value
    if body.api_key is not None and body.api_key != secrets.MASK:
        merged["api_key"] = body.api_key
    cfg = secrets.LLMPresetConfig(**merged)
    if not cfg.base_url.strip():
        raise HTTPException(400, "base_url is required")
    if not cfg.model.strip():
        raise HTTPException(400, "model is required")
    return llm_tagger_svc.test_openai_compatible_connection(
        cfg.base_url,
        cfg.api_key,
        cfg.model,
        endpoint=cfg.endpoint,
        timeout=cfg.timeout,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


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

    # 触发词：先 strip，落到 version 表（持久化，TagEdit / Train 都能读），再
    # 顺手放进 worker params。body.trigger_word=None 表示前端没传字段（不改
    # version 现有值）；空串 "" 表示用户主动清空。
    trigger_word = body.trigger_word.strip() if body.trigger_word is not None else None

    params: dict[str, Any] = {
        "tagger": body.tagger,
        "version_id": vid,
        "output_format": body.output_format,
    }
    if trigger_word:
        params["trigger_word"] = trigger_word
    # 通用：按 tagger 名取 `<name>_overrides` 字段并落到 params 同名键。
    # 仅保留用户实际填写的字段；空 dict 也不写。
    overrides_field = getattr(body, f"{body.tagger}_overrides", None)
    if overrides_field is not None:
        ov = overrides_field.model_dump(exclude_none=True)
        if ov:
            params[f"{body.tagger}_overrides"] = ov

    with db.connection_for() as conn:
        if trigger_word is not None and trigger_word != (v.get("trigger_word") or ""):
            updated = versions.update_version(conn, vid, trigger_word=trigger_word)
            _publish_version_state(updated)
            v = updated
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
    _safe_join_or_400(train, folder)
    return {
        "folder": folder,
        "items": tagedit.list_captions_in_folder(train, folder, full=full),
    }


@app.get("/api/projects/{pid}/versions/{vid}/captions/{folder}/{filename}")
def get_caption_endpoint(
    pid: int, vid: int, folder: str, filename: str
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    _safe_join_or_400(train, folder, filename)
    try:
        return tagedit.read_one(train, folder, filename)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.put("/api/projects/{pid}/versions/{vid}/captions/{folder}/{filename}")
def put_caption_endpoint(
    pid: int, vid: int, folder: str, filename: str, body: CaptionEdit
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    _safe_join_or_400(train, folder, filename)
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
        try:
            img = safe_join(train, it.folder, it.name)
        except ValueError:
            skipped.append(f"{it.folder}/{it.name}")
            continue
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
    if not path:
        raise HTTPException(400, "invalid path")
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    # path 允许含 `/` 子目录；按分隔符拆成片段交给 safe_join 做组件校验 + containment
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    img = _safe_join_or_400(rdir, *parts)
    if not img.exists() or img.suffix.lower() not in datasets.IMAGE_EXTS:
        raise HTTPException(404, "image not found")
    return {"path": path, "tags": tagedit.read_tags(img)}


def _resolve_anima_model_paths() -> dict[str, str]:
    """解析 base 模型默认路径（先验生成 / 测试出图共用）。

    与 version_config 的 model 字段对齐。用户用别的 base 模型时，
    在 Settings → 模型 里改 selected_anima 影响这里的 anima 主权重路径。
    """
    from .services.model_downloader import models_root
    root = models_root()
    return {
        "transformer_path": str(root / "diffusion_models" / "anima-base-v1.0.safetensors"),
        "vae_path": str(root / "vae" / "qwen_image_vae.safetensors"),
        "text_encoder_path": str(root / "text_encoders"),
        "t5_tokenizer_path": str(root / "t5_tokenizer"),
    }


class RegAiRequest(BaseModel):
    """先验生成请求 —— 不含 lora_configs，先验生成不带 LoRA。"""
    excluded_tags: list[str] = []
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 4.0
    sampler_name: str = "er_sde"
    scheduler: str = "simple"
    seed: int = 0
    incremental: bool = False
    mixed_precision: str = "bf16"
    attention_backend: AttentionBackend = "flash_attn"

    # 兼容老前端送 xformers / flash_attn 双 bool（自动映射成 attention_backend）
    @model_validator(mode="before")
    @classmethod
    def _migrate_attention(cls, data: Any) -> Any:
        return migrate_legacy_attention(data)


@app.post("/api/projects/{pid}/versions/{vid}/reg/generate-prior")
def reg_generate_prior(pid: int, vid: int, body: RegAiRequest) -> dict[str, Any]:
    """启动先验生成 task —— base 模型给每张 train 图的 tag 反向出对照图。"""
    model_paths = _resolve_anima_model_paths()
    _, _, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    has_image = train.exists() and any(
        f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
        for f in train.rglob("*")
    )
    if not has_image:
        raise HTTPException(400, "train 还没有图片，请先完成 Step 1（下载）或 Step 2（筛选）")

    rdir = _reg_dir(vdir)
    rdir.mkdir(parents=True, exist_ok=True)

    cfg = RegAiConfig(
        **model_paths,
        train_dir=str(train),
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
        incremental=body.incremental,
        mixed_precision=body.mixed_precision,
        attention_backend=body.attention_backend,
    )

    cfg_dir = STUDIO_DATA / "reg_ai_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name=f"reg-prior p{pid}v{vid}", config_name="reg_ai", priority=0,
        )
        db.update_task(
            conn, task_id, task_type="reg_ai", project_id=pid, version_id=vid,
        )

    cfg_path = cfg_dir / f"reg_ai_{task_id}.json"
    cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    with db.connection_for() as conn:
        db.update_task(conn, task_id, config_path=str(cfg_path))
        task = db.get_task(conn, task_id)

    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    return task or {"id": task_id}


@app.get("/api/projects/{pid}/versions/{vid}/reg/generate-prior/{task_id}")
def get_reg_prior_task(pid: int, vid: int, task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task or task.get("task_type") != "reg_ai":
        raise HTTPException(404)
    return task


# ---------------------------------------------------------------------------
# /api/generate — 测试出图（独立工具页，多 LoRA + multi-prompt）
# ---------------------------------------------------------------------------
#
# 用户决策："测试" 出图不持久化（commit 10 起完全去磁盘）：
#   - daemon 把 PNG bytes base64 推回 server 入 generate_cache（内存 dict）
#   - HTTP `/api/generate/{tid}/sample/{fn}` 从 cache 取
#   - tempdir 仅装 config.json（小 JSON）；task 结束 supervisor 仍调
#     cleanup_generate_tempdir 清掉空目录 + config.json
#   - server 重启 → 内存 cache 自动没；强杀也不残留


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
    lora_configs: list[LoraEntry] = []
    mixed_precision: str = "bf16"
    # commit C：attention_backend 默认从 secrets.generate.attention_backend 读，
    # 前端 Generate 页不再发这个字段；保留 Optional 兼容老客户端 / 临时覆盖。
    attention_backend: Optional[AttentionBackend] = None
    # XY 矩阵：None=单图模式；设值时 schema 强制 prompts 单条 + count=1
    xy_matrix: Optional[XYMatrixSpec] = None

    # 兼容老前端送 xformers / flash_attn 双 bool（自动映射成 attention_backend）
    @model_validator(mode="before")
    @classmethod
    def _migrate_attention(cls, data: Any) -> Any:
        return migrate_legacy_attention(data)


@app.post("/api/generate")
def enqueue_generate(body: GenerateRequest) -> dict[str, Any]:
    """启动测试出图 task。"""
    from .services.inference_core import generate_tempdir

    model_paths = _resolve_anima_model_paths()

    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name="generate", config_name="generate", priority=0,
        )
        db.update_task(conn, task_id, task_type="generate")

    tempdir = generate_tempdir(task_id)
    tempdir.mkdir(parents=True, exist_ok=True)

    # attention_backend：secrets 读默认；body 给值则覆盖（兼容旧客户端）
    # secrets 默认 'auto' → 调 detect_attention_backend 按"装了什么用什么"决定
    try:
        gen_cfg = secrets.load().generate
        attn_default = gen_cfg.attention_backend
        preview_n = int(gen_cfg.preview_every_n_steps or 0)
    except Exception:
        attn_default = "auto"
        preview_n = 0
    attn = body.attention_backend or attn_default
    if attn == "auto":
        from .services.xformers_setup import detect_attention_backend
        attn = detect_attention_backend()

    cfg = GenerateConfig(
        **model_paths,
        output_dir=str(tempdir),
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
        attention_backend=attn,
        xy_matrix=body.xy_matrix.model_dump() if body.xy_matrix else None,
    )

    # commit 14：注入 daemon 端用的 preview 节流参数（settings 全局开关）
    cfg_dict = cfg.model_dump()
    cfg_dict["preview_every_n_steps"] = preview_n

    cfg_path = tempdir / "config.json"
    cfg_path.write_text(json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    with db.connection_for() as conn:
        db.update_task(conn, task_id, config_path=str(cfg_path))
        task = db.get_task(conn, task_id)

    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    return task or {"id": task_id}


@app.get("/api/generate/{task_id}")
def get_generate_task(task_id: int) -> dict[str, Any]:
    """查询测试 task 状态。"""
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task or task.get("task_type") != "generate":
        raise HTTPException(404)
    return task


# ---------------------------------------------------------------------------
# /api/generate/daemon — 测试 daemon 状态查询 + 手动卸载（commit 13）
# ---------------------------------------------------------------------------


@app.get("/api/generate/taeflux/status")
def get_taeflux_status() -> dict[str, Any]:
    """commit 14：查询 TAEFlux 模型是否就绪（中间步预览依赖）。"""
    from .services import model_downloader as _md
    d = _md.taeflux_dir()
    return {
        "available": _md.taeflux_available(),
        "dir": str(d),
        "files": _md.TAEFLUX_FILES,
    }


@app.post("/api/generate/taeflux/install")
def install_taeflux() -> dict[str, Any]:
    """同步下载 TAEFlux（~1.6MB，秒级）。已存在直接返回 OK。"""
    from .services import model_downloader as _md
    if _md.taeflux_available():
        return {"ok": True, "noop": True}
    ok = _md.download_taeflux()
    if not ok:
        raise HTTPException(500, "download failed; check server log")
    return {"ok": True}


@app.get("/api/generate/daemon/status")
def get_daemon_status() -> dict[str, Any]:
    """查询 daemon 当前状态。前端 DaemonControls 用。"""
    from .services.inference_daemon import get_daemon
    daemon = get_daemon()
    return {
        "state": daemon.state,
        "model_loaded": daemon.is_model_loaded,
        "busy": daemon.is_busy,
        "alive": daemon.is_alive,
    }


@app.post("/api/generate/daemon/unload")
def unload_daemon() -> dict[str, Any]:
    """手动卸载 daemon 模型（释放 VRAM）。busy 时拒绝（409）。

    卸载完成后 supervisor 会推 daemon_state_changed SSE，前端按钮自动 disable。
    下次用户点「开始生成」daemon 按需重 load。
    """
    from .services.inference_daemon import get_daemon
    daemon = get_daemon()
    if daemon.is_busy:
        raise HTTPException(409, "daemon is busy, cannot unload")
    if not daemon.is_model_loaded:
        return {"ok": True, "noop": True}
    daemon.request_unload()
    return {"ok": True}


@app.get("/api/generate/{task_id}/sample/{filename}")
def get_generate_sample(task_id: int, filename: str) -> Any:
    """读 generate task 的输出图（commit 10：从 server 内存 cache 取，无磁盘）。

    daemon 出图完成后把 PNG bytes 推回 server 入 generate_cache；HTTP 这里
    直接返回 bytes。LRU / 客户端断连清理在 commit 11 加 —— 在那之前 cache
    跟着 supervisor finalize 释放（一 task 一组 entry，task 终止时全清）。
    """
    _validate_component_or_400(filename)
    if not filename.lower().endswith(".png"):
        raise HTTPException(400, "only .png supported")
    from fastapi.responses import Response
    from .services import generate_cache
    data = generate_cache.get_image(task_id, filename)
    if data is None:
        raise HTTPException(404)
    # 用 no-store 不是 _thumb_response 那套 no-cache + ETag：
    # generate cache 同 (task_id, filename) 内容会随重跑覆盖（用户改 prompt 重生成），
    # 没有稳定 ETag 可发；用 no-store 让浏览器每次都重拉，永远拿到最新结果。
    # 带宽代价小：用户在测试出图页主动看才命中本 endpoint，QPS 低。
    # （Thumbnail / dataset 那种内容稳定的图，继续用 _thumb_response 的 ETag。）
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


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
    """读 version 私有 config；不存在返回 has_config=false / config=null。

    无论 has_config 与否都返回 `project_specific_defaults` —— fork preset 时
    后端将自动注入的项目预填值（项目路径 + 全局模型路径 + reg 检测结果）。
    前端「+ 新建预设」可以在 version 已有 config 的状态下被点（替换当前预设），
    所以这个 hint 跟 has_config 状态无关，永远要返回。
    """
    project, ver = _project_and_version_or_404(pid, vid)
    psf = sorted(version_config.PROJECT_SPECIFIC_FIELDS)
    psd = {
        **version_config.project_specific_overrides(project, ver),
        **model_downloader.default_paths_for_new_version(),
    }
    if not version_config.has_version_config(project, ver):
        return {
            "has_config": False,
            "config": None,
            "project_specific_fields": psf,
            "project_specific_defaults": psd,
        }
    try:
        cfg = version_config.read_version_config(project, ver)
    except version_config.VersionConfigError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "has_config": True,
        "config": cfg,
        "project_specific_fields": psf,
        "project_specific_defaults": psd,
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
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        p = projects.get_project(conn, pid)
    if not v or not p or v["project_id"] != pid:
        raise HTTPException(404, "版本不存在")
    vdir = versions.version_dir(p["id"], p["slug"], v["label"]) / bucket
    if bucket in {"train", "reg"}:
        if not folder:
            raise HTTPException(400, "invalid folder")
        f = _safe_join_or_400(vdir, folder, name)
    else:
        f = _safe_join_or_400(vdir, name)
    if not f.exists() or f.suffix.lower() not in datasets.IMAGE_EXTS:
        logger.info(
            "version thumb 404: pid=%s vid=%s bucket=%s folder=%s name=%s -> %s",
            pid, vid, bucket, folder, name, f,
        )
        raise HTTPException(404)
    return _thumb_response(f, size)


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
def export_queue(ids: str = "") -> Response:
    """`?ids=1,2,3` 指定导出的任务，缺省导出全部。

    响应带 `Content-Disposition: attachment` —— 前端 <a download> 直链就能触发
    浏览器原生下载（和 train.zip / outputs.zip 一套范式）。导出/失败 publish
    queue_export_ready / _failed SSE，前端用来清 app-side spinner + 弹 toast。
    body 仍是合法 JSON，tests / 程序化调用方拿 resp.json() 不受影响。
    """
    import json as _json

    if ids.strip():
        try:
            id_list = [int(x) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "ids must be comma-separated integers")
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


@app.post("/api/queue/import")
def import_queue(body: ImportRequest) -> dict[str, Any]:
    try:
        return queue_io.import_tasks(body.payload)
    except (ValueError, presets_io.PresetError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/queue")
def list_queue(
    status: Optional[str] = None,
    include_generate: bool = False,
) -> dict[str, Any]:
    """队列默认隐藏 generate 测试出图任务（commit 15 P0-2）。

    generate task 走 daemon 不占 train slot，且生命周期短（出完图就结束），
    出现在队列里只会让用户混淆"为什么队列卡住"。需要排查时加
    `?include_generate=true` 兜底。
    """
    if status and status not in db.VALID_STATUSES:
        raise HTTPException(400, f"unknown status: {status}")
    with db.connection_for() as conn:
        items = db.list_tasks(conn, status=status)
    if not include_generate:
        items = db.filter_out_task_types(items, ("generate",))
    # ADR 0006 PR-4 — is_pausable 信号每行注入（§8.1 / 上面 get_queue_item 注释）
    try:
        sup = _supervisor()
        for it in items:
            it["is_pausable"] = sup.is_task_pausable(int(it["id"]))
    except HTTPException:
        for it in items:
            it["is_pausable"] = False
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
    # ADR 0006 PR-4 — is_pausable 信号让 UI 决定是否显示暂停按钮（§8.1）。
    # 仅 supervisor 跑得起来时计算；空载（test / 启动期）默认 False。
    try:
        task["is_pausable"] = _supervisor().is_task_pausable(task_id)
    except HTTPException:
        task["is_pausable"] = False
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


# ---------------------------------------------------------------------------
# ADR 0006 — pause / resume / hold endpoints。
# ---------------------------------------------------------------------------


@app.post("/api/queue/{task_id}/pause")
def pause_task(task_id: int) -> dict[str, Any]:
    """暂停 running task（ADR §4.1 / §4.3）。

    异步：立即返回；UI 端 modal 订阅 SSE 看保存进度。supervisor 收到子进程
    `__EVENT__:pause_state` 后把 status 写为 paused 并 publish task_state_changed。
    """
    ok, reason = _supervisor().pause(task_id)
    if not ok:
        # 区分客户端错误（404/409）vs 状态机不允许（409）
        with db.connection_for() as conn:
            task = db.get_task(conn, task_id)
        if not task:
            raise HTTPException(404, "task not found")
        raise HTTPException(409, reason or "pause rejected")
    return {"task_id": task_id, "pause_pending": True}


@app.get("/api/queue/hold")
def get_queue_hold() -> dict[str, Any]:
    """查看当前队列挂起状态 + 等待恢复调度的 pending task 数（UI banner 用）。"""
    with db.connection_for() as conn:
        held = db.get_queue_held(conn)
        pending = db.list_tasks(conn, status="pending")
    return {"held": held, "pending_waiting": len(pending)}


@app.post("/api/queue/hold")
def hold_queue() -> dict[str, Any]:
    """挂起队列：dispatcher 不再拉新 task。已 running 的不受影响（ADR §3.2）。

    "同时暂停 running task" 由前端 modal 拆成两步：先调本 endpoint，再
    单独调 `/api/queue/{id}/pause`。后端不做合一操作。
    """
    with db.connection_for() as conn:
        db.set_queue_held(conn, True)
    bus.publish({"type": "queue_hold_changed", "held": True})
    return {"held": True}


@app.post("/api/queue/release")
def release_queue() -> dict[str, Any]:
    """恢复调度：dispatcher 重新按 priority + created_at 拉 pending。"""
    with db.connection_for() as conn:
        db.set_queue_held(conn, False)
    bus.publish({"type": "queue_hold_changed", "held": False})
    return {"held": False}


@app.post("/api/queue/{task_id}/resume")
def resume_task(task_id: int) -> dict[str, Any]:
    """恢复 paused task（ADR 0006 §6 路径 A）。

    流程：
      1. 校验 status == 'paused' + paused_state_path 文件存在
      2. task → pending（**保留 paused_* 字段**，cmd_builder 下轮 dispatch 读它）
      3. supervisor 下次 _tick 自然 pick up，cmd 加 `--resume-state <pt>`，
         bootstrap_phase 读 sibling .config.json snapshot 覆盖 args
      4. 子进程 load_training_state 成功后 emit `resume_state_loaded` →
         supervisor `_clear_pause_artifacts` 清文件对 + db 字段

    文件丢失 → 409（ADR §5.5：引导用户走 ResumeFieldPicker 起新 task）。
    """
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
        if not task:
            raise HTTPException(404, "task not found")
        if task["status"] != "paused":
            raise HTTPException(
                409, f"task status is {task['status']!r}, not paused"
            )
        state_path = task.get("paused_state_path")
        config_path = task.get("paused_config_path")
        if not state_path or not Path(state_path).exists():
            raise HTTPException(
                409,
                f"paused state file missing: {state_path!r}; "
                f"use ResumeFieldPicker to start a fresh task from another .pt",
            )
        if config_path and not Path(config_path).exists():
            # snapshot 缺失虽然不致命（bootstrap_phase 会沿用 args.config yaml），
            # 但 resume 语义会漂；按 ADR §5.7 严格 freeze 原则，拒绝继续。
            raise HTTPException(
                409,
                f"paused config snapshot missing: {config_path!r}; "
                f"cannot guarantee config freeze, refusing to resume",
            )
        db.update_task(
            conn, task_id,
            status="pending",
            started_at=None,
            finished_at=None,
            exit_code=None,
            error_msg=None,
        )
    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    return {"task_id": task_id, "status": "pending"}


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


def _task_archive_basename(task: dict[str, Any]) -> Optional[str]:
    """task 关联 project / version → "{slug}-{label}"，用作 outputs zip 文件名
    前缀。和 train.zip 命名风格一致（PP7：{slug}-{label}.train.zip）。

    没 project / version → None，调用方 fallback 到 task_{id}。
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
    return f"{p['slug']}-{v['label']}"


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
        "archive_basename": _task_archive_basename(task),
    }


@app.get("/api/queue/{task_id}/outputs.zip")
def download_task_outputs_zip(
    task_id: int,
    background: BackgroundTasks,
    files: Optional[str] = None,
) -> FileResponse:
    """把 output 目录里的文件打包成 zip 一次性下载。

    没传 `files` → 全量打包（向后兼容）。
    传了 `files`（逗号分隔的文件名）→ 仅打包这些，路径必须是 out_dir 下的直接
    子文件，禁止 path traversal / 子目录。

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
    all_files = [f for f in sorted(out_dir.iterdir()) if f.is_file()]
    if not all_files:
        raise HTTPException(404, "empty output dir")

    selected: list[Path]
    partial = False
    if files is None or files == "":
        selected = all_files
    else:
        wanted = [n for n in files.split(",") if n]
        if not wanted:
            raise HTTPException(400, "empty files list")
        by_name = {f.name: f for f in all_files}
        missing: list[str] = []
        selected = []
        for name in wanted:
            _validate_component_or_400(name)
            f = by_name.get(name)
            if not f:
                missing.append(name)
                continue
            selected.append(f)
        if missing:
            raise HTTPException(404, f"file(s) not found: {', '.join(missing)}")
        partial = True

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(
            tmp_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zf:
            for f in selected:
                zf.write(f, arcname=f.name)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        bus.publish({
            "type": "task_outputs_zip_failed",
            "task_id": task_id,
            "error": str(e),
        })
        raise

    bus.publish({"type": "task_outputs_zip_ready", "task_id": task_id})
    background.add_task(lambda: tmp_path.unlink(missing_ok=True))

    basename = _task_archive_basename(task) or f"task_{task_id}"
    archive_name = (
        f"{basename}_outputs_selected.zip" if partial
        else f"{basename}_outputs.zip"
    )
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
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    out_dir = _task_output_dir(task)
    if not out_dir or not out_dir.exists():
        raise HTTPException(404, "no output dir")
    f = _safe_join_or_400(out_dir, filename)
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
    """目录浏览（给前端 path picker 用）。缺省 = REPO_ROOT。

    PathPicker 设计本就是给用户选外部模型路径用的（云端机器把模型放数据盘），
    所以这里 allow_outside_repo=True；安全边界在 list_dir 本身（只读 entries
    名字+类型，不返回内容）。
    """
    target = Path(path) if path else REPO_ROOT
    if not target.is_absolute():
        target = (REPO_ROOT / target).resolve()
    try:
        return browse.list_dir(target, allow_outside_repo=True)
    except browse.BrowseError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/datasets/thumbnail")
def get_dataset_thumbnail(folder: str, name: str) -> FileResponse:
    """返回 dataset 缩略图（实际是原图，前端用 CSS 缩放）。

    `folder` 可以是绝对路径或相对路径（用户在 dataset 浏览器里点出来的），
    `name` 必须是单一文件名（不含分隔符）。最终路径必须在 REPO_ROOT 内。
    """
    _validate_component_or_400(name)
    p = (Path(folder) / name).resolve()
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
    _validate_component_or_400(filename)

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


# ---------------------------------------------------------------------------
# /api/system — 进程生命周期（重启 / 更新 / 回滚）
# ---------------------------------------------------------------------------
#
# 重启协议（参见 docs/adr/0002-webui-self-update.md）：
#   1. server 写 REPO_ROOT/tmp/restart 标志
#   2. server 通过 BackgroundTask 在响应发出后给自己发 SIGINT
#   3. uvicorn 捕获 SIGINT 走 graceful shutdown（lifespan teardown + 在飞请求收尾）
#   4. 进程退出 → cli.py 的 subprocess.call 返回
#   5. cli.py 检测到 tmp/restart 存在 → 删除标志 → loop 回去重新 bootstrap + 起 server
#
# 跨平台 SIGINT：用 signal.raise_signal(SIGINT)（Python 3.8+），它在 Windows /
# POSIX 都把当前进程置为收到 SIGINT，uvicorn 的内置 handler 会按 graceful
# 路径处理。os.kill(getpid, SIGINT) 在 Windows 上不工作。
#
# PR-A 仅实现 /restart（不带 git pull / 不检查 running task）。PR-B / PR-C
# 在此基础上叠加 update / rollback / 任务保护。


_RESTART_FLAG = REPO_ROOT / "tmp" / "restart"


_SHUTDOWN_FORCE_EXIT_TIMEOUT = 5.0


def _raise_sigint_after_response() -> None:
    """在响应已经发完后给自己发 SIGINT，触发 uvicorn graceful shutdown。

    BackgroundTask 在 starlette 路径上是 response 完成后调度的；这里再 sleep
    一点点保险（防止某些代理 / keep-alive 情况下还有数据没冲走）。

    Force-exit 兜底（PR-D fix）：`/api/events` 是长 SSE，generator 内的
    `asyncio.wait_for(queue.get(), 15)` 不响应 uvicorn 关停信号，graceful
    shutdown 会等 client 主动断开 → 表现为「后端卡在 waiting for
    connection to close」，用户必须刷页让浏览器关 SSE 才能继续。给 graceful
    5 秒窗口后强退（正常 in-flight 1-2s 收尾够用）。BackgroundTask 跑在
    threadpool，graceful 成功路径主进程退出会带走此线程，os._exit 不会
    触达；只有 graceful 卡住时才真正强退。
    """
    import signal
    time.sleep(0.3)
    try:
        signal.raise_signal(signal.SIGINT)
    except Exception:
        # 兜底：raise_signal 抛错（极少见）→ 直接强退
        os._exit(0)
        return
    time.sleep(_SHUTDOWN_FORCE_EXIT_TIMEOUT)
    os._exit(0)


def _check_no_running_tasks() -> None:
    """重启 / 更新前置：所有 task 必须 done / failed / canceled / pending。

    有 running 直接 422 + task 列表，让前端给用户友好的提示（"先暂停以下任务"）。
    """
    with db.connection_for() as conn:
        running = db.list_tasks(conn, status="running")
    if running:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "running_tasks_present",
                "message": "有任务正在运行，请先取消 / 等待完成",
                "tasks": [
                    {
                        "id": t["id"],
                        "name": t.get("name", ""),
                        "task_type": t.get("task_type", "train"),
                    }
                    for t in running
                ],
            },
        )


@app.post("/api/system/restart")
def system_restart(background: BackgroundTasks) -> dict[str, Any]:
    """重启 server（不 pull 代码）。

    流程：写 tmp/restart 标志 → 响应 200 → BackgroundTask 发 SIGINT 触发
    uvicorn graceful shutdown → cli.py loop 拾起 → 重新起新 server。

    PR-B 起加 running task 强制约束。
    """
    _check_no_running_tasks()
    _RESTART_FLAG.parent.mkdir(parents=True, exist_ok=True)
    _RESTART_FLAG.touch()
    background.add_task(_raise_sigint_after_response)
    return {"ok": True, "message": "restart scheduled"}


@app.get("/api/system/version")
def system_version() -> dict[str, Any]:
    """当前仓库状态：__version__ / commit / tag / branch / dirty。"""
    from dataclasses import asdict
    return asdict(updater.current_version())


@app.get("/api/system/update_check")
def system_update_check(channel: str = "master", force: bool = False) -> dict[str, Any]:
    """git fetch + 比对。master 通道用 24h cache（force=true 强制重 fetch）；
    dev 通道每次都 fetch，不缓存（开发者主动触发，避免污染 master 信号）。
    """
    from dataclasses import asdict
    if channel not in ("master", "dev"):
        raise HTTPException(400, f"invalid channel: {channel}")
    return asdict(updater.check_update(channel=channel, use_cache=not force))


class UpdateRequest(BaseModel):
    target: str = "origin/master"  # ref / commit sha / origin/branch


@app.post("/api/system/update")
def system_update(body: UpdateRequest, background: BackgroundTasks) -> dict[str, Any]:
    """请求 update：precondition 校验 + 写 .update_pending + 触发重启。

    实际 git pull 在 cli.py 启动期 updater.apply_pending() 完成（避免在 server
    进程里跑 git pull，规避 native module 已锁的问题）。
    """
    _check_no_running_tasks()

    cur = updater.current_version()
    if cur.is_dirty:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "dirty_working_tree",
                "message": "本地有未提交的修改，请先 commit / stash",
            },
        )

    updater.request_update(body.target)
    background.add_task(_raise_sigint_after_response)
    return {"ok": True, "message": f"update scheduled → {body.target}"}


@app.post("/api/system/rollback")
def system_rollback(background: BackgroundTasks) -> dict[str, Any]:
    """回滚到 .last_version 记录的上一版本（PR-C）。

    走与正向 update 完全一致的路径（写 .update_pending=<sha> + tmp/restart
    → cli.py 启动期 apply_pending 实际 reset），所以 dirty / running task
    precondition 一样适用，回滚成功后 .last_version 会被写成"回滚前的版本"
    （即正向)，支持来回切。
    """
    _check_no_running_tasks()

    cur = updater.current_version()
    if cur.is_dirty:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "dirty_working_tree",
                "message": "本地有未提交的修改，请先 commit / stash",
            },
        )

    target = updater.request_rollback()
    if target is None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "no_rollback_target",
                "message": ".last_version 不存在或 commit 已不在仓库里（被 GC？）",
            },
        )

    background.add_task(_raise_sigint_after_response)
    return {"ok": True, "message": f"rollback scheduled → {target[:8]}", "target": target}


@app.get("/api/system/update_status")
def system_update_status() -> dict[str, Any]:
    """最近一次 update 的结构化结果 + rollback target（PR-C）。

    rollback_target 与 status 解耦：即使从未走过 update（.update_status 不存在），
    只要 .last_version 指向的 commit 还在仓库里，回滚按钮就应当能用（user 手动
    git reset 后想"还原到上一版"也是合法场景）。

    UI 上：
    - status=null：没有 update 历史，不展示 banner / 不展示"查看上次日志"按钮
    - status='ok'：可选展示"已更新到 X，X 秒前"
    - status='aborted' / 'failed' / 'partial'：红色 banner + reason + 跳日志
    - rollback_target 非 null（不依赖 status）：显示"切换到 sha"按钮
    """
    from dataclasses import asdict
    rollback_to = updater.rollback_target()
    rollback_tag = updater.exact_tag_for(rollback_to) if rollback_to else None
    st = updater.last_status()
    if st is None:
        return {
            "status": None,
            "rollback_target": rollback_to,
            "rollback_target_tag": rollback_tag,
        }
    return {
        **asdict(st),
        "rollback_target": rollback_to,
        "rollback_target_tag": rollback_tag,
    }


@app.get("/api/system/update_log")
def system_update_log() -> dict[str, Any]:
    """完整 .update_log 文本内容（PR-C 失败时 UI 弹 modal 用）。"""
    return {"content": updater.read_update_log()}


@app.get("/api/system/preflight")
def system_preflight(target: str = "origin/master") -> dict[str, Any]:
    """更新前置检查（chunk 4）— VersionSection preview 状态展开时拉取。

    返回 4 项结构化检查 + target_resolved sha + requirements.txt diff 摘要。
    每项含 level (ok / warn / err)；任一 err → blocking=true，前端禁用
    确认按钮。target 接受任意 git ref（tag / branch / commit sha）。
    """
    cur = updater.current_version()

    with db.connection_for() as conn:
        running = db.list_tasks(conn, status="running")

    target_resolved = updater.resolve_ref(target)
    req_diff = updater.requirements_diff(target) if target_resolved else updater.RequirementsDiff()
    req_total = len(req_diff.added) + len(req_diff.removed) + len(req_diff.changed)

    checks: list[dict[str, str]] = []

    if cur.is_dirty:
        checks.append({"key": "dirty", "level": "err",
                       "label": "工作树有未提交修改 — 操作会被拒绝"})
    else:
        checks.append({"key": "dirty", "level": "ok",
                       "label": "工作树干净 · 无未提交改动"})

    if running:
        names = ", ".join((t.get("name") or f"#{t['id']}") for t in running[:3])
        more = f" + 还有 {len(running) - 3}" if len(running) > 3 else ""
        checks.append({"key": "running_tasks", "level": "err",
                       "label": f"{len(running)} 个任务正在运行：{names}{more}"})
    else:
        checks.append({"key": "running_tasks", "level": "ok",
                       "label": "当前 0 个训练 / 打标任务运行中"})

    if not target_resolved:
        checks.append({"key": "requirements_diff", "level": "err",
                       "label": f"target ref 解析失败：{target}"})
    elif req_total > 0:
        parts = []
        if req_diff.added:    parts.append(f"+{len(req_diff.added)}")
        if req_diff.removed:  parts.append(f"-{len(req_diff.removed)}")
        if req_diff.changed:  parts.append(f"~{len(req_diff.changed)}")
        checks.append({"key": "requirements_diff", "level": "warn",
                       "label": f"requirements.txt 变化 · {' / '.join(parts)} 包 · 预计 pip install 1-2 分钟"})
    else:
        checks.append({"key": "requirements_diff", "level": "ok",
                       "label": "requirements.txt 未变化 · 跳过 pip install"})

    checks.append({"key": "last_version", "level": "ok",
                   "label": f"更新后 .last_version = {cur.commit_short}（可一键切回）"})

    # Safety net：目标 ref 早于 self-update feature 引入 → 切过去就丢失 webui
    # 升级能力（只能 CLI git pull 救援）。err 级别阻断，前端 confirm 自动 disable。
    if target_resolved and not updater.target_has_self_update(target):
        checks.append({"key": "self_update_compat", "level": "err",
                       "label": "目标版本早于 webui 自更新 feature — 切过去后只能 CLI / shell 升级（webui 无救援能力）"})

    blocking = any(c["level"] == "err" for c in checks)

    return {
        "target": target,
        "target_resolved": target_resolved,
        "checks": checks,
        "blocking": blocking,
        "requirements_diff": {
            "added": req_diff.added,
            "removed": req_diff.removed,
            "changed": req_diff.changed,
        },
    }


@app.get("/api/system/dev_commits")
def system_dev_commits(limit: int = 10) -> dict[str, Any]:
    """`git log origin/dev -N` 摘要（chunk 3）— VersionSection dev 卡时间线用。

    每次拉 git fetch + log；fetch 失败仍尝试用本地 origin/dev 缓存（带
    error 文案）。limit clamp 1-50。
    """
    from dataclasses import asdict
    result = updater.dev_commits(limit=limit)
    return {
        "commits": [asdict(c) for c in result.commits],
        "fetched": result.fetched,
        "error": result.error,
    }


@app.post("/api/system/init_git")
def system_init_git() -> dict[str, Any]:
    """zip 解压用户一键初始化 git 仓库（0.8.1 hotfix）。

    幂等：调用前 / 调用后都跑 `git_repo_status()`，如已是仓库直接返 ok=true。
    流程见 `updater.bootstrap_git_repo()`：init + remote add origin + fetch master
    + reset --mixed 到对应 release tag。

    失败状态码：
    - 500 + error 字符串：git binary 缺失 / fetch 网络问题 / 磁盘问题
    """
    from dataclasses import asdict
    pre = updater.git_repo_status()
    if pre.is_repo:
        return {"ok": True, "already_initialized": True}

    result = updater.bootstrap_git_repo()
    if not result.ok:
        raise HTTPException(
            status_code=500,
            detail={"error": "bootstrap_failed", "message": result.error or "未知错误"},
        )

    return {"ok": True, "already_initialized": False, **asdict(result)}


@app.get("/api/system/release_notes")
def system_release_notes(tag: str) -> dict[str, Any]:
    """读 release_notes.yaml，返回指定 tag 的结构化 release notes。

    数据模型见 docs/release-notes-spec.md。tag 接受 `v0.6.0` 或 `0.6.0`。
    yaml 缺该 tag → found=false，UI 退化到 CHANGELOG.md 链接占位。
    """
    from dataclasses import asdict
    result = release_notes_svc.parse(tag)
    return {
        "tag": result.tag,
        "found": result.found,
        "date": result.date,
        "summary": result.summary,
        "entries": [asdict(e) for e in result.entries],
    }


@app.get("/", response_model=None)
def root() -> RedirectResponse | JSONResponse:
    """根路径 302 跳转到 React 应用 `/studio/`。

    若前端尚未构建（dist 缺失），返回 JSON 提示。"""
    if WEB_DIST.exists():
        return RedirectResponse(url="/studio/", status_code=302)
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
