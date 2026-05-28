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
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import (
    BackgroundTasks,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response, StreamingResponse
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
    versions_phase,
)
from .api.app import app
from .api.errors import (
    _data_export_path,
    _export_result,
    _preset_err_code as _err_code,
    _safe_join_or_400,
    _unique_data_export_path,
    _validate_component_or_400,
)
from .api.responses import EMPTY_STATE, _thumb_response
from .api.static import SPAStaticFiles
from .event_bus import bus
from .services import (
    caption_snapshot,
    downloader,
    duplicate_finder,
    presets as preset_flow,
    model_downloader,
    preprocess_manifest,
    reg_builder,
    tagedit,
    train_io,
    uploads as uploads_svc,
    version_config,
)
from .services.tagger import VALID_TAGGER_NAMES
from .paths import (
    DATA_EXPORTS,
    LOGS_DIR,
    OUTPUT_DIR,  # noqa: F401  test fixture monkeypatch path 兼容（samples router 用真位置）
    REPO_ROOT,
    STUDIO_DATA,
    STUDIO_DB,
    USER_PRESETS_DIR,
    WEB_DIST,
    safe_join,
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

logger = logging.getLogger(__name__)


# health / system stats / state, presets CRUD + import/export, /api/schema,
# /api/configs/* 308 redirects 已 PR-5 commit 2 抽到 api/routers/{health,presets}.py。
# 仍需 server.py 内的 _err_code helper 给其它 router 用？无 —— 仅 presets 内部用。
# DuplicateRequest / PresetXxxBody pydantic 模型也已抽到 api/schemas/presets.py。


# ---------------------------------------------------------------------------
# /api/secrets  (PP0 全局凭证 / 服务配置)
# ---------------------------------------------------------------------------


# /api/secrets, /api/models/*, /api/upscalers/* 已 PR-6 commit 2 抽到
# api/routers/{secrets,models,upscalers}.py + api/schemas/models.py。


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
    })


def _publish_version_state(v: dict[str, Any]) -> None:
    bus.publish({
        "type": "version_state_changed",
        "project_id": v["project_id"],
        "version_id": v["id"],
        "status": versions.get_status(v),
        "phase": versions.get_phase(v),
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
    """ADR-0007 §11.8-E：enrich active version label + status，卡片右上角 badge 用。"""
    with db.connection_for() as conn:
        rows = projects.list_projects(conn)
        enriched: list[dict[str, Any]] = []
        for r in projects.projects_with_stats(rows):
            r["active_version_label"] = None
            r["active_version_status"] = None
            avid = r.get("active_version_id")
            if avid:
                av = versions.get_version(conn, int(avid))
                if av:
                    r["active_version_label"] = av["label"]
                    r["active_version_status"] = versions.get_status(av)
            enriched.append(r)
    return {"items": enriched}


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


# ---------------------------------------------------------------------------
# Phase cursor 推进 / 跳过 — ADR-0007 §11.5-A / §11.5-B
# ---------------------------------------------------------------------------


def _phase_advance_payload(
    advanced: bool, result: versions_phase.CheckResult,
    new_phase: Optional[str], version: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "advanced": advanced,
        "ok": result.ok,
        "reason": result.reason,
        "new_phase": new_phase,
        "version": version,
    }


@app.post("/api/projects/{pid}/versions/{vid}/advance-phase")
def advance_phase_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """phase cursor 推进 —— "下一步" 按钮调用（ADR-0007 §11.5-A）。

    成功 → cursor++ + 返回新 phase + publish version_state_changed。
    失败 → ok=False + reason（前端 toast），cursor 不动。
    """
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        advanced, result, new_phase = versions_phase.advance_phase(conn, vid)
        v_after = versions.get_version(conn, vid)
    if advanced and v_after is not None:
        _publish_version_state(v_after)
    return _phase_advance_payload(advanced, result, new_phase, v_after)


@app.post("/api/projects/{pid}/versions/{vid}/skip-phase")
def skip_phase_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """跳过可跳过的 phase（当前仅 regularizing；ADR-0007 §11.5-A）。"""
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        advanced, result, new_phase = versions_phase.skip_phase(conn, vid)
        v_after = versions.get_version(conn, vid)
    if advanced and v_after is not None:
        _publish_version_state(v_after)
    return _phase_advance_payload(advanced, result, new_phase, v_after)


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


class _BundleOptionsBody(BaseModel):
    train: bool = True
    train_captions: bool = True
    reg: bool = False
    reg_captions: bool = False
    include_config: bool = False

    def to_options(self) -> train_io.BundleOptions:
        return train_io.BundleOptions(
            train=self.train,
            train_captions=self.train_captions,
            reg=self.reg,
            reg_captions=self.reg_captions,
            include_config=self.include_config,
        )


class _BundleImportBody(BaseModel):
    path: Optional[str] = None
    filename: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "_BundleImportBody":
        if sum(bool(v) for v in (self.path, self.filename)) != 1:
            raise ValueError("exactly one of path or filename is required")
        return self


# /api/data-exports 已 PR-6 commit 1 抽到 api/routers/data_exports.py。


@app.get("/api/projects/{pid}/versions/{vid}/bundle.zip")
def export_version_bundle(
    pid: int,
    vid: int,
    background: BackgroundTasks,
    train: bool = True,
    train_captions: bool = True,
    reg: bool = False,
    reg_captions: bool = False,
    include_config: bool = False,
) -> FileResponse:
    """按选项临时打包 bundle.zip（schema_version 2）并交给浏览器下载。"""
    import tempfile

    opts = train_io.BundleOptions(
        train=train,
        train_captions=train_captions,
        reg=reg,
        reg_captions=reg_captions,
        include_config=include_config,
    )

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
            train_io.export_bundle(conn, vid, tmp_path, opts)
        except train_io.TrainIOError as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise

    bus.publish({"type": "version_bundle_zip_ready", "project_id": pid, "version_id": vid})
    background.add_task(lambda: tmp_path.unlink(missing_ok=True))
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=f"{p['slug']}-{v['label']}.bundle.zip",
        background=background,
    )


@app.post("/api/projects/{pid}/versions/{vid}/export-bundle")
def export_version_bundle_to_data_exports(
    pid: int,
    vid: int,
    body: _BundleOptionsBody,
) -> dict[str, Any]:
    """按选项打包 bundle.zip 并保存到 data_exports/。"""
    opts = body.to_options()
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        p = projects.get_project(conn, pid)
        assert p is not None
        DATA_EXPORTS.mkdir(parents=True, exist_ok=True)
        dest = _unique_data_export_path(f"{p['slug']}-{v['label']}.bundle.zip")
        try:
            train_io.export_bundle(conn, vid, dest, opts)
        except train_io.TrainIOError as exc:
            dest.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            dest.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise

    bus.publish({"type": "version_bundle_zip_ready", "project_id": pid, "version_id": vid})
    return _export_result(dest)


def _bundle_import_payload(result: dict[str, Any]) -> dict[str, Any]:
    p = result["project"]
    _publish_project_state(p)
    _publish_version_state(result["version"])
    return {
        "project": _project_payload(p),
        "version": result["version"],
        "stats": result["stats"],
    }


def _import_bundle_from_path(dest: Path, original: str) -> dict[str, Any]:
    if not dest.exists():
        raise HTTPException(404, f"文件不存在: {original}")
    if not dest.is_file():
        raise HTTPException(400, "请选择 zip 文件")
    if dest.suffix.lower() != ".zip":
        raise HTTPException(400, "请选择 .zip 文件")
    with db.connection_for() as conn:
        try:
            result = train_io.import_bundle(conn, dest, USER_PRESETS_DIR)
        except train_io.TrainIOError as exc:
            raise HTTPException(400, str(exc)) from exc
    return _bundle_import_payload(result)


@app.post("/api/projects/import-bundle")
def import_bundle_zip(body: _BundleImportBody) -> dict[str, Any]:
    """从 PathPicker 路径或 data_exports 文件名导入 bundle（v1/v2 均支持）。"""
    if body.filename:
        return _import_bundle_from_path(_data_export_path(body.filename), body.filename)
    assert body.path is not None
    dest = Path(body.path)
    if not dest.is_absolute():
        dest = (REPO_ROOT / dest).resolve()
    else:
        dest = dest.resolve()
    return _import_bundle_from_path(dest, body.path)


@app.post("/api/projects/import-bundle/upload")
async def import_bundle_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """上传 bundle zip → 新建 project + version。"""
    import tempfile

    if not file.filename:
        raise HTTPException(400, "缺少上传文件")
    if Path(file.filename).suffix.lower() != ".zip":
        raise HTTPException(400, "请选择 .zip 文件")
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.close()
        return _import_bundle_from_path(Path(tmp.name), file.filename)
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass


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
    _publish_job_state(job)
    return job


def _apply_project_upload_result(pid: int, result: uploads_svc.UploadResult) -> dict[str, Any]:
    # ADR-0007 PR-5: project 无 stage 字段；upload 完成由前端实时扫 download/ 派生数字
    return result.as_dict()


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
    return _apply_project_upload_result(pid, result)


class _UploadFromPathBody(BaseModel):
    path: str


@app.post("/api/projects/{pid}/upload-from-path")
def upload_local_file_from_path(pid: int, body: _UploadFromPathBody) -> dict[str, Any]:
    """从 server 可见路径导入单图或 zip → project 的 download/。"""
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    src = Path(body.path)
    if not src.is_absolute():
        src = (REPO_ROOT / src).resolve()
    else:
        src = src.resolve()
    if not src.exists():
        raise HTTPException(404, f"文件不存在: {body.path}")
    if not src.is_file():
        raise HTTPException(400, "请选择文件")
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    with src.open("rb") as fh:
        result = uploads_svc.accept_many([(src.name, fh)], pdir)
    return _apply_project_upload_result(pid, result)


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
    _publish_job_state(job)
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


@app.get("/api/projects/{pid}/preprocess/duplicates/removed")
def list_duplicate_removed(pid: int) -> dict[str, Any]:
    """总览页「已删除」tab：列出被去重审核标记的 manifest entries。

    返回 `{images: [{name, source, w, h, mtime, size}, ...]}`。物理图仍在
    `download/{source}`，缩略图按 download bucket + source 取。恢复走
    `POST /api/projects/{pid}/preprocess/files/restore`（restore() 对
    duplicate_removed entry 也 work：删 entry，没 PNG 时静默跳过）。
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    return {"images": preprocess_svc.list_duplicate_removed_workspace(p)}


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
    _publish_job_state(job)
    return job


@app.post("/api/projects/{pid}/preprocess/files/reset")
def reset_preprocess_files(pid: int) -> dict[str, Any]:
    """整项目预处理状态归零：删 manifest 所有 entry + 删 preprocess/ 所有 PNG。

    工具栏「总览」tab 的「撤销全部」走这个；下游 resolver 回看 download/ 原图。
    `preprocess_manifest.clear_all` 已存在；这里只是 HTTP 入口 + 项目存在校验。
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    pdir = projects.project_dir(p["id"], p["slug"])
    preprocess_manifest.clear_all(pdir)
    _publish_project_state(p)
    return {"ok": True}


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


# `_thumb_response` 已 PR-6 抽到 api/responses.py，project_thumb 走 import 复用。


@app.get("/api/projects/{pid}/thumb")
def project_thumb(
    pid: int,
    bucket: str = "download",
    name: str = "",
    size: int = 256,
    raw: int = 0,
) -> FileResponse:
    """缩略图：默认 256px JPEG（缓存）；size=0 → 原图。

    两种 bucket：
      - `bucket=download`（默认）：`name` 是 download/ 下的原始文件名。
        后端通过 `preprocess_manifest.resolve_origin()` 决定实际字节路径：
        未处理 → download/{name}，已处理 → preprocess/ 下第一个 origin 匹配
        的派生。前端"按 download 名"调用时不需要感知预处理。
      - `bucket=preprocess`：`name` 是 preprocess/ 下的**实际产物文件名**
        （含 multi-crop 派生的 _c0 / _c1 后缀）。直接按文件名取，**不走**
        resolve_origin —— multi-crop 后多个产物共享同一 origin，按 origin
        永远落到 [0] 是 bug。裁剪 / 总览页应该走这条来精确寻址。

    `raw=1`（仅 bucket=download）：跳过 resolve_origin，强制读 download/{name}
    原始字节。给「对比预览」场景用：左 pane 永远要 download 原图，不能被
    preprocess 派生 hijack。

    缓存路径：`studio_data/thumb_cache/{sha1(src+mtime+size)}.jpg`。
    源文件 mtime 变化会自动 invalidate（hash 变）。
    """
    if bucket not in ("download", "preprocess"):
        raise HTTPException(400, f"unknown bucket: {bucket}")
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    pdir = projects.project_dir(p["id"], p["slug"])
    preprocess_manifest.ensure_manifest(pdir)

    if bucket == "preprocess":
        # Direct addressing — no resolve. Path traversal guard against the
        # actual preprocess/ dir (any filename including _c0/_c1 derivatives).
        _safe_join_or_400(pdir / "preprocess", name)
        f = pdir / "preprocess" / name
    elif raw:
        # bucket=download + raw=1: bypass resolve_origin, hand back the
        # untouched download/{name} bytes. Used by the processed-tab compare
        # preview left pane (need the original, not the derivative).
        _safe_join_or_400(pdir / "download", name)
        f = pdir / "download" / name
    else:
        # bucket=download — historical behavior: address by download name,
        # resolve to first preprocess product if any (1:1 / multi-crop cases).
        # duplicate_removed origins: resolve_origin returns [] but the original
        # file in download/ still exists; the Download page must keep showing
        # it (软删除 ≠ 不可见). Fall back to download/{name} like any other
        # un-resolved origin.
        _safe_join_or_400(pdir / "download", name)
        candidates = preprocess_manifest.resolve_origin(pdir, name)
        f = candidates[0] if candidates else (pdir / "download" / name)
        # Curation passes multi-crop derivative names (X_c0.png) through this
        # endpoint with bucket=download. resolve_origin only matches by origin,
        # not by entry key, so derivatives miss → f points at a non-existent
        # download/X_c0.png. Fall back: if the name IS a preprocess entry key,
        # serve preprocess/{name} directly. Filename was already safety-checked
        # against download/, same validation applies to preprocess/.
        if not f.exists() and preprocess_manifest.get_entry(pdir, name) is not None:
            f = pdir / "preprocess" / name

    if not f.exists() or f.suffix.lower() not in datasets.IMAGE_EXTS:
        logger.info("thumb 404: pid=%s bucket=%s name=%s -> %s", pid, bucket, name, f)
        raise HTTPException(404)
    return _thumb_response(f, size)


# /api/jobs/{jid} / log / cancel 已 PR-6 commit 2 抽到 api/routers/jobs.py。
# /api/projects/{pid}/versions/{vid}/jobs/latest 仍在本文件下（projects 域）。


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


# jobs log / cancel 已搬到 api/routers/jobs.py


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


class DuplicateScanRequest(BaseModel):
    match_scope: str = "both"
    hash_size: int = duplicate_finder.DEFAULT_HASH_SIZE
    hash_workers: int = duplicate_finder.DEFAULT_HASH_WORKERS
    tile_grids: list[int] = list(duplicate_finder.DEFAULT_TILE_GRIDS)
    structure_threshold: int = duplicate_finder.DEFAULT_STRUCTURE_THRESHOLD
    variant_score: float = duplicate_finder.DEFAULT_VARIANT_SCORE
    aspect_tolerance: float = duplicate_finder.DEFAULT_ASPECT_TOLERANCE
    min_close_tiles: float = duplicate_finder.DEFAULT_MIN_CLOSE_TILES
    tile_median: float = duplicate_finder.DEFAULT_TILE_MEDIAN
    min_gray_close: float = duplicate_finder.DEFAULT_MIN_GRAY_CLOSE


class DuplicateApplyRequest(BaseModel):
    names: list[str]


def _curation_err_code(exc: curation.CurationError) -> int:
    msg = str(exc)
    if "不存在" in msg:
        return 404
    if "已存在" in msg or "非法" in msg:
        return 400
    return 422


@app.get("/api/projects/{pid}/versions/{vid}/curation")
def get_curation(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            return curation.curation_view(conn, pid, vid)
        except curation.CurationError as exc:
            raise HTTPException(_curation_err_code(exc), str(exc)) from exc


def _duplicate_err_code(exc: duplicate_finder.DuplicateFinderError) -> int:
    msg = str(exc)
    if "not found" in msg or "不存在" in msg:
        return 404
    if "invalid" in msg or "非法" in msg:
        return 400
    if "not installed" in msg:
        return 422
    return 422


@app.post("/api/projects/{pid}/preprocess/duplicates/scan")
def scan_preprocess_duplicates(
    pid: int, body: DuplicateScanRequest
) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            options = duplicate_finder.options_from_payload(body.model_dump())
            last_progress_at = 0.0

            def publish_progress(payload: dict[str, Any]) -> None:
                nonlocal last_progress_at
                now = time.monotonic()
                if now - last_progress_at < 1.0:
                    return
                last_progress_at = now
                bus.publish({
                    "type": "duplicate_scan_progress",
                    "project_id": pid,
                    "status": "running",
                    **payload,
                })

            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "status": "running",
                "text": "Scanning duplicate candidates...",
            })
            result = duplicate_finder.scan_project_duplicates(
                conn,
                pid,
                options,
                on_progress=publish_progress,
            )
            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "status": "done",
                "total_images": result["total_images"],
                "group_count": result["group_count"],
                "candidate_count": result["candidate_count"],
                "elapsed_seconds": result["elapsed_seconds"],
                "text": (
                    f"Scanned {result['total_images']} images; "
                    f"found {result['group_count']} groups / "
                    f"{result['candidate_count']} candidates."
                ),
            })
            return result
        except curation.CurationError as exc:
            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "status": "failed",
                "text": str(exc),
            })
            raise HTTPException(_curation_err_code(exc), str(exc)) from exc
        except duplicate_finder.DuplicateFinderError as exc:
            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "status": "failed",
                "text": str(exc),
            })
            raise HTTPException(_duplicate_err_code(exc), str(exc)) from exc


@app.post("/api/projects/{pid}/preprocess/duplicates/apply")
def apply_preprocess_duplicates(
    pid: int, body: DuplicateApplyRequest
) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            result = duplicate_finder.apply_duplicate_removals(
                conn,
                pid,
                names=body.names,
            )
            project = projects.get_project(conn, pid)
        except curation.CurationError as exc:
            raise HTTPException(_curation_err_code(exc), str(exc)) from exc
        except duplicate_finder.DuplicateFinderError as exc:
            raise HTTPException(_duplicate_err_code(exc), str(exc)) from exc
    if project:
        _publish_project_state(project)
    return result


@app.post("/api/projects/{pid}/duplicates/scan")
def scan_project_duplicates(
    pid: int, body: DuplicateScanRequest
) -> dict[str, Any]:
    """Backward-compatible alias; UI uses /preprocess/duplicates/scan."""
    return scan_preprocess_duplicates(pid, body)


@app.post("/api/projects/{pid}/duplicates/apply")
def apply_project_duplicates(
    pid: int, body: DuplicateApplyRequest
) -> dict[str, Any]:
    """Backward-compatible alias; now marks manifest duplicate_removed."""
    return apply_preprocess_duplicates(pid, body)


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


# LLM tagger admin / WD14 / torch / flash-attention / xformers 请求 BaseModel
# 已 PR-6 commit 3 抽到 api/schemas/installs.py。


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


# /api/wd14/{runtime,install}, /api/torch/{status,reinstall},
# /api/flash-attention/{status,install}, /api/xformers/{status,install},
# /api/llm-tagger/{models/refresh,test} 已 PR-6 commit 3 抽到 api/routers/installs.py。
# (/api/tagger/{name}/check 在 commit 1 抽到 api/routers/tagger.py。)


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


# `_resolve_anima_model_paths()` 已 PR-6 commit 5 抽到 api/deps.py（reg_generate_prior
# 也在用，与 generate router 共享）。


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

    from studio.services.xformers_setup import detect_attention_backend
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
        attention_backend=detect_attention_backend(),
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


# /api/generate/* (8 routes) + GenerateRequest BaseModel 已 PR-6 commit 5 抽到
# api/routers/generate.py + api/schemas/generate.py。


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
        # ADR-0007 PR-5: version.status 由 supervisor 在 _spawn_task 推到 training；
        # project 无 stage；这里不再 advance。
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


# /api/queue/* (20 routes) + EnqueueRequest / ReorderRequest / ImportRequest /
# ExportOutputsBody + _task_output_* helpers 已 PR-6 commit 6 抽到
# api/routers/queue/{lifecycle,io,outputs}.py + api/schemas/queue.py（3 文件
# 子包 internal split：12 lifecycle + 3 io + 5 outputs = 20）。

# `_supervisor()` / `_resolve_anima_model_paths()` 已 PR-6 commit 2/5 抽到
# api/deps.py，server.py 内剩余 handler 复用同函数
from .api.deps import _resolve_anima_model_paths, _supervisor  # noqa: E402




# ---------------------------------------------------------------------------
# /api/datasets  (P4)
# ---------------------------------------------------------------------------


# /api/datasets, /api/browse, /api/datasets/thumbnail 已 PR-5 commit 2 抽到 api/routers/browse.py。


# ---------------------------------------------------------------------------


# /api/logs/{task_id} 已 PR-6 commit 1 抽到 api/routers/logs.py。


# /api/events SSE 已 PR-5 commit 2 抽到 api/routers/events_sse.py。


# ---------------------------------------------------------------------------
# /samples
# ---------------------------------------------------------------------------


# /samples/{filename} 已 PR-6 commit 1 抽到 api/routers/samples.py。


# ---------------------------------------------------------------------------
# 静态资源
# ---------------------------------------------------------------------------


# React 应用：构建后通过 /studio 访问。开发期请用 `npm run dev` 起 5173。
# `SPAStaticFiles` 已 PR-5 抽到 studio/api/static.py。
if WEB_DIST.exists():
    app.mount(
        "/studio",
        SPAStaticFiles(directory=str(WEB_DIST), html=True),
        name="studio",
    )


# /api/system/* (11 routes) + UpdateRequest BaseModel + _RESTART_FLAG /
# _SHUTDOWN_FORCE_EXIT_TIMEOUT / _raise_sigint_after_response / _check_no_running_tasks
# helper 已 PR-6 commit 4 抽到 api/routers/system.py + api/schemas/system.py。


# `/` 根路径 redirect 已 PR-6 commit 1 抽到 api/routers/root.py。


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

from .api.main import main  # noqa: E402  # PR-5 抽到 api.main，re-export 保 `from studio.server import main` 兼容


if __name__ == "__main__":
    main()
