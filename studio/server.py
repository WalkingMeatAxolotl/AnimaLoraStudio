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


# ProjectCreate / ProjectUpdate / VersionCreate / VersionUpdate BaseModel 已
# PR-6.5 commit 1 抽到 api/schemas/projects.py。
#
# _project_payload / _publish_project_state / _publish_version_state /
# _project_err_code 已 PR-6.5 commit 1 抽到 api/routers/projects/_shared.py
# （projects 子包内共用，避免跨域反向耦合）。剩余 server.py 内 projects-域
# handler（commit 2-5 抽走）通过下方 import 复用。
from .api.routers.projects._shared import (  # noqa: E402
    _project_and_version_or_404,
    _project_err_code,
    _project_payload,
    _publish_project_state,
    _publish_version_state,
    _reg_dir,
    _version_dir_or_404,
    _version_train_dir_or_404,
)


# /api/projects + versions CRUD + activate + advance/skip-phase + lora_ckpts /
# state_ckpts （16 routes） 已 PR-6.5 commit 1 抽到 api/routers/projects/crud.py。


# Train / Bundle export + import (6 routes) + _BundleOptionsBody / _BundleImportBody +
# _bundle_import_payload / _import_bundle_from_path helpers 已 PR-6.5 commit 2 抽到
# api/routers/projects/exports.py + api/schemas/exports.py。


# /api/projects/{pid}/download/* (4 routes), /upload + /upload-from-path (2),
# /preprocess/* (8 routes incl. crop / files reset+restore / thumb) — 14 routes
# + 6 BaseModel (Download/Estimate/UploadFromPath/PreprocessStart/PreprocessRestore/
# CropRect/PreprocessCrop) + _publish_job_state / _apply_project_upload_result helpers
# 已 PR-6.5 commit 3 抽到 api/routers/projects/ingestion.py + api/schemas/ingestion.py。
# _publish_job_state 升 api/routers/projects/_shared.py（tag / reg 也用）。

# /api/projects/{pid}/files (delete + list), /thumb, /versions/{vid}/jobs/latest,
# /versions/{vid}/curation get + copy/remove/folder, /preprocess/duplicates/scan+apply
# + backward-alias /duplicates/scan+apply (12 routes) + 6 BaseModel + 2 err_code
# helpers 已 PR-6.5 commit 4 抽到 api/routers/projects/curation.py +
# api/schemas/curation.py。


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


# `_version_train_dir_or_404` 已 PR-6.5 commit 1 抽到 api/routers/projects/_shared.py
# （顶部 import 复用）。


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


# `_version_dir_or_404` 已 PR-6.5 commit 1 抽到 api/routers/projects/_shared.py。


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


# `_reg_dir` 已 PR-6.5 commit 1 抽到 api/routers/projects/_shared.py。


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


# `_project_and_version_or_404` 已 PR-6.5 commit 1 抽到 api/routers/projects/_shared.py。


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
