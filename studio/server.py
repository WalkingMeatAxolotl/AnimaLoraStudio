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


# /api/projects/{pid}/versions/{vid}/tag, /captions/* (8), /reg + /reg/preview-tags/build/caption/delete (5),
# /reg/generate-prior + /reg/generate-prior/{task_id} (2), /config get/put/from_preset/save_as_preset (4),
# /queue (training launch), /thumb (version thumb) = 23 routes + 12 BaseModel (Wd14/CLTagger/LLMTagger
# Overrides + TagJob + Caption/CommitItem/Commit/BatchOp + RegBuild/RegAi + FromPreset/SaveAsPreset)
# 已 PR-6.5 commit 5 抽到 api/routers/projects/training.py + api/schemas/training.py。



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
