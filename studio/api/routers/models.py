"""模型 catalog / 下载（PR-6 commit 2 从 server.py 抽出）。

3 routes（PP7 第一刀域）：
    GET  /api/models/catalog         列已知模型 + 各自磁盘状态 + 当前下载状态
    GET  /api/models/path-defaults   当前 Settings 算出的 4 个模型字段绝对路径
    POST /api/models/download        启动后台下载，返回 status key
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..schemas.models import ModelDownloadRequest
from ...services import models as model_downloader

router = APIRouter()


@router.get("/api/models/catalog")
def get_models_catalog() -> dict[str, Any]:
    """前端设置页 Models 区块用：列已知模型 + 各自磁盘状态 + 当前下载状态。"""
    return model_downloader.build_catalog()


@router.get("/api/models/path-defaults")
def get_models_path_defaults() -> dict[str, str]:
    """当前 Settings 算出的 4 个模型字段绝对路径。

    给预设页 reset 按钮和「新建预设」初始填充用——这两个场景没有 project
    上下文，拿不到 /api/projects/{pid}/versions/{vid}/config 里的
    project_specific_defaults，所以单独开一个端点。
    """
    return model_downloader.default_paths_for_new_version()


@router.post("/api/models/download")
def start_model_download(body: ModelDownloadRequest) -> dict[str, Any]:
    """启动后台下载，立即返回 status key；前端通过 SSE
    (`model_download_changed`) 或轮询 catalog 看进度。"""
    try:
        key = model_downloader.trigger(body.model_id, body.variant)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    snap = model_downloader.get_status_snapshot()
    return {"key": key, "status": snap.get(key, {}).get("status", "running")}
