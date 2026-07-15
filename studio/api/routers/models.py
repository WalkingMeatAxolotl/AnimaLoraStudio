"""模型 catalog / 下载（PR-6 commit 2 从 server.py 抽出）。

5 routes（PP7 第一刀域）：
    GET    /api/models/catalog        列已知模型 + 各自磁盘状态 + 当前下载状态
    GET    /api/models/path-defaults  当前 Settings 算出的 4 个模型字段绝对路径
    POST   /api/models/download       启动后台下载，返回 status key
    POST   /api/models/anima/custom   注册一个本地 .safetensors 主模型
    DELETE /api/models/anima/custom   注销一个本地 custom 主模型
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..schemas.models import AnimaCustomModelRequest, ModelDownloadRequest
from ... import secrets
from ...domain.errors import ValidationError
from ...services import models as model_downloader

router = APIRouter()

# 本地 custom 主模型只接受单文件 .safetensors（防误传可执行 / 目录穿越）。
ANIMA_CUSTOM_EXTS = (".safetensors",)


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
        raise ValidationError(
            f"Invalid model selection: {exc}",
            code="model.invalid", details={"reason": str(exc)}, http_status=400,
        ) from exc
    snap = model_downloader.get_status_snapshot()
    return {"key": key, "status": snap.get(key, {}).get("status", "running")}


@router.post("/api/models/anima/custom")
def add_custom_anima(body: AnimaCustomModelRequest) -> dict[str, Any]:
    """注册一个本地 `.safetensors` 主模型权重（去重 append），返回新 catalog。

    仅登记路径，不下载 / 不复制。校验文件存在 + 后缀白名单。注册后可在设置页
    选作默认主模型，驱动训练新建默认 + 测试出图（在微调权重上炼丹 / 验证）。
    """
    raw = (body.path or "").strip()
    if not raw:
        raise ValidationError(
            "Model path is required",
            code="model.path_required", http_status=400,
        )
    p = Path(raw).expanduser()
    if p.suffix.lower() not in ANIMA_CUSTOM_EXTS:
        raise ValidationError(
            "Select a .safetensors file",
            code="model.ext_invalid",
            details={"types": ".safetensors"}, http_status=400,
        )
    if not p.is_file():
        raise ValidationError(
            "File not found",
            code="model.not_found", details={"path": str(p)}, http_status=400,
        )
    resolved = str(p)
    cur = secrets.load()
    paths = list(cur.models.custom_anima_paths)
    if resolved not in paths:
        paths.append(resolved)
        new_models = cur.models.model_copy(update={"custom_anima_paths": paths})
        secrets.save(cur.model_copy(update={"models": new_models}))
    return model_downloader.build_catalog()


@router.delete("/api/models/anima/custom")
def remove_custom_anima(body: AnimaCustomModelRequest) -> dict[str, Any]:
    """注销一个本地 custom 主模型，返回新 catalog。

    若被删路径正是当前默认（`selected_anima`），把默认重置回最新官方 variant，
    避免训练 / 出图解析落到已注销的路径。
    """
    target = (body.path or "").strip()
    cur = secrets.load()
    paths = [p for p in cur.models.custom_anima_paths if p != target]
    update: dict[str, Any] = {"custom_anima_paths": paths}
    if cur.models.selected_anima == target:
        update["selected"] = {**cur.models.selected, "anima": model_downloader.LATEST_ANIMA}
    new_models = cur.models.model_copy(update=update)
    secrets.save(cur.model_copy(update={"models": new_models}))
    return model_downloader.build_catalog()
