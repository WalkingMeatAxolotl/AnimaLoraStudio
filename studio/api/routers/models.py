"""模型 catalog / 下载（PR-6 commit 2 从 server.py 抽出）。

7 routes（PP7 第一刀域）：
    GET    /api/models/catalog        列已知模型 + 各自磁盘状态 + 当前下载状态
    GET    /api/models/path-defaults  当前 Settings 算出的 4 个模型字段绝对路径
    POST   /api/models/download       启动后台下载，返回 status key
    POST   /api/models/anima/custom   注册一个本地 .safetensors 主模型
    DELETE /api/models/anima/custom   注销一个本地 custom 主模型
    POST   /api/models/krea2/custom   注册一个 Krea2 本地主模型
    DELETE /api/models/krea2/custom   注销一个 Krea2 本地主模型
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..schemas.models import (
    AnimaCustomModelRequest,
    FamilySwitchRequest,
    ModelDownloadRequest,
)
from ... import secrets
from ...domain.errors import ValidationError
from ...domain.family_switch import switch_family
from ...services import models as model_downloader

router = APIRouter()

# 本地 custom 主模型只接受单文件 .safetensors（防误传可执行 / 目录穿越）。
ANIMA_CUSTOM_EXTS = (".safetensors",)


def _validated_custom_model_path(raw_path: str) -> str:
    raw = (raw_path or "").strip()
    if not raw:
        raise ValidationError(
            "Model path is required",
            code="model.path_required", http_status=400,
        )
    path = Path(raw).expanduser()
    if path.suffix.lower() not in ANIMA_CUSTOM_EXTS:
        raise ValidationError(
            "Select a .safetensors file",
            code="model.ext_invalid",
            details={"types": ".safetensors"}, http_status=400,
        )
    if not path.is_file():
        raise ValidationError(
            "File not found",
            code="model.not_found", details={"path": str(path)}, http_status=400,
        )
    return str(path)


def _add_custom_model(family_id: str, raw_path: str) -> dict[str, Any]:
    resolved = _validated_custom_model_path(raw_path)
    current = secrets.load()
    custom = {key: list(paths) for key, paths in current.models.custom.items()}
    paths = custom.setdefault(family_id, [])
    if resolved not in paths:
        paths.append(resolved)
        new_models = current.models.model_copy(update={"custom": custom})
        secrets.save(current.model_copy(update={"models": new_models}))
    return model_downloader.build_catalog()


def _remove_custom_model(family_id: str, raw_path: str, fallback: str) -> dict[str, Any]:
    target = (raw_path or "").strip()
    current = secrets.load()
    custom = {key: list(paths) for key, paths in current.models.custom.items()}
    custom[family_id] = [path for path in custom.get(family_id, []) if path != target]
    update: dict[str, Any] = {"custom": custom}
    if current.models.selected.get(family_id) == target:
        update["selected"] = {**current.models.selected, family_id: fallback}
    new_models = current.models.model_copy(update=update)
    secrets.save(current.model_copy(update={"models": new_models}))
    return model_downloader.build_catalog()


@router.get("/api/models/catalog")
def get_models_catalog() -> dict[str, Any]:
    """前端设置页 Models 区块用：列已知模型 + 各自磁盘状态 + 当前下载状态。"""
    return model_downloader.build_catalog()


@router.get("/api/models/path-defaults")
def get_models_path_defaults(family: str = "anima") -> dict[str, str]:
    """当前 Settings 算出的 4 个模型字段绝对路径（按 `family` query 参数解析）。

    给预设页 reset 按钮和「新建预设」初始填充用——这两个场景没有 project
    上下文，拿不到 /api/projects/{pid}/versions/{vid}/config 里的
    project_specific_defaults，所以单独开一个端点。
    """
    try:
        return model_downloader.default_paths_for_new_version(family=family)
    except ValueError as exc:
        raise ValidationError(
            f"Unknown model family: {family}",
            code="model.family_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc


@router.post("/api/models/family-switch")
def switch_model_family(body: FamilySwitchRequest) -> dict[str, Any]:
    """训练配置切换模型族的预览计算（多模型 P4-3）。

    纯计算不落盘：重算 4 个权重路径 + 重置族风味字段（sampler / scheduler /
    timestep 等）+ 关闭目标族不支持的能力字段，返回切换后的完整 config 与
    变更清单。前端拿去弹确认对话框，用户确认后走正常保存链路。
    """
    try:
        path_defaults = model_downloader.default_paths_for_new_version(
            family=body.target)
    except ValueError as exc:
        raise ValidationError(
            f"Unknown model family: {body.target}",
            code="model.family_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc
    new_config, changes = switch_family(body.config, body.target, path_defaults)
    return {"config": new_config, "changes": changes}


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


def _assets_or_400(family: str):
    from ...services.models.families import get_assets

    try:
        return get_assets(family)
    except ValueError as exc:
        raise ValidationError(
            f"Unknown model family: {family}",
            code="model.family_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc


@router.post("/api/models/{family}/custom")
def add_custom_model(family: str, body: AnimaCustomModelRequest) -> dict[str, Any]:
    """注册一个本地 `.safetensors` 主模型权重（去重 append），返回新 catalog。

    仅登记路径，不下载 / 不复制。校验文件存在 + 后缀白名单。注册后可在设置页
    选作该族默认主模型，驱动训练新建默认 + 测试出图（在微调权重上炼丹 / 验证）。
    路由按族参数化（多模型 P4-5）：未知族 400。
    """
    assets = _assets_or_400(family)
    return _add_custom_model(assets.family_id, body.path)


@router.delete("/api/models/{family}/custom")
def remove_custom_model(family: str, body: AnimaCustomModelRequest) -> dict[str, Any]:
    """注销一个本地 custom 主模型，返回新 catalog。

    若被删路径正是该族当前默认，把默认重置回最新官方 variant，
    避免训练 / 出图解析落到已注销的路径。
    """
    assets = _assets_or_400(family)
    return _remove_custom_model(assets.family_id, body.path, assets.latest)
