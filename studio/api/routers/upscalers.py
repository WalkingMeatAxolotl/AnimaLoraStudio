"""放大器（upscaler）切换 / 自定义下载（PR-6 commit 2 从 server.py 抽出）。

2 routes：
    POST /api/upscalers/select          切换默认放大器（预设 / custom 文件名）
    POST /api/upscalers/download_custom 自定义放大器下载（HF / MS repo + filename）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..schemas.models import UpscalerCustomDownloadRequest, UpscalerSelectRequest
from ... import secrets
from ...domain.errors import InvalidPathError, NotFoundError, ValidationError
from ...services import models as model_downloader

router = APIRouter()


@router.post("/api/upscalers/select")
def select_upscaler(body: UpscalerSelectRequest) -> dict[str, Any]:
    """切换默认放大器。写入 secrets.models.selected_upscaler。

    接受预设 label 或本地已有的 custom 文件名；非法值（既不在预设也不在
    upscalers/ 目录）返回 400。
    """
    label = body.label.strip()
    if not label:
        raise ValidationError(
            "Upscaler name is required", code="upscaler.label_required", http_status=400,
        )
    valid = label in model_downloader.UPSCALER_VARIANTS
    if not valid:
        # custom 文件名：必须已经在磁盘上
        try:
            target = model_downloader.upscaler_target(label)
        except ValueError as exc:
            raise InvalidPathError("Invalid path", details={"reason": str(exc)}) from exc
        if not target.exists():
            raise NotFoundError(
                f'Upscaler "{label}" not found',
                code="upscaler.not_found", details={"name": label},
            )
    cur = secrets.load()
    new_models = cur.models.model_copy(update={"selected_upscaler": label})
    new = cur.model_copy(update={"models": new_models})
    secrets.save(new)
    return {"selected": label}


@router.post("/api/upscalers/download_custom")
def start_upscaler_custom_download(
    body: UpscalerCustomDownloadRequest,
) -> dict[str, Any]:
    """自定义放大器下载：用户填 HF/MS repo + 文件名，落到 `{upscalers}/{filename}`。

    复用通用 start_download_async；key 形如 `upscaler:custom:foo.pth` 便于前端 SSE
    过滤 + catalog 状态匹配。
    """
    if body.source not in ("hf", "ms"):
        raise ValidationError(
            f"Unsupported download source: {body.source}",
            code="upscaler.download_source_invalid",
            details={"source": body.source}, http_status=400,
        )
    if not body.repo_id.strip() or not body.filename.strip():
        raise ValidationError(
            "Repository ID and file name are required",
            code="upscaler.download_fields_required", http_status=400,
        )
    save_name = Path(body.filename).name
    if not save_name.lower().endswith(model_downloader.UPSCALER_EXTS):
        _exts = " / ".join(model_downloader.UPSCALER_EXTS)
        raise ValidationError(
            f"Select a {_exts} file",
            code="file.ext_invalid", details={"types": _exts}, http_status=400,
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
