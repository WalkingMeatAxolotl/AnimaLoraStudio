"""放大器（upscaler）切换（PR-6 commit 2 从 server.py 抽出）。

1 route：
    POST /api/upscalers/select  切换默认放大器（预设 / custom 文件名 / 本地路径）

自定义下载端点（POST /api/upscalers/download_custom）已被统一来源候选取代
（POST /api/model-sources/upscaler 登记候选 + 通用 /api/models/download 触发，
model_id=upscaler_custom）。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..schemas.models import UpscalerSelectRequest
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
