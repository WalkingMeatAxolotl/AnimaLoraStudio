"""统一 LoRA catalog 只读端点。"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from ...services import lora_catalog
from ..schemas.lora_catalog import LoraCatalogResponse

router = APIRouter()


@router.get("/api/lora-catalog", response_model=LoraCatalogResponse)
def get_lora_catalog(
    q: str = "",
    source: str | None = None,
    sort: Literal["recommended", "name", "mtime", "size", "source"] = "recommended",
    order: Literal["asc", "desc"] = "asc",
    include_archived: bool = False,
    limit: int = Query(default=100, ge=1, le=500),
    cursor: int = Query(default=0, ge=0),
    refresh: bool = False,
) -> dict:
    """聚合项目输出、`{models_root}/loras` 与配置的第三方 LoRA 目录。"""
    return lora_catalog.query_catalog(
        q=q,
        source=source,
        sort=sort,
        order=order,
        include_archived=include_archived,
        limit=limit,
        cursor=cursor,
        refresh=refresh,
    )
