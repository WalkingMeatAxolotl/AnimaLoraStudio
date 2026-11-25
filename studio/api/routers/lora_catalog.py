"""统一 LoRA catalog 只读端点。"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from ...services import lora_catalog
from ..schemas.lora_catalog import (
    LoraArchRequest,
    LoraArchResponse,
    LoraCatalogResponse,
)

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
    sources_only: bool = False,
) -> dict:
    """聚合项目输出、`{models_root}/loras` 与配置的第三方 LoRA 目录。

    ``sources_only=true`` 只返回来源列表与计数（打开抽屉用），不扫任何来源的
    条目；条目请带 ``source`` 按来源取。
    """
    if sources_only:
        return lora_catalog.list_sources(
            include_archived=include_archived, refresh=refresh,
        )
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


@router.post("/api/lora-catalog/arch", response_model=LoraArchResponse)
def post_lora_catalog_arch(body: LoraArchRequest) -> dict:
    """按需读取 catalog 内 LoRA 的训练底模层数（列表徽标用，不在列表关键路径上）。"""
    return {"items": lora_catalog.read_arch(body.paths)}
