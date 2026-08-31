"""GET /api/lora-catalog 响应模型。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class LoraCatalogItem(BaseModel):
    path: str
    name: str
    relative_path: str
    size: int
    mtime: float
    source_type: Literal["project", "studio_models", "external"]
    source_id: str
    source_label: str
    project_id: int | None = None
    version_id: int | None = None
    project_title: str | None = None
    version_label: str | None = None
    project_archived: bool = False
    kind: Literal["final", "step", "epoch", "other"] = "other"


class LoraCatalogSource(BaseModel):
    source_type: Literal["project", "studio_models", "external"]
    source_id: str
    source_label: str
    path: str
    item_count: int
    error: str | None = None
    project_archived: bool = False


class LoraCatalogResponse(BaseModel):
    items: list[LoraCatalogItem]
    sources: list[LoraCatalogSource]
    total: int
    cursor: int
    next_cursor: int | None
    generated_at: float
    cached: bool
    cache_ttl_seconds: float
