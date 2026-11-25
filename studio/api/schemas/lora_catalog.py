"""/api/lora-catalog 请求 / 响应模型。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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
    # 训练底模层数不在列表里：读 header 有 IO 成本，改由 POST /api/lora-catalog/arch 按需取。


class LoraCatalogSource(BaseModel):
    source_type: Literal["project", "studio_models", "external"]
    source_id: str
    source_label: str
    path: str
    item_count: int
    error: str | None = None
    project_archived: bool = False
    created_at: float | None = None
    updated_at: float | None = None


class LoraCatalogResponse(BaseModel):
    items: list[LoraCatalogItem]
    sources: list[LoraCatalogSource]
    total: int
    cursor: int
    next_cursor: int | None
    generated_at: float
    cached: bool
    cache_ttl_seconds: float


class LoraArchRequest(BaseModel):
    paths: list[str] = Field(default_factory=list, max_length=500)


class LoraArchItem(BaseModel):
    path: str
    #: 训练底模 DiT 层数（lora_compat 契约：元数据 / 键扫描下界 / 未知）
    base_num_blocks: int | None = None
    base_arch_source: Literal["metadata", "keys", "unknown"] = "unknown"


class LoraArchResponse(BaseModel):
    items: list[LoraArchItem]
