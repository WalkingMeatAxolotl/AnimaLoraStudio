"""测试页远程 Booru 画廊 API schema。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GallerySource = Literal["danbooru", "gelbooru"]
GalleryRating = Literal["general", "sensitive", "questionable", "explicit"]
GalleryTagger = Literal["wd14", "cltagger", "llm"]


class GalleryItem(BaseModel):
    source: GallerySource
    post_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    tags: list[str]
    thumbnail_url: str
    image_url: str


class GallerySearchResponse(BaseModel):
    items: list[GalleryItem]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    has_more: bool


class GalleryTagRequest(BaseModel):
    source: GallerySource
    post_id: str = Field(min_length=1, max_length=32, pattern=r"^\d+$")
    image_url: str = Field(min_length=1, max_length=4096)
    tagger: GalleryTagger


class GalleryTagResponse(BaseModel):
    prompt: str
