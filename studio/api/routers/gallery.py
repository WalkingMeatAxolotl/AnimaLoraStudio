"""测试页远程 Booru 画廊端点。"""
from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from ...services.booru import gallery
from ..schemas.gallery import (
    GalleryRating,
    GallerySearchResponse,
    GalleryTagRequest,
    GalleryTagResponse,
)

router = APIRouter()


@router.get("/api/gallery/search", response_model=GallerySearchResponse)
def search_gallery(
    source: Literal["danbooru", "gelbooru"] = "danbooru",
    query: str = Query(default="", max_length=500),
    rating: list[GalleryRating] | None = Query(default=None),
    date_from: date | None = None,
    date_to: date | None = None,
    page: int = Query(default=1, ge=1, le=10_000),
) -> dict:
    return gallery.search_gallery(
        source=source,
        query=query,
        ratings=rating or ["general"],
        date_from=date_from,
        date_to=date_to,
        page=page,
    )


@router.get("/api/gallery/image")
def gallery_image(
    source: Literal["danbooru", "gelbooru"],
    post_id: str = Query(min_length=1, max_length=32, pattern=r"^\d+$"),
    url: str = Query(min_length=1, max_length=4096),
) -> FileResponse:
    path = gallery.fetch_cached_image(source, url, post_id=post_id)
    media_type = {
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            "Cache-Control": "private, max-age=604800, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/api/gallery/tag", response_model=GalleryTagResponse)
def tag_gallery_image(body: GalleryTagRequest) -> dict[str, str]:
    prompt = gallery.tag_gallery_image(
        source=body.source,
        post_id=body.post_id,
        image_url=body.image_url,
        tagger_name=body.tagger,
    )
    return {"prompt": prompt}
