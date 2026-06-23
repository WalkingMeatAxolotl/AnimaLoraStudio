"""Tag 翻译词典端点。

4 routes：
    GET  /api/tag-dictionary/meta     当前词典 meta（前端启动时 ping，看是否已加载）
    GET  /api/tag-dictionary/data     完整 dict（默认源 20 万条，约 7MB / gzip 3.5MB）给前端 in-memory 用
    POST /api/tag-dictionary/upload   multipart 上传 csv/txt 替换词典
    POST /api/tag-dictionary/reset    重新拉默认源（用户首次下载失败 / 想恢复时用）
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ...domain.errors import DomainError, NotFoundError, ValidationError
from ...infrastructure import tag_dictionary as td

router = APIRouter()


@router.get("/api/tag-dictionary/meta")
def get_meta() -> dict[str, Any]:
    meta = td.get_meta()
    return {"loaded": meta is not None, "meta": meta}


@router.get("/api/tag-dictionary/data")
def get_data() -> JSONResponse:
    loaded = td.load_active()
    if loaded is None:
        raise NotFoundError(
            "Tag dictionary is not loaded", code="tag.dictionary_not_loaded",
        )
    entries, meta = loaded
    # `Cache-Control: public, max-age=300` 让浏览器 5 分钟内不重发；上传/reset 后
    # 后端没有 ETag，前端通过 meta.downloaded_at 比对决定是否强刷。
    # `keys` 单独下发：JS 对象会把整数型 key（"69"、年份 tag 等）重排到最前，
    # 前端靠这个数组拿到文件原始行序（默认源 = post_count 降序的热度序）。
    return JSONResponse(
        content={"entries": entries, "keys": list(entries.keys()), "meta": meta},
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.post("/api/tag-dictionary/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    try:
        meta = td.apply_uploaded(content, file.filename or "user-upload")
    except ValueError as exc:
        raise ValidationError(
            f"Invalid tag dictionary file: {exc}",
            code="tag.dictionary_upload_invalid",
            details={"reason": str(exc)}, http_status=400,
        ) from exc
    return {"loaded": True, "meta": meta}


@router.post("/api/tag-dictionary/reset")
def reset() -> dict[str, Any]:
    try:
        meta = td.reset_to_default()
    except RuntimeError as exc:
        raise DomainError(
            f"Failed to download the default tag dictionary: {exc}",
            code="tag.dictionary_download_failed",
            details={"reason": str(exc)}, http_status=502,
        ) from exc
    return {"loaded": True, "meta": meta}
