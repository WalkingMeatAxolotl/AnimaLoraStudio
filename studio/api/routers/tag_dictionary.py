"""Tag 翻译词典端点。

4 routes：
    GET  /api/tag-dictionary/meta     当前词典 meta（前端启动时 ping，看是否已加载）
    GET  /api/tag-dictionary/data     完整 dict（约 600KB gzip）给前端 in-memory 用
    POST /api/tag-dictionary/upload   multipart 上传 csv/txt 替换词典
    POST /api/tag-dictionary/reset    重新拉默认源（用户首次下载失败 / 想恢复时用）
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

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
        raise HTTPException(status_code=404, detail="tag dictionary not initialized")
    entries, meta = loaded
    # `Cache-Control: public, max-age=300` 让浏览器 5 分钟内不重发；上传/reset 后
    # 后端没有 ETag，前端通过 meta.downloaded_at 比对决定是否强刷。
    return JSONResponse(
        content={"entries": entries, "meta": meta},
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.post("/api/tag-dictionary/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    try:
        meta = td.apply_uploaded(content, file.filename or "user-upload")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"loaded": True, "meta": meta}


@router.post("/api/tag-dictionary/reset")
def reset() -> dict[str, Any]:
    try:
        meta = td.reset_to_default()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"loaded": True, "meta": meta}
