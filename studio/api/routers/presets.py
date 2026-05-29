"""预设 CRUD + 导入导出 + schema（PR-5 从 server.py 抽出）。

13 routes：
    GET  /api/schema                            TrainingConfig JSON schema + GROUP_ORDER
    GET  /api/presets                           list
    GET  /api/presets/{name}                    read
    PUT  /api/presets/{name}                    write
    DELETE /api/presets/{name}                  delete
    POST /api/presets/{name}/duplicate          duplicate
    GET  /api/presets/{name}/download           download yaml
    POST /api/presets/{name}/export             export to data_exports/
    POST /api/presets/import-from-data-exports  import from data_exports/
    POST /api/presets/import-from-path          import from absolute server path
    POST /api/presets/import                    upload + parse + schema validate
    *    /api/configs                           308 redirect → /api/presets （legacy）
    *    /api/configs/{rest:path}               308 redirect → /api/presets/{rest}（legacy）
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from .. import errors as _errors
from ..errors import _preset_err_code as _err_code
from ..schemas.presets import (
    DuplicateRequest,
    PresetExportBody,
    PresetImportBody,
    PresetImportFromPathBody,
)
from ...services.presets import io as presets_io
from ...paths import DATA_EXPORTS
from ...schema import GROUP_ORDER, TrainingConfig

router = APIRouter()


@router.get("/api/schema")
def get_schema() -> dict[str, Any]:
    """返回 TrainingConfig 的 JSON Schema + 分组顺序，前端据此渲染表单。"""
    return {
        "schema": TrainingConfig.model_json_schema(),
        "groups": [
            {"key": k, "label": label, "default_collapsed": dc}
            for k, label, dc in GROUP_ORDER
        ],
    }


@router.get("/api/presets")
def list_presets_endpoint() -> dict[str, Any]:
    return {"items": presets_io.list_presets()}


@router.get("/api/presets/{name}")
def get_preset(name: str) -> dict[str, Any]:
    try:
        return presets_io.read_preset(name)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope


@router.put("/api/presets/{name}")
def put_preset(name: str, body: dict[str, Any]) -> dict[str, str]:
    try:
        path = presets_io.write_preset(name, body)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    return {"name": name, "path": str(path)}


@router.delete("/api/presets/{name}")
def delete_preset_endpoint(name: str) -> dict[str, str]:
    try:
        presets_io.delete_preset(name)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    return {"deleted": name}


@router.post("/api/presets/{name}/duplicate")
def duplicate_preset_endpoint(name: str, body: DuplicateRequest) -> dict[str, str]:
    try:
        path = presets_io.duplicate_preset(name, body.new_name)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    return {"name": body.new_name, "path": str(path)}


@router.get("/api/presets/{name}/download")
def download_preset(name: str) -> FileResponse:
    """端到端文件 I/O：直接返回 `studio_data/presets/{name}.yaml` 原文件。"""
    try:
        path = presets_io.preset_path(name)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"预设不存在: {name}")
    return FileResponse(path, media_type="application/yaml", filename=f"{name}.yaml")


@router.post("/api/presets/{name}/export")
def export_preset_to_data_exports(name: str, body: PresetExportBody) -> dict[str, Any]:
    """把当前预设表单完整参数校验后保存到 data_exports/。"""
    DATA_EXPORTS.mkdir(parents=True, exist_ok=True)
    try:
        dest = _errors._unique_data_export_path(f"{name}.yaml", (".yaml", ".yml"))
        path = presets_io.write_preset(dest.stem, body.config, DATA_EXPORTS)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    return _errors._export_result(path)


@router.post("/api/presets/import-from-data-exports")
def import_preset_from_data_exports(body: PresetImportBody) -> dict[str, Any]:
    """从 data_exports/ 里的 yaml/json 预设导入到用户预设池。"""
    src = _errors._data_export_path(body.filename, (".yaml", ".yml", ".json"))
    if not src.exists():
        raise HTTPException(404, f"文件不存在: {body.filename}")
    if not src.is_file():
        raise HTTPException(400, "请选择文件")
    try:
        config, suggested = presets_io.parse_preset_bytes(src.read_bytes(), src.name)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    if presets_io.preset_path(suggested).exists():
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"预设已存在: {suggested}",
                "config": config,
                "suggested_name": suggested,
            },
        )
    try:
        path = presets_io.write_preset(suggested, config)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    return {"name": suggested, "path": str(path)}


@router.post("/api/presets/import-from-path")
def import_preset_from_path(body: PresetImportFromPathBody) -> dict[str, Any]:
    """从服务器绝对路径导入预设（yaml/yml/json）。"""
    from pathlib import Path
    src = Path(body.path)
    if not src.is_file():
        raise HTTPException(400, f"文件不存在或不可读: {body.path}")
    if src.suffix.lower() not in (".yaml", ".yml", ".json"):
        raise HTTPException(400, "请选择 .yaml / .yml / .json 文件")
    try:
        config, suggested = presets_io.parse_preset_bytes(src.read_bytes(), src.name)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    if presets_io.preset_path(suggested).exists():
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"预设已存在: {suggested}",
                "config": config,
                "suggested_name": suggested,
            },
        )
    try:
        path = presets_io.write_preset(suggested, config)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    return {"name": suggested, "path": str(path)}


@router.post("/api/presets/import")
async def import_preset(file: UploadFile = File(...)) -> dict[str, Any]:
    """接 .yaml/.yml/.json 上传 → 解析 + schema 校验 → 落盘到 `suggested_name`。

    无冲突 → write_preset 直接写,返回 200 `{name, path}`。
    冲突(`suggested_name.yaml` 已存在)→ 409 + 结构化 detail
    `{message, config, suggested_name}`,前端 ImportConflictDialog 让用户选
    覆盖 / 另存为 / 取消,选定后走 PUT /api/presets/{name} 完成落盘。
    解析/校验失败 → 400/422。
    """
    raw = await file.read()
    try:
        config, suggested = presets_io.parse_preset_bytes(raw, file.filename or "")
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    if presets_io.preset_path(suggested).exists():
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"预设已存在: {suggested}",
                "config": config,
                "suggested_name": suggested,
            },
        )
    try:
        path = presets_io.write_preset(suggested, config)
    except presets_io.PresetError as exc:
        _err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    return {"name": suggested, "path": str(path)}


# 旧 /api/configs/* 端点保留为 308 redirect（保护任何外部脚本）。
# 308 保持 method + body，所以 PUT/POST/DELETE 都能透明转发。
@router.api_route(
    "/api/configs",
    methods=["GET", "POST", "PUT", "DELETE"],
    include_in_schema=False,
)
def _configs_root_redirect(request: Request) -> RedirectResponse:
    qs = ("?" + request.url.query) if request.url.query else ""
    return RedirectResponse(url=f"/api/presets{qs}", status_code=308)


@router.api_route(
    "/api/configs/{rest:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
    include_in_schema=False,
)
def _configs_redirect(rest: str, request: Request) -> RedirectResponse:
    qs = ("?" + request.url.query) if request.url.query else ""
    return RedirectResponse(url=f"/api/presets/{rest}{qs}", status_code=308)
