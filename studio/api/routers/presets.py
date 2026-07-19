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

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from .. import errors as _errors
from ...domain.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
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
    schema = TrainingConfig.model_json_schema()
    _apply_feature_flags(schema)
    return {
        "schema": schema,
        "groups": [
            {"key": k, "label": label, "default_collapsed": dc}
            for k, label, dc in GROUP_ORDER
        ],
    }


@router.post("/api/schema/preview-yaml")
def preview_config_yaml_endpoint(body: dict[str, Any]) -> dict[str, Any]:
    """当前表单 config → 与保存后落盘文件同一序列化路径的 yaml 文本(R4)。

    取代前端 pruneInactiveConfig + configToYaml 双镜像:预览不再「声称一致」
    而是物理一致。纯计算不落盘;tolerant 修复语义与保存一致。
    """
    return {"yaml": presets_io.preview_config_yaml_text(dict(body.get("config") or {}))}


def _apply_feature_flags(schema: dict[str, Any]) -> None:
    """按 SystemConfig 的实验性 flag 动态调整 schema（只影响 UI 渲染）。

    Automagic v2 未正式发布：flag 关闭时给 automagic_variant 打 hidden（值仍
    透传，CLI/yaml 不受影响）。agreement_threshold 的 show_when 含
    automagic_variant==v2，variant 隐藏后停在默认 v1 自然不显示，无需处理。
    启用方式：手改 studio_data/secrets.json 的 system.enable_automagic_v2，
    Settings 页故意不渲染该开关。
    """
    from ...infrastructure import secrets as secrets_infra

    if not secrets_infra.load().system.enable_automagic_v2:
        props = schema.get("properties", {})
        if "automagic_variant" in props:
            props["automagic_variant"]["hidden"] = True


@router.get("/api/presets")
def list_presets_endpoint() -> dict[str, Any]:
    return {"items": presets_io.list_presets()}


@router.get("/api/presets/{name}")
def get_preset(name: str, warnings: bool = False) -> dict[str, Any]:
    if warnings:
        config, dropped, defaulted = presets_io.read_preset_with_warnings(name)
        return {
            "config": config,
            "dropped_fields": dropped,
            "defaulted_fields": defaulted,
        }
    return presets_io.read_preset(name)


@router.put("/api/presets/{name}")
def put_preset(name: str, body: dict[str, Any]) -> dict[str, str]:
    path = presets_io.write_preset(name, body)
    return {"name": name, "path": str(path)}


@router.delete("/api/presets/{name}")
def delete_preset_endpoint(name: str) -> dict[str, str]:
    presets_io.delete_preset(name)
    return {"deleted": name}


@router.post("/api/presets/{name}/duplicate")
def duplicate_preset_endpoint(name: str, body: DuplicateRequest) -> dict[str, str]:
    path = presets_io.duplicate_preset(name, body.new_name)
    return {"name": body.new_name, "path": str(path)}


@router.get("/api/presets/{name}/download")
def download_preset(name: str) -> FileResponse:
    """端到端文件 I/O：直接返回 `studio_data/presets/{name}.yaml` 原文件。"""
    path = presets_io.preset_path(name)
    if not path.exists():
        raise NotFoundError(
            f'Preset "{name}" not found',
            code="preset.not_found", details={"name": name},
        )
    return FileResponse(path, media_type="application/yaml", filename=f"{name}.yaml")


@router.post("/api/presets/{name}/export")
def export_preset_to_data_exports(name: str, body: PresetExportBody) -> dict[str, Any]:
    """把当前预设表单完整参数校验后保存到 data_exports/。"""
    DATA_EXPORTS.mkdir(parents=True, exist_ok=True)
    dest = _errors._unique_data_export_path(f"{name}.yaml", (".yaml", ".yml"))
    path = presets_io.write_preset(dest.stem, body.config, DATA_EXPORTS)
    return _errors._export_result(path)


@router.post("/api/presets/import-from-data-exports")
def import_preset_from_data_exports(body: PresetImportBody) -> dict[str, Any]:
    """从 data_exports/ 里的 yaml/json 预设导入到用户预设池。"""
    src = _errors._data_export_path(body.filename, (".yaml", ".yml", ".json"))
    if not src.exists():
        raise NotFoundError(
            f'File "{body.filename}" not found',
            code="file.not_found", details={"filename": body.filename},
        )
    if not src.is_file():
        raise ValidationError(
            "Select a file", code="file.required", http_status=400,
        )
    config, suggested = presets_io.parse_preset_bytes(src.read_bytes(), src.name)
    if presets_io.preset_path(suggested).exists():
        raise ConflictError(
            f'Preset "{suggested}" already exists',
            code="preset.exists",
            details={
                "name": suggested,
                "config": config,
                "suggested_name": suggested,
            },
        )
    path = presets_io.write_preset(suggested, config)
    return {"name": suggested, "path": str(path)}


@router.post("/api/presets/import-from-path")
def import_preset_from_path(body: PresetImportFromPathBody) -> dict[str, Any]:
    """从服务器绝对路径导入预设（yaml/yml/json）。"""
    from pathlib import Path
    src = Path(body.path)
    if not src.is_file():
        raise NotFoundError(
            f'File "{src.name}" not found or not readable',
            code="file.not_found",
            details={"filename": src.name, "path": body.path},
            http_status=400,
        )
    if src.suffix.lower() not in (".yaml", ".yml", ".json"):
        raise ValidationError(
            "Select a .yaml, .yml, or .json file",
            code="file.ext_invalid",
            details={"types": ".yaml, .yml, .json"},
            http_status=400,
        )
    config, suggested = presets_io.parse_preset_bytes(src.read_bytes(), src.name)
    if presets_io.preset_path(suggested).exists():
        raise ConflictError(
            f'Preset "{suggested}" already exists',
            code="preset.exists",
            details={
                "name": suggested,
                "config": config,
                "suggested_name": suggested,
            },
        )
    path = presets_io.write_preset(suggested, config)
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
    config, suggested = presets_io.parse_preset_bytes(raw, file.filename or "")
    if presets_io.preset_path(suggested).exists():
        raise ConflictError(
            f'Preset "{suggested}" already exists',
            code="preset.exists",
            details={
                "name": suggested,
                "config": config,
                "suggested_name": suggested,
            },
        )
    path = presets_io.write_preset(suggested, config)
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
