"""全局凭证 / 服务配置（PR-6 commit 2 从 server.py 抽出）。

6 routes：
    GET /api/secrets    masked secrets snapshot（API key 等敏感字段 masked）
    PUT /api/secrets    更新 secrets，返回新 masked snapshot
    GET /api/secrets/wandb/presets/{id}/export   下载 wandb preset yaml（含真实 api_key）
    POST /api/secrets/wandb/presets/import       导入 wandb preset（yaml/json 上传）
    GET /api/secrets/llm/presets/{id}/export     下载 llm preset json（不含 API 信息）
    POST /api/secrets/llm/presets/import         导入 llm preset（json/yaml 上传）
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import yaml
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from pydantic import ValidationError

from ... import secrets
from ...domain.errors import DomainError

logger = logging.getLogger(__name__)
router = APIRouter()


def _masked_snapshot() -> dict[str, Any]:
    result = secrets.to_masked_dict(secrets.load())
    from ...infrastructure.storage_layout import is_split_complete
    if not is_split_complete():
        return result
    from ...services import llm_presets as preset_service

    listing = preset_service.list_presets()
    presets: list[dict[str, Any]] = []
    for item in listing["items"]:
        public = dict(item)
        origin = str(public.get("origin", "custom"))
        configured = bool(public.get("credential_configured", False))
        public["builtin"] = origin != "custom"
        public["api_key"] = secrets.MASK if configured else ""
        presets.append(public)
    result["llm_tagger"] = {
        "current_preset": listing["default_preset_id"],
        "presets": presets,
    }
    return result


@router.get("/api/secrets")
def get_secrets() -> dict[str, Any]:
    return _masked_snapshot()


@router.put("/api/secrets")
def put_secrets(body: dict[str, Any]) -> dict[str, Any]:
    from ...infrastructure.storage_layout import is_split_complete
    if is_split_complete() and "llm_tagger" in body:
        from ...services import llm_presets as preset_service

        body = dict(body)
        llm_patch = body.pop("llm_tagger")
        if not isinstance(llm_patch, dict):
            raise DomainError(
                "llm_tagger patch must be an object",
                code="llm_preset.invalid_patch",
                http_status=400,
            )
        if "presets" in llm_patch:
            raise DomainError(
                "LLM presets must be mutated through /api/llm-tagger/presets",
                code="llm_preset.legacy_mutation_disabled",
                http_status=409,
            )
        unknown = set(llm_patch) - {"current_preset"}
        if unknown:
            raise DomainError(
                f"Unknown LLM settings: {sorted(unknown)}",
                code="llm_preset.invalid_patch",
                http_status=400,
            )
        if "current_preset" in llm_patch:
            preset_service.set_default_preset(str(llm_patch["current_preset"]))
    new = secrets.update(body) if body else secrets.load()
    # 用户在 Settings 里改了 generate.idle_timeout_minutes 后，立即同步给跑着的
    # daemon —— 不然要等下次出图 dispatch 才生效。daemon 还没起的话 set 也安全
    # （走 noop 分支，下次 dispatch 时一并应用）。
    try:
        from ...services.inference.daemon import get_daemon
        get_daemon().sync_idle_timeout_from_secrets()
    except Exception:
        logger.warning(
            "failed to sync idle_timeout to daemon; the daemon keeps the previous value",
            exc_info=True,
        )
    return _masked_snapshot()


@router.get("/api/secrets/wandb/presets/{preset_id}/export")
def export_wandb_preset(preset_id: str) -> Response:
    """下载单个 wandb preset 的 yaml。**包含真实 api_key**（用户显式导出动作，
    绕过 GET /api/secrets 的 mask）——文件自行保管。"""
    preset = secrets.get_wandb_preset(preset_id)
    if preset is None:
        raise DomainError(
            f"WandB preset {preset_id!r} not found",
            code="secrets.wandb_preset_not_found",
            details={"id": preset_id},
            http_status=404,
        )
    text = yaml.safe_dump(
        preset.model_dump(), allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    # preset.id 经 validator 归一为 [A-Za-z0-9_-]，直接进 filename 安全
    return Response(
        content=text,
        media_type="application/yaml",
        headers={
            "Content-Disposition": f'attachment; filename="wandb-preset-{preset.id}.yaml"'
        },
    )


@router.post("/api/secrets/wandb/presets/import")
async def import_wandb_preset(file: UploadFile = File(...)) -> dict[str, Any]:
    """上传 yaml/json 导入 wandb preset；撞名自动加后缀并切换为当前选中。"""
    raw = await file.read()
    try:
        data = yaml.safe_load(raw.decode("utf-8"))  # yaml 是 json 的 superset
    except Exception as exc:
        raise DomainError(
            f"WandB preset file could not be parsed: {exc}",
            code="secrets.wandb_preset_invalid",
            http_status=400,
        ) from exc
    stem = re.sub(r"\.(ya?ml|json)$", "", file.filename or "", flags=re.I)
    try:
        new, preset = secrets.import_wandb_preset(data, fallback_label=stem)
    except (ValueError, ValidationError) as exc:
        raise DomainError(
            f"WandB preset is invalid: {exc}",
            code="secrets.wandb_preset_invalid",
            http_status=400,
        ) from exc
    return {
        "id": preset.id,
        "label": preset.label,
        "secrets": secrets.to_masked_dict(new),
    }


@router.get("/api/secrets/llm/presets/{preset_id}/export")
def export_llm_preset(preset_id: str) -> Response:
    """Compatibility alias for the canonical secret-free preset export."""
    from ...infrastructure.storage_layout import is_split_complete
    if is_split_complete():
        from ...infrastructure import llm_preset_store
        try:
            text = llm_preset_store.export_portable(preset_id).decode("utf-8")
        except llm_preset_store.LLMPresetNotFoundError as exc:
            raise DomainError(
                f"LLM preset {preset_id!r} not found",
                code="secrets.llm_preset_not_found",
                details={"id": preset_id},
                http_status=404,
            ) from exc
        filename_id = preset_id
    else:
        data = secrets.export_llm_preset(preset_id)
        if data is None:
            raise DomainError(
                f"LLM preset {preset_id!r} not found",
                code="secrets.llm_preset_not_found",
                details={"id": preset_id},
                http_status=404,
            )
        text = json.dumps(data, ensure_ascii=False, indent=2)
        filename_id = data["id"]
    return Response(
        content=text,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="llm-preset-{filename_id}.json"'
        },
    )


@router.post("/api/secrets/llm/presets/import")
async def import_llm_preset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Compatibility alias; new storage accepts the canonical JSON format."""
    raw = await file.read()
    stem = re.sub(r"\.(ya?ml|json)$", "", file.filename or "", flags=re.I)
    from ...infrastructure.storage_layout import is_split_complete
    if is_split_complete():
        from ...infrastructure import llm_preset_store
        try:
            preset = llm_preset_store.import_portable(raw, fallback_label=stem)
        except (ValueError, llm_preset_store.LLMPresetInvalidError) as exc:
            raise DomainError(
                f"LLM preset is invalid: {exc}",
                code="secrets.llm_preset_invalid",
                http_status=400,
            ) from exc
        return {
            "id": preset.config.id,
            "label": preset.config.label,
            "secrets": _masked_snapshot(),
        }

    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except Exception as exc:
        raise DomainError(
            f"LLM preset file could not be parsed: {exc}",
            code="secrets.llm_preset_invalid",
            http_status=400,
        ) from exc
    try:
        new, preset = secrets.import_llm_preset(data, fallback_label=stem)
    except (ValueError, ValidationError) as exc:
        raise DomainError(
            f"LLM preset is invalid: {exc}",
            code="secrets.llm_preset_invalid",
            http_status=400,
        ) from exc
    return {
        "id": preset.id,
        "label": preset.label,
        "secrets": secrets.to_masked_dict(new),
    }
