"""Resource API for file-backed LLM Tagger presets (ADR 0017)."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, File, Header, Response, UploadFile
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from ...domain.errors import ConflictError, DomainError, NotFoundError, ValidationError
from ...infrastructure import config_store
from ...infrastructure import credentials
from ...infrastructure import llm_model_cache
from ...infrastructure import llm_preset_store as preset_store
from ...infrastructure import settings_store
from ...services import llm_presets as preset_service
from ...services.tagging import llm as llm_tagger_service

router = APIRouter()


class CreatePresetRequest(BaseModel):
    preset: dict[str, Any]
    credential_ref: str = ""


class DuplicatePresetRequest(BaseModel):
    label: Optional[str] = None


class SetDefaultPresetRequest(BaseModel):
    id: str


class RestorePresetRequest(BaseModel):
    backup_id: str


class LLMRefreshRequest(BaseModel):
    timeout: Optional[int] = Field(default=None, ge=1)


class LLMConnectionRequest(BaseModel):
    timeout: Optional[int] = Field(default=None, ge=1)


def _etag_header(value: Optional[str]) -> str:
    if not value:
        raise DomainError(
            "If-Match is required for this mutation",
            code="llm_preset.precondition_required",
            http_status=428,
        )
    result = value.strip()
    if result.startswith("W/"):
        result = result[2:]
    return result.strip('"')


def _set_etag(response: Response, etag: str) -> None:
    response.headers["ETag"] = f'"{etag}"'


def _raise_store_error(exc: Exception) -> None:
    if isinstance(exc, preset_store.LLMPresetNotFoundError):
        raise NotFoundError(
            str(exc), code="llm_preset.not_found"
        ) from exc
    if isinstance(exc, preset_store.LLMPresetConflictError):
        raise DomainError(
            str(exc),
            code="llm_preset.conflict",
            details={"current_etag": exc.current_etag},
            http_status=412 if exc.current_etag else 409,
        ) from exc
    if isinstance(exc, preset_store.LLMPresetInvalidError):
        raise ValidationError(
            str(exc), code="llm_preset.invalid", http_status=400
        ) from exc
    if isinstance(exc, credentials.CredentialNotFoundError):
        raise ValidationError(
            str(exc), code="credential.not_found", http_status=400
        ) from exc
    if isinstance(exc, credentials.CredentialStoreCorruptError):
        raise DomainError(
            str(exc), code="credential.store_corrupt", http_status=503
        ) from exc
    if isinstance(exc, settings_store.SettingsStoreCorruptError):
        raise DomainError(
            str(exc), code="settings.store_corrupt", http_status=503
        ) from exc
    raise exc


def _public(stored: preset_store.StoredLLMPreset) -> dict[str, Any]:
    return preset_service.present(stored)


def _project_legacy() -> None:
    config_store.project_current_best_effort()


@router.get("/api/llm-tagger/presets")
def list_llm_presets() -> dict[str, Any]:
    try:
        return preset_service.list_presets()
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")


@router.post("/api/llm-tagger/presets", status_code=201)
def create_llm_preset(body: CreatePresetRequest, response: Response) -> dict[str, Any]:
    try:
        stored = preset_service.create_preset(
            body.preset, credential_ref=body.credential_ref
        )
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    _project_legacy()
    _set_etag(response, stored.etag)
    response.headers["Location"] = f"/api/llm-tagger/presets/{stored.config.id}"
    return _public(stored)


@router.put("/api/llm-tagger/presets/default")
def set_default_llm_preset(body: SetDefaultPresetRequest) -> dict[str, str]:
    try:
        preset_id = preset_service.set_default_preset(body.id)
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    _project_legacy()
    return {"default_preset_id": preset_id}


@router.get("/api/llm-tagger/presets/{preset_id}")
def get_llm_preset(preset_id: str, response: Response) -> dict[str, Any]:
    try:
        stored = preset_store.get(preset_id)
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    _set_etag(response, stored.etag)
    return _public(stored)


@router.patch("/api/llm-tagger/presets/{preset_id}")
def patch_llm_preset(
    preset_id: str,
    body: dict[str, Any],
    response: Response,
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> dict[str, Any]:
    expected = _etag_header(if_match)
    try:
        stored = preset_service.update_preset(
            preset_id, body, expected_etag=expected
        )
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    _project_legacy()
    _set_etag(response, stored.etag)
    return _public(stored)


@router.delete("/api/llm-tagger/presets/{preset_id}")
def delete_llm_preset(
    preset_id: str,
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> dict[str, str]:
    expected = _etag_header(if_match)
    try:
        preset_service.delete_preset(preset_id, expected_etag=expected)
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    _project_legacy()
    return {"deleted": preset_id}


@router.post("/api/llm-tagger/presets/{preset_id}/duplicate", status_code=201)
def duplicate_llm_preset(
    preset_id: str, body: DuplicatePresetRequest, response: Response
) -> dict[str, Any]:
    try:
        stored = preset_store.duplicate(preset_id, label=body.label)
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    _project_legacy()
    _set_etag(response, stored.etag)
    response.headers["Location"] = f"/api/llm-tagger/presets/{stored.config.id}"
    return _public(stored)


@router.post("/api/llm-tagger/presets/{preset_id}/reset")
def reset_llm_preset(
    preset_id: str,
    response: Response,
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> dict[str, Any]:
    expected = _etag_header(if_match)
    try:
        stored = preset_store.reset_builtin(preset_id, expected_etag=expected)
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    _project_legacy()
    _set_etag(response, stored.etag)
    return _public(stored)


@router.post("/api/llm-tagger/presets/import", status_code=201)
async def import_llm_preset(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    fallback = (file.filename or "Imported").rsplit(".", 1)[0]
    try:
        stored = preset_store.import_portable(raw, fallback_label=fallback)
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    _project_legacy()
    return _public(stored)


@router.get("/api/llm-tagger/presets/{preset_id}/export")
def export_llm_preset(preset_id: str) -> FastAPIResponse:
    try:
        raw = preset_store.export_portable(preset_id)
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    return FastAPIResponse(
        content=raw,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="llm-preset-{preset_id}.json"'
        },
    )


@router.get("/api/llm-tagger/presets/{preset_id}/history")
def list_llm_preset_history(preset_id: str) -> dict[str, Any]:
    try:
        return {"items": preset_store.history(preset_id)}
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")


@router.post("/api/llm-tagger/presets/{preset_id}/history/restore")
def restore_llm_preset(
    preset_id: str,
    body: RestorePresetRequest,
    response: Response,
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> dict[str, Any]:
    expected = _etag_header(if_match)
    try:
        stored = preset_store.restore(
            preset_id, body.backup_id, expected_etag=expected
        )
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    _project_legacy()
    _set_etag(response, stored.etag)
    return _public(stored)


@router.post("/api/llm-tagger/presets/{preset_id}/models/refresh")
def refresh_llm_preset_models(
    preset_id: str, body: LLMRefreshRequest
) -> dict[str, Any]:
    try:
        stored, secret = preset_service.resolved_connection(preset_id)
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    config = stored.config
    if not config.base_url.strip():
        raise ValidationError(
            "API base URL is required",
            code="llm_tagger.base_url_required",
            http_status=400,
        )
    try:
        items = llm_tagger_service.fetch_openai_compatible_models(
            config.base_url,
            secret,
            timeout=body.timeout or config.timeout,
        )
    except Exception as exc:
        raise DomainError(
            f"Could not reach the model service: {exc}",
            code="llm_tagger.connect_failed",
            details={"reason": str(exc)},
            http_status=502,
        ) from exc
    # Discovered model IDs are disposable cache data, never preset content.
    llm_model_cache.save(
        stored.config.id,
        items,
        base_url=config.base_url,
        credential_ref=stored.credential_ref,
    )
    _project_legacy()
    return {"items": items, "preset_id": preset_id, "preset_etag": stored.etag}


@router.post("/api/llm-tagger/presets/{preset_id}/connection/test")
def test_llm_preset_connection(
    preset_id: str, body: LLMConnectionRequest
) -> dict[str, Any]:
    try:
        stored, secret = preset_service.resolved_connection(preset_id)
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    config = stored.config
    if not config.base_url.strip():
        raise ValidationError(
            "API base URL is required",
            code="llm_tagger.base_url_required",
            http_status=400,
        )
    if not config.model.strip():
        raise ValidationError(
            "A model must be selected",
            code="llm_tagger.model_required",
            http_status=400,
        )
    return llm_tagger_service.test_openai_compatible_connection(
        config.base_url,
        secret,
        config.model,
        endpoint=config.endpoint,
        timeout=body.timeout or config.timeout,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
