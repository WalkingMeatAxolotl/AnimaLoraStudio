"""Write-only credential metadata and secret mutation API (ADR 0017)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from ...domain.errors import ConflictError, DomainError, NotFoundError, ValidationError
from ...infrastructure import config_store
from ...infrastructure import credentials
from ...services import llm_presets as preset_service

router = APIRouter()


class CreateCredentialRequest(BaseModel):
    label: str
    kind: Literal["api_key", "token"] = "api_key"
    secret: str = Field(default="", repr=False)


class UpdateCredentialRequest(BaseModel):
    label: str


class ReplaceCredentialSecretRequest(BaseModel):
    secret: str = Field(repr=False)


def _etag_header(value: Optional[str], *, required: bool = True) -> Optional[str]:
    if not value:
        if not required:
            return None
        raise DomainError(
            "If-Match is required for this mutation",
            code="credential.precondition_required",
            http_status=428,
        )
    result = value.strip()
    if result.startswith("W/"):
        result = result[2:]
    return result.strip('"')


def _raise_store_error(exc: Exception) -> None:
    if isinstance(exc, credentials.CredentialNotFoundError):
        raise NotFoundError(str(exc), code="credential.not_found") from exc
    if isinstance(exc, credentials.CredentialConflictError):
        raise DomainError(
            str(exc),
            code="credential.conflict",
            details={"current_etag": exc.current_etag} if exc.current_etag else {},
            http_status=412 if exc.current_etag else 409,
        ) from exc
    if isinstance(exc, credentials.CredentialStoreCorruptError):
        raise DomainError(
            str(exc), code="credential.store_corrupt", http_status=503
        ) from exc
    if isinstance(exc, preset_service.CredentialReferenceError):
        raise ConflictError(
            str(exc),
            code="credential.in_use",
            details={"referenced_by": exc.referenced_by},
        ) from exc
    if isinstance(exc, ValueError):
        raise ValidationError(
            str(exc), code="credential.invalid", http_status=400
        ) from exc
    raise exc


@router.get("/api/credentials")
def list_credentials() -> dict[str, Any]:
    try:
        items = credentials.list_metadata()
        for item in items:
            item["referenced_by"] = preset_service.credential_references(item["id"])
        return {"items": items}
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")


@router.post("/api/credentials", status_code=201)
def create_credential(body: CreateCredentialRequest) -> dict[str, Any]:
    try:
        return credentials.create(
            label=body.label, kind=body.kind, secret=body.secret
        )
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")


@router.patch("/api/credentials/{credential_id}")
def patch_credential(
    credential_id: str,
    body: UpdateCredentialRequest,
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> dict[str, Any]:
    try:
        return credentials.update_metadata(
            credential_id,
            label=body.label,
            expected_etag=_etag_header(if_match),
        )
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")


@router.put("/api/credentials/{credential_id}/secret")
def replace_credential_secret(
    credential_id: str,
    body: ReplaceCredentialSecretRequest,
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> dict[str, Any]:
    try:
        result = credentials.replace_secret(
            credential_id,
            body.secret,
            expected_etag=_etag_header(if_match),
        )
        config_store.project_current_best_effort()
        return result
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")


@router.delete("/api/credentials/{credential_id}")
def delete_credential(
    credential_id: str,
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> dict[str, str]:
    try:
        preset_service.delete_credential(
            credential_id,
            expected_etag=_etag_header(if_match),
        )
    except Exception as exc:
        _raise_store_error(exc)
        raise AssertionError("unreachable")
    return {"deleted": credential_id}
