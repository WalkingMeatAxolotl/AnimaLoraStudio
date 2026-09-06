"""Write-only local credential store.

This is deliberately separate from settings, presets, and ``studio.db``. It is
plaintext at rest because ADR 0017 rejects OS keyrings and fake same-host
"encryption"; API layers must never serialize ``secret`` back to clients.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .atomic_files import (
    InvalidExistingFileError,
    atomic_write_text,
    recover_latest_valid_backup,
)
from .paths import STUDIO_DATA

CREDENTIALS_FILE = STUDIO_DATA / "credentials.json"
_CREDENTIALS_LOCK = threading.RLock()
_CREDENTIALS_BACKUP_COUNT = 3
_CREDENTIAL_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{2,95}$")
_LOGGER = logging.getLogger(__name__)


class CredentialStoreError(RuntimeError):
    """Base class for credential-store failures."""


class CredentialStoreCorruptError(CredentialStoreError):
    """The primary credential document has no valid recovery source."""


class CredentialNotFoundError(CredentialStoreError):
    """A requested credential ID does not exist."""


class CredentialConflictError(CredentialStoreError):
    """A credential ID or ETag conflicts with current state."""

    def __init__(self, message: str, *, current_etag: str = "") -> None:
        super().__init__(message)
        self.current_etag = current_etag


class CredentialRecord(BaseModel):
    id: str
    kind: Literal["api_key", "token"] = "api_key"
    label: str = ""
    secret: str = Field(default="", repr=False)

    @model_validator(mode="after")
    def _validate_identity(self) -> "CredentialRecord":
        self.id = str(self.id or "").strip().lower()
        if not _CREDENTIAL_ID_RE.fullmatch(self.id):
            raise ValueError(
                "credential id must match [a-z][a-z0-9_-]{2,95}"
            )
        self.label = str(self.label or self.id).strip() or self.id
        self.secret = str(self.secret or "")
        return self


class CredentialsDocument(BaseModel):
    kind: Literal["anima-credentials"] = "anima-credentials"
    schema_version: Literal[1] = 1
    items: dict[str, CredentialRecord] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_index(self) -> "CredentialsDocument":
        normalized: dict[str, CredentialRecord] = {}
        for key, record in self.items.items():
            normalized_key = str(key).strip().lower()
            if normalized_key != record.id:
                raise ValueError(
                    f"credential index {key!r} does not match record id {record.id!r}"
                )
            if normalized_key in normalized:
                raise ValueError(f"duplicate credential id: {normalized_key}")
            normalized[normalized_key] = record
        self.items = normalized
        return self


def _backup_dir() -> Path:
    return CREDENTIALS_FILE.parent / "backups" / "credentials"


def _decode(raw: bytes) -> CredentialsDocument:
    return CredentialsDocument.model_validate_json(raw)


def _validate(raw: bytes) -> None:
    _decode(raw)


def _etag(record: CredentialRecord) -> str:
    payload = record.model_dump_json().encode("utf-8")
    return f'sha256:{hashlib.sha256(payload).hexdigest()}'


def _metadata(record: CredentialRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "kind": record.kind,
        "label": record.label,
        "configured": bool(record.secret),
        "etag": _etag(record),
    }


def _load_unlocked() -> CredentialsDocument:
    if not CREDENTIALS_FILE.exists():
        return CredentialsDocument()
    raw = CREDENTIALS_FILE.read_bytes()
    try:
        return _decode(raw)
    except Exception as original_exc:
        try:
            recovered = recover_latest_valid_backup(
                CREDENTIALS_FILE,
                _backup_dir(),
                validate=_validate,
                mode=0o600,
            )
        except Exception as recovery_exc:
            raise CredentialStoreCorruptError(
                f"Could not read or recover {CREDENTIALS_FILE}"
            ) from recovery_exc
        if recovered is None:
            raise CredentialStoreCorruptError(
                f"Existing credential store is invalid and no valid backup exists: "
                f"{CREDENTIALS_FILE}"
            ) from original_exc
        _LOGGER.error(
            "Recovered invalid credential store from backup %s; corrupt bytes kept at %s",
            recovered.backup_path,
            recovered.corrupt_path,
        )
        return _decode(recovered.data)


def _save_unlocked(document: CredentialsDocument) -> None:
    try:
        atomic_write_text(
            CREDENTIALS_FILE,
            document.model_dump_json(indent=2),
            backup_dir=_backup_dir(),
            keep_backups=_CREDENTIALS_BACKUP_COUNT,
            validate_existing=_validate,
            mode=0o600,
        )
    except InvalidExistingFileError as exc:
        raise CredentialStoreCorruptError(
            f"Refusing to overwrite invalid credential store: {CREDENTIALS_FILE}"
        ) from exc


def load() -> CredentialsDocument:
    with _CREDENTIALS_LOCK:
        return _load_unlocked()


def list_metadata() -> list[dict[str, Any]]:
    with _CREDENTIALS_LOCK:
        document = _load_unlocked()
        return [_metadata(item) for item in document.items.values()]


def resolve(credential_id: str) -> str:
    with _CREDENTIALS_LOCK:
        record = _load_unlocked().items.get(credential_id.strip().lower())
        if record is None:
            raise CredentialNotFoundError(f"Credential not found: {credential_id}")
        return record.secret


def put(
    credential_id: str,
    *,
    label: str,
    secret: str,
    kind: Literal["api_key", "token"] = "api_key",
) -> dict[str, Any]:
    """Create or replace a well-known credential without exposing its value."""
    with _CREDENTIALS_LOCK:
        document = _load_unlocked()
        cid = credential_id.strip().lower()
        record = CredentialRecord(
            id=cid,
            kind=kind,
            label=label,
            secret=secret,
        )
        if document.items.get(cid) == record:
            return _metadata(record)
        document.items[cid] = record
        _save_unlocked(document)
        return _metadata(record)


def create(
    *,
    label: str,
    secret: str = "",
    kind: Literal["api_key", "token"] = "api_key",
    credential_id: Optional[str] = None,
) -> dict[str, Any]:
    with _CREDENTIALS_LOCK:
        document = _load_unlocked()
        cid = (credential_id or f"cred_{uuid.uuid4().hex[:16]}").strip().lower()
        if cid in document.items:
            raise CredentialConflictError(f"Credential already exists: {cid}")
        record = CredentialRecord(id=cid, kind=kind, label=label, secret=secret)
        document.items[cid] = record
        _save_unlocked(document)
        return _metadata(record)


def replace_secret(
    credential_id: str,
    secret: str,
    *,
    expected_etag: Optional[str] = None,
) -> dict[str, Any]:
    with _CREDENTIALS_LOCK:
        document = _load_unlocked()
        cid = credential_id.strip().lower()
        record = document.items.get(cid)
        if record is None:
            raise CredentialNotFoundError(f"Credential not found: {credential_id}")
        if expected_etag is not None and expected_etag != _etag(record):
            raise CredentialConflictError(
                f"Credential changed: {credential_id}", current_etag=_etag(record)
            )
        updated = record.model_copy(update={"secret": str(secret or "")})
        document.items[cid] = CredentialRecord.model_validate(updated.model_dump())
        _save_unlocked(document)
        return _metadata(document.items[cid])


def update_metadata(
    credential_id: str,
    *,
    label: Optional[str] = None,
    expected_etag: Optional[str] = None,
) -> dict[str, Any]:
    with _CREDENTIALS_LOCK:
        document = _load_unlocked()
        cid = credential_id.strip().lower()
        record = document.items.get(cid)
        if record is None:
            raise CredentialNotFoundError(f"Credential not found: {credential_id}")
        if expected_etag is not None and expected_etag != _etag(record):
            raise CredentialConflictError(
                f"Credential changed: {credential_id}", current_etag=_etag(record)
            )
        changes = {"label": label} if label is not None else {}
        updated = CredentialRecord.model_validate(
            record.model_copy(update=changes).model_dump()
        )
        document.items[cid] = updated
        _save_unlocked(document)
        return _metadata(updated)


def delete(credential_id: str, *, expected_etag: Optional[str] = None) -> None:
    """Delete an unreferenced credential.

    Reference checks belong to the service/API layer because active tasks and
    preset repositories are outside this store.
    """
    with _CREDENTIALS_LOCK:
        document = _load_unlocked()
        cid = credential_id.strip().lower()
        record = document.items.get(cid)
        if record is None:
            raise CredentialNotFoundError(f"Credential not found: {credential_id}")
        if expected_etag is not None and expected_etag != _etag(record):
            raise CredentialConflictError(
                f"Credential changed: {credential_id}", current_etag=_etag(record)
            )
        del document.items[cid]
        _save_unlocked(document)


def import_records(records: list[CredentialRecord]) -> CredentialsDocument:
    """Idempotently install migration records without replacing conflicts."""
    with _CREDENTIALS_LOCK:
        document = _load_unlocked()
        changed = False
        for record in records:
            existing = document.items.get(record.id)
            if existing is not None:
                if existing != record:
                    raise CredentialConflictError(
                        f"Migration credential conflicts with existing record: {record.id}"
                    )
                continue
            document.items[record.id] = record
            changed = True
        if changed:
            _save_unlocked(document)
        return document
