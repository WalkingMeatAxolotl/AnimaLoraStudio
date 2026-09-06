"""Versioned non-secret application settings document introduced by ADR 0017.

``values`` retains the public settings shape used by the Studio API while LLM
preset documents and credentials live in their dedicated stores.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .atomic_files import (
    InvalidExistingFileError,
    atomic_write_text,
    recover_latest_valid_backup,
)
from .paths import STUDIO_DATA

SETTINGS_FILE = STUDIO_DATA / "settings.json"
_SETTINGS_LOCK = threading.RLock()
_SETTINGS_BACKUP_COUNT = 10


class SettingsStoreError(RuntimeError):
    pass


class SettingsStoreCorruptError(SettingsStoreError):
    pass


class AppSettingsDocument(BaseModel):
    kind: Literal["anima-settings"] = "anima-settings"
    schema_version: Literal[1] = 1
    values: dict[str, Any] = Field(default_factory=dict)


def _backup_dir() -> Path:
    return SETTINGS_FILE.parent / "backups" / "settings"


def _decode(raw: bytes) -> AppSettingsDocument:
    return AppSettingsDocument.model_validate_json(raw)


def _validate(raw: bytes) -> None:
    _decode(raw)


def _load_unlocked() -> AppSettingsDocument:
    if not SETTINGS_FILE.exists():
        return AppSettingsDocument()
    raw = SETTINGS_FILE.read_bytes()
    try:
        return _decode(raw)
    except Exception as original_exc:
        try:
            recovered = recover_latest_valid_backup(
                SETTINGS_FILE,
                _backup_dir(),
                validate=_validate,
            )
        except Exception as recovery_exc:
            raise SettingsStoreCorruptError(
                f"Could not read or recover settings store: {SETTINGS_FILE}"
            ) from recovery_exc
        if recovered is None:
            raise SettingsStoreCorruptError(
                f"Existing settings store is invalid and no valid backup exists: "
                f"{SETTINGS_FILE}"
            ) from original_exc
        return _decode(recovered.data)


def _save_unlocked(document: AppSettingsDocument) -> None:
    try:
        atomic_write_text(
            SETTINGS_FILE,
            document.model_dump_json(indent=2),
            backup_dir=_backup_dir(),
            keep_backups=_SETTINGS_BACKUP_COUNT,
            validate_existing=_validate,
        )
    except InvalidExistingFileError as exc:
        raise SettingsStoreCorruptError(
            f"Refusing to overwrite invalid settings store: {SETTINGS_FILE}"
        ) from exc


def load() -> AppSettingsDocument:
    with _SETTINGS_LOCK:
        return _load_unlocked()


def save(document: AppSettingsDocument) -> AppSettingsDocument:
    with _SETTINGS_LOCK:
        _save_unlocked(document)
        return document


def install_migrated(document: AppSettingsDocument) -> AppSettingsDocument:
    """Install an exact migration result, idempotently."""
    with _SETTINGS_LOCK:
        if SETTINGS_FILE.exists():
            current = _load_unlocked()
            if current != document:
                raise SettingsStoreError(
                    "Migration settings conflict with an existing settings document"
                )
            return current
        _save_unlocked(document)
        return document


def get_default_llm_preset_id() -> str:
    with _SETTINGS_LOCK:
        document = _load_unlocked()
        llm = document.values.get("llm_tagger")
        if not isinstance(llm, dict):
            return "style_json"
        return str(llm.get("default_preset_id") or "style_json")


def set_default_llm_preset_id(preset_id: str) -> AppSettingsDocument:
    with _SETTINGS_LOCK:
        document = _load_unlocked()
        values = dict(document.values)
        llm = dict(values.get("llm_tagger") or {})
        llm["default_preset_id"] = str(preset_id)
        values["llm_tagger"] = llm
        updated = AppSettingsDocument(values=values)
        _save_unlocked(updated)
        return updated
