"""One-file-per-entity persistence for LLM Tagger presets."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .atomic_files import (
    InvalidExistingFileError,
    archive_file,
    atomic_write_text,
    list_backups,
    replace_invalid_bytes,
)
from .llm_presets import BUILTIN_PRESET_ORDER, builtin_llm_presets
from .paths import STUDIO_DATA
from .secrets import LLMPresetConfig

LLM_PRESETS_DIR = STUDIO_DATA / "llm_presets"
_LLM_PRESETS_LOCK = threading.RLock()
_PRESET_BACKUP_COUNT = 10
_PRESET_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{2,95}$")
_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul", "clock$",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_KIND = "anima-llm-preset"
_SCHEMA_VERSION = 1
_FILE_METADATA = {"kind", "schema_version", "credential_ref"}
_FORBIDDEN_PATCH_FIELDS = {"id", "builtin", "api_key", "model_ids"}


class LLMPresetStoreError(RuntimeError):
    """Base class for LLM preset repository failures."""


class LLMPresetNotFoundError(LLMPresetStoreError):
    pass


class LLMPresetConflictError(LLMPresetStoreError):
    def __init__(self, message: str, *, current_etag: str = "") -> None:
        super().__init__(message)
        self.current_etag = current_etag


class LLMPresetInvalidError(LLMPresetStoreError):
    pass


@dataclass(frozen=True)
class StoredLLMPreset:
    config: LLMPresetConfig
    credential_ref: str
    origin: str
    etag: str
    updated_at: Optional[float]

    def public_dict(self) -> dict[str, Any]:
        data = self.config.model_dump()
        data.pop("api_key", None)
        return {
            **data,
            "credential_ref": self.credential_ref,
            "origin": self.origin,
            "etag": self.etag,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class InvalidPresetFile:
    id: str
    path: str
    error: str
    etag: str

    def public_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "path": self.path,
            "error": self.error,
            "etag": self.etag,
        }


def _preset_id(value: str) -> str:
    result = str(value or "").strip().lower()
    if not _PRESET_ID_RE.fullmatch(result) or result.rstrip("_") != result:
        raise LLMPresetInvalidError(
            "preset id must match [a-z][a-z0-9_-]{2,95} and not end with '_'"
        )
    if result in _WINDOWS_RESERVED_NAMES:
        raise LLMPresetInvalidError(f"preset id is reserved on Windows: {result}")
    return result


def validate_preset_id(value: str) -> str:
    """Return the canonical cross-platform ID or raise without changing identity."""
    return _preset_id(value)


def _path(preset_id: str) -> Path:
    return LLM_PRESETS_DIR / f"{_preset_id(preset_id)}.json"


def _backup_dir(preset_id: str) -> Path:
    return LLM_PRESETS_DIR.parent / "backups" / "llm_presets" / _preset_id(preset_id)


def _config_payload(config: LLMPresetConfig) -> dict[str, Any]:
    data = config.model_dump()
    for field in ("api_key", "model_ids", "builtin"):
        data.pop(field, None)
    return data


def _document_bytes(config: LLMPresetConfig, credential_ref: str) -> bytes:
    payload = {
        "kind": _KIND,
        "schema_version": _SCHEMA_VERSION,
        **_config_payload(config),
        "credential_ref": str(credential_ref or "").strip().lower(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _etag(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _decode(raw: bytes, *, expected_id: Optional[str] = None) -> tuple[LLMPresetConfig, str]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise LLMPresetInvalidError(f"preset is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMPresetInvalidError("preset document must be an object")
    if payload.get("kind") != _KIND:
        raise LLMPresetInvalidError(f"preset kind must be {_KIND!r}")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise LLMPresetInvalidError(
            f"unsupported preset schema_version: {payload.get('schema_version')!r}"
        )
    raw_id = _preset_id(str(payload.get("id") or ""))
    if expected_id is not None and raw_id != _preset_id(expected_id):
        raise LLMPresetInvalidError(
            f"preset id {raw_id!r} does not match filename {expected_id!r}"
        )
    credential_ref = str(payload.get("credential_ref") or "").strip().lower()
    config_payload = {
        key: value for key, value in payload.items() if key not in _FILE_METADATA
    }
    config_payload["id"] = raw_id
    config_payload["api_key"] = ""
    config_payload["model_ids"] = []
    try:
        config = LLMPresetConfig.model_validate(config_payload)
    except Exception as exc:
        raise LLMPresetInvalidError(f"preset schema is invalid: {exc}") from exc
    if config.id != raw_id:
        raise LLMPresetInvalidError("preset id normalization changed its identity")
    return config, credential_ref


def _validate_for(preset_id: str):
    def _validate(raw: bytes) -> None:
        _decode(raw, expected_id=preset_id)

    return _validate


def _builtin_map() -> dict[str, LLMPresetConfig]:
    return {
        item["id"]: LLMPresetConfig.model_validate(
            {**item, "api_key": "", "model_ids": []}
        )
        for item in builtin_llm_presets()
    }


def _builtin_stored(preset_id: str, config: LLMPresetConfig) -> StoredLLMPreset:
    config = config.model_copy(deep=True)
    config.builtin = True
    raw = _document_bytes(config, "")
    return StoredLLMPreset(
        config=config,
        credential_ref="",
        origin="builtin",
        etag=_etag(raw),
        updated_at=None,
    )


def _read_user_unlocked(path: Path, preset_id: str) -> StoredLLMPreset:
    # Reads are side-effect free. Invalid files remain available for explicit
    # history selection/restore instead of silently rolling back on GET.
    raw = path.read_bytes()
    config, credential_ref = _decode(raw, expected_id=preset_id)
    config.builtin = preset_id in _builtin_map()
    return StoredLLMPreset(
        config=config,
        credential_ref=credential_ref,
        origin="builtin_override" if config.builtin else "custom",
        etag=_etag(raw),
        updated_at=path.stat().st_mtime,
    )


def _get_unlocked(preset_id: str) -> StoredLLMPreset:
    pid = _preset_id(preset_id)
    path = _path(pid)
    if path.exists():
        return _read_user_unlocked(path, pid)
    builtin = _builtin_map().get(pid)
    if builtin is not None:
        return _builtin_stored(pid, builtin)
    raise LLMPresetNotFoundError(f"LLM preset not found: {pid}")


def get(preset_id: str) -> StoredLLMPreset:
    with _LLM_PRESETS_LOCK:
        return _get_unlocked(preset_id)


def list_all() -> tuple[list[StoredLLMPreset], list[InvalidPresetFile]]:
    with _LLM_PRESETS_LOCK:
        builtins = _builtin_map()
        items: list[StoredLLMPreset] = []
        invalid: list[InvalidPresetFile] = []
        seen: set[str] = set()
        for preset_id in BUILTIN_PRESET_ORDER:
            path = _path(preset_id)
            if path.exists():
                try:
                    items.append(_read_user_unlocked(path, preset_id))
                except Exception as exc:
                    invalid.append(
                        InvalidPresetFile(
                            preset_id, str(path), str(exc), _etag(path.read_bytes())
                        )
                    )
            elif preset_id in builtins:
                items.append(_builtin_stored(preset_id, builtins[preset_id]))
            seen.add(preset_id.casefold())

        if LLM_PRESETS_DIR.exists():
            custom_paths = sorted(
                LLM_PRESETS_DIR.glob("*.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            for path in custom_paths:
                pid = path.stem.lower()
                if pid.casefold() in seen:
                    continue
                seen.add(pid.casefold())
                try:
                    items.append(_read_user_unlocked(path, pid))
                except Exception as exc:
                    invalid.append(
                        InvalidPresetFile(pid, str(path), str(exc), _etag(path.read_bytes()))
                    )
        return items, invalid


def _assert_etag(current: StoredLLMPreset, expected_etag: Optional[str]) -> None:
    if expected_etag is not None and expected_etag != current.etag:
        raise LLMPresetConflictError(
            f"LLM preset changed: {current.config.id}",
            current_etag=current.etag,
        )


def _write_unlocked(config: LLMPresetConfig, credential_ref: str) -> StoredLLMPreset:
    pid = _preset_id(config.id)
    path = _path(pid)
    raw = _document_bytes(config, credential_ref)
    try:
        atomic_write_text(
            path,
            raw.decode("utf-8"),
            backup_dir=_backup_dir(pid),
            keep_backups=_PRESET_BACKUP_COUNT,
            validate_existing=_validate_for(pid),
        )
    except InvalidExistingFileError as exc:
        raise LLMPresetInvalidError(
            f"refusing to overwrite invalid preset: {pid}"
        ) from exc
    return _read_user_unlocked(path, pid)


def create(
    payload: dict[str, Any],
    *,
    credential_ref: str = "",
    preset_id: Optional[str] = None,
) -> StoredLLMPreset:
    with _LLM_PRESETS_LOCK:
        pid = _preset_id(preset_id or f"usr_{uuid.uuid4().hex[:16]}")
        if pid in _builtin_map() or _path(pid).exists():
            raise LLMPresetConflictError(f"LLM preset already exists: {pid}")
        candidate = dict(payload)
        for field in _FORBIDDEN_PATCH_FIELDS:
            candidate.pop(field, None)
        candidate["id"] = pid
        candidate["api_key"] = ""
        candidate["model_ids"] = []
        try:
            config = LLMPresetConfig.model_validate(candidate)
        except Exception as exc:
            raise LLMPresetInvalidError(f"preset schema is invalid: {exc}") from exc
        if config.id != pid:
            raise LLMPresetInvalidError(
                f"preset id changes during normalization: {pid!r} -> {config.id!r}"
            )
        return _write_unlocked(config, credential_ref)


def update(
    preset_id: str,
    patch: dict[str, Any],
    *,
    expected_etag: Optional[str],
) -> StoredLLMPreset:
    with _LLM_PRESETS_LOCK:
        current = _get_unlocked(preset_id)
        _assert_etag(current, expected_etag)
        forbidden = _FORBIDDEN_PATCH_FIELDS.intersection(patch)
        if forbidden:
            raise LLMPresetInvalidError(
                f"immutable or non-persistent fields: {sorted(forbidden)}"
            )
        allowed = set(_config_payload(current.config)) - {"id"}
        unknown = set(patch) - allowed - {"credential_ref"}
        if unknown:
            raise LLMPresetInvalidError(f"unknown preset fields: {sorted(unknown)}")
        payload = _config_payload(current.config)
        payload.update({key: value for key, value in patch.items() if key in allowed})
        payload["id"] = current.config.id
        payload["api_key"] = ""
        payload["model_ids"] = []
        credential_ref = str(
            patch.get("credential_ref", current.credential_ref) or ""
        ).strip().lower()
        try:
            config = LLMPresetConfig.model_validate(payload)
        except Exception as exc:
            raise LLMPresetInvalidError(f"preset schema is invalid: {exc}") from exc
        return _write_unlocked(config, credential_ref)


def duplicate(preset_id: str, *, label: Optional[str] = None) -> StoredLLMPreset:
    with _LLM_PRESETS_LOCK:
        source = _get_unlocked(preset_id)
        payload = _config_payload(source.config)
        payload["label"] = label or f"{source.config.label} Copy"
        return create(payload, credential_ref=source.credential_ref)


def delete(preset_id: str, *, expected_etag: Optional[str]) -> Path:
    with _LLM_PRESETS_LOCK:
        pid = _preset_id(preset_id)
        if pid in _builtin_map():
            try:
                current = _get_unlocked(pid)
                current_etag = current.etag
            except LLMPresetInvalidError:
                current_etag = _etag(_path(pid).read_bytes())
            raise LLMPresetConflictError(
                "builtin presets must be reset instead of deleted",
                current_etag=current_etag,
            )
        path = _path(pid)
        try:
            current = _get_unlocked(pid)
        except LLMPresetInvalidError:
            raw_etag = _etag(path.read_bytes())
            if expected_etag is not None and expected_etag != raw_etag:
                raise LLMPresetConflictError(
                    f"Invalid LLM preset changed: {pid}", current_etag=raw_etag
                )
            return archive_file(path, _backup_dir(pid), suffix="corrupt")
        _assert_etag(current, expected_etag)
        if not path.exists():
            raise LLMPresetNotFoundError(f"LLM preset not found: {pid}")
        return archive_file(
            path,
            _backup_dir(pid),
            validate_existing=_validate_for(pid),
        )


def reset_builtin(preset_id: str, *, expected_etag: Optional[str]) -> StoredLLMPreset:
    with _LLM_PRESETS_LOCK:
        pid = _preset_id(preset_id)
        builtin = _builtin_map().get(pid)
        if builtin is None:
            current = _get_unlocked(pid)
            raise LLMPresetConflictError(
                "custom presets cannot be reset",
                current_etag=current.etag,
            )
        path = _path(pid)
        try:
            current = _get_unlocked(pid)
        except LLMPresetInvalidError:
            raw_etag = _etag(path.read_bytes())
            if expected_etag is not None and expected_etag != raw_etag:
                raise LLMPresetConflictError(
                    f"Invalid LLM preset changed: {pid}", current_etag=raw_etag
                )
            archive_file(path, _backup_dir(pid), suffix="corrupt")
            return _builtin_stored(pid, builtin)
        _assert_etag(current, expected_etag)
        if path.exists():
            archive_file(
                path,
                _backup_dir(pid),
                validate_existing=_validate_for(pid),
            )
        return _builtin_stored(pid, builtin)


def export_portable(preset_id: str) -> bytes:
    with _LLM_PRESETS_LOCK:
        stored = _get_unlocked(preset_id)
        config = stored.config.model_copy(deep=True)
        config.base_url = ""
        return _document_bytes(config, "")


def parse_import(raw: bytes, *, fallback_label: str = "Imported") -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise LLMPresetInvalidError(f"preset is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMPresetInvalidError("preset import must be an object")
    nested = payload.get("preset")
    if isinstance(nested, dict):
        payload = dict(nested)
    else:
        payload = dict(payload)
    for field in (*_FILE_METADATA, *_FORBIDDEN_PATCH_FIELDS):
        payload.pop(field, None)
    payload["label"] = str(payload.get("label") or fallback_label).strip() or "Imported"
    payload["api_key"] = ""
    payload["model_ids"] = []
    try:
        probe = LLMPresetConfig.model_validate({**payload, "id": "usr_import_probe"})
    except Exception as exc:
        raise LLMPresetInvalidError(f"preset schema is invalid: {exc}") from exc
    return _config_payload(probe)


def import_portable(raw: bytes, *, fallback_label: str = "Imported") -> StoredLLMPreset:
    return create(parse_import(raw, fallback_label=fallback_label))


def history(preset_id: str) -> list[dict[str, Any]]:
    pid = _preset_id(preset_id)
    with _LLM_PRESETS_LOCK:
        result: list[dict[str, Any]] = []
        for path in list_backups(_backup_dir(pid), f"{pid}.json"):
            try:
                raw = path.read_bytes()
                config, _ = _decode(raw, expected_id=pid)
            except Exception:
                continue
            result.append(
                {
                    "backup_id": path.name,
                    "label": config.label,
                    "etag": _etag(raw),
                    "mtime": path.stat().st_mtime,
                }
            )
        return result


def restore(
    preset_id: str,
    backup_id: str,
    *,
    expected_etag: Optional[str],
) -> StoredLLMPreset:
    pid = _preset_id(preset_id)
    with _LLM_PRESETS_LOCK:
        candidates = {
            path.name: path
            for path in list_backups(_backup_dir(pid), f"{pid}.json")
        }
        backup = candidates.get(backup_id)
        if backup is None:
            raise LLMPresetNotFoundError(f"Preset backup not found: {backup_id}")
        backup_raw = backup.read_bytes()
        try:
            current = _get_unlocked(pid)
        except LLMPresetNotFoundError:
            # A deleted custom preset has no current ETag. Require the caller to
            # identify the exact archived bytes it selected for restoration.
            if expected_etag is not None and expected_etag != _etag(backup_raw):
                raise LLMPresetConflictError(
                    f"Archived LLM preset changed: {pid}",
                    current_etag=_etag(backup_raw),
                )
            config, credential_ref = _decode(backup_raw, expected_id=pid)
            return _write_unlocked(config, credential_ref)
        except LLMPresetInvalidError:
            if expected_etag is not None and expected_etag != _etag(backup_raw):
                raise LLMPresetConflictError(
                    f"Archived LLM preset changed: {pid}",
                    current_etag=_etag(backup_raw),
                )
            _decode(backup_raw, expected_id=pid)
            replace_invalid_bytes(_path(pid), backup_raw, _backup_dir(pid))
            return _read_user_unlocked(_path(pid), pid)
        else:
            _assert_etag(current, expected_etag)
        config, credential_ref = _decode(backup_raw, expected_id=pid)
        return _write_unlocked(config, credential_ref)


def install_migrated(
    config: LLMPresetConfig,
    *,
    preset_id: str,
    credential_ref: str = "",
) -> StoredLLMPreset:
    """Idempotently install one legacy preset using an explicit migrated ID."""
    with _LLM_PRESETS_LOCK:
        pid = _preset_id(preset_id)
        normalized = LLMPresetConfig.model_validate(
            {**config.model_dump(), "id": pid, "api_key": "", "model_ids": []}
        )
        if normalized.id != pid:
            raise LLMPresetInvalidError(
                f"migrated preset id changes during normalization: {pid!r}"
            )
        path = _path(pid)
        wanted = _document_bytes(normalized, credential_ref)
        if path.exists():
            current = path.read_bytes()
            if current != wanted:
                raise LLMPresetConflictError(
                    f"Migration preset conflicts with existing file: {pid}",
                    current_etag=_etag(current),
                )
            return _read_user_unlocked(path, pid)
        return _write_unlocked(normalized, credential_ref)
