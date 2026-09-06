"""Crash-resumable migration from monolithic secrets to ADR 0017 stores."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import config_store
from . import credentials
from . import llm_preset_store as preset_store
from . import settings_store
from .atomic_files import atomic_write_bytes, atomic_write_text
from .llm_presets import builtin_llm_presets
from .paths import STUDIO_DATA
from . import secrets as legacy_secrets

STORAGE_LAYOUT_FILE = STUDIO_DATA / "storage-layout.json"
_MIGRATION_KEY = "split_llm_presets_v1"
_LAYOUT_LOCK = threading.RLock()


class StorageMigrationError(RuntimeError):
    pass


class MigrationRecord(BaseModel):
    state: Literal["prepared", "complete"]
    source_sha256: str
    backup_path: str = ""
    id_map: dict[str, str] = Field(default_factory=dict)
    default_preset_id: str = "style_json"


class StorageLayout(BaseModel):
    kind: Literal["anima-storage-layout"] = "anima-storage-layout"
    schema_version: Literal[1] = 1
    migrations: dict[str, MigrationRecord] = Field(default_factory=dict)


def _layout_backup_dir() -> Path:
    return STORAGE_LAYOUT_FILE.parent / "backups" / "storage-layout"


def _decode_layout(raw: bytes) -> StorageLayout:
    return StorageLayout.model_validate_json(raw)


def _load_layout() -> StorageLayout:
    if not STORAGE_LAYOUT_FILE.exists():
        return StorageLayout()
    try:
        return _decode_layout(STORAGE_LAYOUT_FILE.read_bytes())
    except Exception as exc:
        raise StorageMigrationError(
            f"Storage layout marker is invalid: {STORAGE_LAYOUT_FILE}"
        ) from exc


def _save_layout(layout: StorageLayout) -> None:
    atomic_write_text(
        STORAGE_LAYOUT_FILE,
        layout.model_dump_json(indent=2),
        backup_dir=_layout_backup_dir(),
        keep_backups=3,
        validate_existing=lambda raw: _decode_layout(raw),
    )


def _source_hash(raw: bytes | None) -> str:
    if raw is None:
        return "missing"
    return hashlib.sha256(raw).hexdigest()


def _legacy_backup(raw: bytes | None, source_hash: str) -> Path | None:
    if raw is None:
        return None
    target = (
        STORAGE_LAYOUT_FILE.parent
        / "backups"
        / "legacy"
        / f"secrets-pre-split-{source_hash[:16]}.json"
    )
    if target.exists():
        if target.read_bytes() != raw:
            raise StorageMigrationError(f"Legacy backup hash collision: {target}")
        return target
    atomic_write_bytes(target, raw, mode=0o600)
    return target


def _portable_id(old_id: str, used: set[str]) -> str:
    lowered = str(old_id or "").strip().lower()
    try:
        candidate = preset_store.validate_preset_id(lowered)
    except preset_store.LLMPresetInvalidError:
        candidate = ""
    if candidate and candidate not in used:
        used.add(candidate)
        return candidate
    digest = hashlib.sha256(str(old_id).encode("utf-8")).hexdigest()
    for width in range(16, 65, 4):
        candidate = f"usr_{digest[:width]}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise StorageMigrationError(f"Could not allocate preset id for {old_id!r}")


def _recipe(config: legacy_secrets.LLMPresetConfig) -> dict:
    data = config.model_dump()
    for key in ("id", "api_key", "model_ids", "builtin"):
        data.pop(key, None)
    return data


def _migration_plan(
    legacy: legacy_secrets.Secrets,
) -> tuple[
    dict[str, str],
    list[tuple[legacy_secrets.LLMPresetConfig, str, str]],
    list[credentials.CredentialRecord],
    str,
]:
    builtin_configs = {
        item["id"]: legacy_secrets.LLMPresetConfig.model_validate(
            {**item, "api_key": "", "model_ids": []}
        )
        for item in builtin_llm_presets()
    }
    used = set(builtin_configs)
    id_map: dict[str, str] = {}
    # Exact builtin IDs retain identity. A custom preset that only differs by
    # case is remapped because Windows filenames are case-insensitive.
    for preset in legacy.llm_tagger.presets:
        if preset.id in builtin_configs:
            id_map[preset.id] = preset.id
        else:
            id_map[preset.id] = _portable_id(preset.id, used)

    installs: list[tuple[legacy_secrets.LLMPresetConfig, str, str]] = []
    credential_records: list[credentials.CredentialRecord] = []
    for preset in legacy.llm_tagger.presets:
        target_id = id_map[preset.id]
        raw_secret = str(preset.api_key or "")
        has_secret = bool(raw_secret and raw_secret != legacy_secrets.MASK)
        credential_ref = ""
        if has_secret:
            digest = hashlib.sha256(preset.id.encode("utf-8")).hexdigest()[:16]
            credential_ref = f"cred_llm_{digest}"
            credential_records.append(
                credentials.CredentialRecord(
                    id=credential_ref,
                    label=f"LLM · {preset.label}",
                    secret=raw_secret,
                )
            )
        builtin = builtin_configs.get(preset.id)
        needs_file = (
            builtin is None
            or _recipe(preset) != _recipe(builtin)
            or bool(credential_ref)
        )
        if needs_file:
            installs.append((preset, target_id, credential_ref))

    default_id = id_map.get(legacy.llm_tagger.current_preset)
    if default_id is None:
        default_id = next(iter(id_map.values()), "style_json")
    return id_map, installs, credential_records, default_id


def _validate_final(
    legacy: legacy_secrets.Secrets,
    id_map: dict[str, str],
    installs: list[tuple[legacy_secrets.LLMPresetConfig, str, str]],
    credential_records: list[credentials.CredentialRecord],
    default_id: str,
) -> None:
    items, invalid = preset_store.list_all()
    if invalid:
        raise StorageMigrationError(
            f"Invalid migrated preset files: {[item.id for item in invalid]}"
        )
    available = {item.config.id: item for item in items}
    legacy_by_id = {item.id: item for item in legacy.llm_tagger.presets}
    if set(id_map.values()) - set(available):
        raise StorageMigrationError("Migrated preset count/identity validation failed")
    install_refs = {target: ref for _, target, ref in installs}
    for old_id, target_id in id_map.items():
        stored = available[target_id]
        if _recipe(stored.config) != _recipe(legacy_by_id[old_id]):
            raise StorageMigrationError(
                f"Migrated preset content differs for {old_id!r}"
            )
        if stored.credential_ref != install_refs.get(target_id, ""):
            raise StorageMigrationError(
                f"Migrated credential reference differs for {old_id!r}"
            )
    document = credentials.load()
    for record in credential_records:
        if document.items.get(record.id) != record:
            raise StorageMigrationError(
                f"Migrated credential differs for {record.id!r}"
            )
    if settings_store.get_default_llm_preset_id() != default_id:
        raise StorageMigrationError("Migrated default preset differs")


def _authorities_share_storage_root() -> bool:
    """Return whether every split authority belongs to this layout marker.

    Besides catching invalid runtime wiring, this prevents a marker from one
    Studio Data root from activating stores in another root.  That matters
    when a backup is mounted independently and keeps tests that isolate the
    legacy source from accidentally reading the developer's live stores.
    """
    root = STORAGE_LAYOUT_FILE.parent.resolve()
    authority_roots = (
        legacy_secrets.SECRETS_FILE.parent.resolve(),
        credentials.CREDENTIALS_FILE.parent.resolve(),
        settings_store.SETTINGS_FILE.parent.resolve(),
        preset_store.LLM_PRESETS_DIR.parent.resolve(),
    )
    return all(candidate == root for candidate in authority_roots)


def is_split_complete() -> bool:
    with _LAYOUT_LOCK:
        if not _authorities_share_storage_root():
            return False
        record = _load_layout().migrations.get(_MIGRATION_KEY)
        return record is not None and record.state == "complete"


def ensure_storage_layout() -> MigrationRecord:
    """Run or resume the v1 split before the HTTP server starts accepting work."""
    with _LAYOUT_LOCK:
        layout = _load_layout()
        existing = layout.migrations.get(_MIGRATION_KEY)
        if existing is not None and existing.state == "complete":
            return existing

        if legacy_secrets.SECRETS_FILE.exists():
            # ``load`` is strict/recovering; migration must never synthesize
            # defaults over an unreadable existing source.
            legacy = legacy_secrets.load()
            source_raw: bytes | None = legacy_secrets.SECRETS_FILE.read_bytes()
        else:
            legacy = legacy_secrets.Secrets()
            source_raw = None
        source_hash = _source_hash(source_raw)
        id_map, installs, credential_records, default_id = _migration_plan(legacy)
        settings_values, settings_credentials = config_store.split_legacy_settings(legacy)
        settings_values["llm_tagger"] = {"default_preset_id": default_id}
        credential_records.extend(settings_credentials)

        if existing is not None:
            if existing.source_sha256 != source_hash:
                raise StorageMigrationError(
                    "Legacy secrets changed while storage migration was prepared"
                )
            if existing.id_map != id_map or existing.default_preset_id != default_id:
                raise StorageMigrationError(
                    "Prepared storage migration plan no longer matches its source"
                )
            record = existing
        else:
            backup = _legacy_backup(source_raw, source_hash)
            record = MigrationRecord(
                state="prepared",
                source_sha256=source_hash,
                backup_path=str(backup) if backup else "",
                id_map=id_map,
                default_preset_id=default_id,
            )
            layout.migrations[_MIGRATION_KEY] = record
            _save_layout(layout)

        credentials.import_records(credential_records)
        for config, target_id, credential_ref in installs:
            preset_store.install_migrated(
                config,
                preset_id=target_id,
                credential_ref=credential_ref,
            )
        settings_store.install_migrated(
            settings_store.AppSettingsDocument(values=settings_values)
        )
        _validate_final(
            legacy,
            id_map,
            installs,
            credential_records,
            default_id,
        )

        completed = record.model_copy(update={"state": "complete"})
        layout.migrations[_MIGRATION_KEY] = completed
        _save_layout(layout)
        return completed
