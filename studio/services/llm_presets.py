"""LLM preset orchestration across preset files and write-only credentials."""
from __future__ import annotations

from typing import Any, Optional

from ..infrastructure import credentials
from ..infrastructure import llm_model_cache
from ..infrastructure import llm_preset_store as preset_store
from ..infrastructure import settings_store


class CredentialReferenceError(RuntimeError):
    def __init__(self, credential_id: str, referenced_by: list[str]) -> None:
        super().__init__(f"Credential is still referenced: {credential_id}")
        self.credential_id = credential_id
        self.referenced_by = referenced_by


def _assert_credential_exists(credential_ref: str) -> None:
    if not credential_ref:
        return
    document = credentials.load()
    if credential_ref not in document.items:
        raise credentials.CredentialNotFoundError(
            f"Credential not found: {credential_ref}"
        )


def present(
    stored: preset_store.StoredLLMPreset, *, default_id: Optional[str] = None
) -> dict[str, Any]:
    if default_id is None:
        default_id = settings_store.get_default_llm_preset_id()
    data = stored.public_dict()
    data["model_ids"] = llm_model_cache.load(
        stored.config.id,
        base_url=stored.config.base_url,
        credential_ref=stored.credential_ref,
    )
    status = "unconfigured"
    if stored.credential_ref:
        try:
            status = "configured" if credentials.resolve(stored.credential_ref) else "empty"
        except credentials.CredentialNotFoundError:
            status = "missing"
        except (credentials.CredentialStoreCorruptError, OSError):
            # Preset documents remain usable/inspectable when the credential
            # store is degraded; connection operations still fail explicitly.
            status = "degraded"
    data["credential_status"] = status
    data["credential_configured"] = status == "configured"
    data["is_default"] = bool(default_id and stored.config.id == default_id)
    return data


def list_presets() -> dict[str, Any]:
    items, invalid = preset_store.list_all()
    default_id = settings_store.get_default_llm_preset_id()
    return {
        "items": [present(item, default_id=default_id) for item in items],
        "invalid_items": [item.public_dict() for item in invalid],
        "default_preset_id": default_id,
    }


def get_preset(preset_id: str) -> dict[str, Any]:
    return present(
        preset_store.get(preset_id),
        default_id=settings_store.get_default_llm_preset_id(),
    )


def create_preset(
    payload: dict[str, Any], *, credential_ref: str = ""
) -> preset_store.StoredLLMPreset:
    ref = credential_ref.strip().lower()
    _assert_credential_exists(ref)
    return preset_store.create(payload, credential_ref=ref)


def update_preset(
    preset_id: str,
    patch: dict[str, Any],
    *,
    expected_etag: str,
) -> preset_store.StoredLLMPreset:
    if "credential_ref" in patch:
        _assert_credential_exists(str(patch.get("credential_ref") or "").strip().lower())
    return preset_store.update(
        preset_id,
        patch,
        expected_etag=expected_etag,
    )


def set_default_preset(preset_id: str) -> str:
    stored = preset_store.get(preset_id)
    settings_store.set_default_llm_preset_id(stored.config.id)
    return stored.config.id


def delete_preset(preset_id: str, *, expected_etag: str) -> None:
    if settings_store.get_default_llm_preset_id() == preset_id:
        raise preset_store.LLMPresetConflictError(
            "Select another default before deleting this preset"
        )
    preset_store.delete(preset_id, expected_etag=expected_etag)
    llm_model_cache.discard(preset_id)


def credential_references(credential_id: str) -> list[str]:
    from ..infrastructure import config_store

    cid = credential_id.strip().lower()
    items, _ = preset_store.list_all()
    preset_refs = [item.config.id for item in items if item.credential_ref == cid]
    return sorted(set(preset_refs + config_store.credential_references(cid)))


def delete_credential(
    credential_id: str,
    *,
    expected_etag: Optional[str],
) -> None:
    references = credential_references(credential_id)
    if references:
        raise CredentialReferenceError(credential_id, references)
    credentials.delete(credential_id, expected_etag=expected_etag)


def snapshot(preset_id: Optional[str] = None) -> dict[str, Any]:
    """Freeze a non-secret recipe for a queued tagging task."""
    target_id = preset_id or settings_store.get_default_llm_preset_id()
    stored = preset_store.get(target_id)
    config = stored.config.model_dump(exclude={"api_key", "model_ids"})
    return {
        "kind": "anima-llm-preset-snapshot",
        "schema_version": 1,
        "preset_id": stored.config.id,
        "preset_etag": stored.etag,
        "credential_ref": stored.credential_ref,
        "config": config,
    }


def legacy_config():
    """Compose the runtime legacy shape without making it a persistence source."""
    from ..infrastructure import secrets as legacy_secrets

    listing = list_presets()
    presets: list[legacy_secrets.LLMPresetConfig] = []
    for item in listing["items"]:
        payload = {
            key: value
            for key, value in item.items()
            if key in legacy_secrets.LLMPresetConfig.model_fields
        }
        credential_ref = str(item.get("credential_ref") or "")
        if credential_ref:
            try:
                payload["api_key"] = credentials.resolve(credential_ref)
            except credentials.CredentialNotFoundError:
                payload["api_key"] = ""
        else:
            payload["api_key"] = ""
        presets.append(legacy_secrets.LLMPresetConfig.model_validate(payload))
    return legacy_secrets.LLMTaggerConfig(
        current_preset=listing["default_preset_id"],
        presets=presets,
    )


def resolved_connection(
    preset_id: str,
) -> tuple[preset_store.StoredLLMPreset, str]:
    stored = preset_store.get(preset_id)
    secret = ""
    if stored.credential_ref:
        secret = credentials.resolve(stored.credential_ref)
    return stored, secret
