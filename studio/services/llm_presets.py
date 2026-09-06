"""LLM preset orchestration across preset files and write-only credentials."""
from __future__ import annotations

from typing import Any, Optional

from ..infrastructure import credentials
from ..infrastructure import llm_preset_store as preset_store


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


def present(stored: preset_store.StoredLLMPreset) -> dict[str, Any]:
    data = stored.public_dict()
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
    return data


def list_presets() -> dict[str, Any]:
    items, invalid = preset_store.list_all()
    return {
        "items": [present(item) for item in items],
        "invalid_items": [item.public_dict() for item in invalid],
    }


def get_preset(preset_id: str) -> dict[str, Any]:
    return present(preset_store.get(preset_id))


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


def credential_references(credential_id: str) -> list[str]:
    cid = credential_id.strip().lower()
    items, _ = preset_store.list_all()
    return sorted(item.config.id for item in items if item.credential_ref == cid)


def delete_credential(
    credential_id: str,
    *,
    expected_etag: Optional[str],
    force: bool = False,
) -> None:
    references = credential_references(credential_id)
    if references and not force:
        raise CredentialReferenceError(credential_id, references)
    credentials.delete(credential_id, expected_etag=expected_etag)


def resolved_connection(
    preset_id: str,
) -> tuple[preset_store.StoredLLMPreset, str]:
    stored = preset_store.get(preset_id)
    secret = ""
    if stored.credential_ref:
        secret = credentials.resolve(stored.credential_ref)
    return stored, secret
