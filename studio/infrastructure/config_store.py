"""Compatibility facade over the split settings, credential, and LLM stores."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from . import credentials, settings_store

if TYPE_CHECKING:
    from .secrets import Secrets

_CONFIG_LOCK = threading.RLock()
_LOGGER = logging.getLogger(__name__)

_FIXED_CREDENTIALS = (
    (("gelbooru", "api_key"), "cred_setting_gelbooru", "Gelbooru API key", "api_key"),
    (("danbooru", "api_key"), "cred_setting_danbooru", "Danbooru API key", "api_key"),
    (("huggingface", "token"), "cred_setting_huggingface", "Hugging Face token", "token"),
    (("modelscope", "token"), "cred_setting_modelscope", "ModelScope token", "token"),
)


def _wandb_credential_id(preset_id: str) -> str:
    digest = hashlib.sha256(preset_id.encode("utf-8")).hexdigest()[:16]
    return f"cred_setting_wandb_{digest}"


def _get_path(values: dict[str, Any], path: tuple[str, str]) -> str:
    parent = values.get(path[0])
    if not isinstance(parent, dict):
        return ""
    return str(parent.get(path[1]) or "")


def _set_path(values: dict[str, Any], path: tuple[str, str], value: str) -> None:
    parent = values.get(path[0])
    if not isinstance(parent, dict):
        return
    parent[path[1]] = value


def _strip_compat_computed(values: dict[str, Any]) -> None:
    models = values.get("models")
    if isinstance(models, dict):
        models.pop("selected_anima", None)
        models.pop("custom_anima_paths", None)
        models.pop("custom", None)
    wd14 = values.get("wd14")
    if isinstance(wd14, dict):
        wd14.pop("model_ids", None)


def split_legacy_settings(
    legacy: "Secrets", *, include_empty_credentials: bool = False
) -> tuple[dict[str, Any], list[credentials.CredentialRecord]]:
    """Return non-secret settings values and deterministic credential records."""
    values = legacy.model_dump()
    _strip_compat_computed(values)
    llm = values.pop("llm_tagger", {})
    default_id = str(llm.get("current_preset") or "style_json")
    values["llm_tagger"] = {"default_preset_id": default_id}
    records: list[credentials.CredentialRecord] = []

    for path, credential_id, label, kind in _FIXED_CREDENTIALS:
        secret = _get_path(values, path)
        _set_path(values, path, "")
        if secret or include_empty_credentials:
            records.append(credentials.CredentialRecord(
                id=credential_id,
                kind=kind,
                label=label,
                secret=secret,
            ))

    wandb = values.get("wandb")
    if isinstance(wandb, dict):
        presets = wandb.get("presets")
        if isinstance(presets, list):
            for preset in presets:
                if not isinstance(preset, dict):
                    continue
                secret = str(preset.get("api_key") or "")
                preset["api_key"] = ""
                if secret or include_empty_credentials:
                    preset_id = str(preset.get("id") or "default")
                    records.append(credentials.CredentialRecord(
                        id=_wandb_credential_id(preset_id),
                        kind="api_key",
                        label=f"WandB · {preset.get('label') or preset_id}",
                        secret=secret,
                    ))
    return values, records


def _inject_credentials(values: dict[str, Any]) -> None:
    document = credentials.load()
    for path, credential_id, _label, _kind in _FIXED_CREDENTIALS:
        record = document.items.get(credential_id)
        if record is not None:
            _set_path(values, path, record.secret)

    wandb = values.get("wandb")
    if isinstance(wandb, dict) and isinstance(wandb.get("presets"), list):
        for preset in wandb["presets"]:
            if not isinstance(preset, dict):
                continue
            record = document.items.get(
                _wandb_credential_id(str(preset.get("id") or "default"))
            )
            if record is not None:
                preset["api_key"] = record.secret


def credential_references(credential_id: str) -> list[str]:
    cid = credential_id.strip().lower()
    references: list[str] = []
    for path, known_id, _label, _kind in _FIXED_CREDENTIALS:
        if cid == known_id:
            references.append("settings." + ".".join(path))
    values = settings_store.load().values
    wandb = values.get("wandb")
    if isinstance(wandb, dict) and isinstance(wandb.get("presets"), list):
        for preset in wandb["presets"]:
            if not isinstance(preset, dict):
                continue
            preset_id = str(preset.get("id") or "default")
            if cid == _wandb_credential_id(preset_id):
                references.append(f"settings.wandb.presets.{preset_id}")
    return references


def load() -> "Secrets":
    from . import secrets as legacy_secrets
    from ..services import llm_presets

    with _CONFIG_LOCK:
        values = deepcopy(settings_store.load().values)
        values.pop("llm_tagger", None)
        _inject_credentials(values)
        result = legacy_secrets.Secrets.model_validate(values)
        result.llm_tagger = llm_presets.legacy_config()
        return result


def public_snapshot() -> dict[str, Any]:
    """Return the masked compatibility view plus preset resource metadata."""
    from . import secrets as legacy_secrets
    from ..services import llm_presets

    result = legacy_secrets.to_masked_dict(load())
    listing = llm_presets.list_presets()
    presets: list[dict[str, Any]] = []
    for item in listing["items"]:
        public = dict(item)
        public["builtin"] = str(public.get("origin", "custom")) != "custom"
        public["api_key"] = (
            legacy_secrets.MASK
            if bool(public.get("credential_configured", False))
            else ""
        )
        presets.append(public)
    result["llm_tagger"] = {
        "current_preset": listing["default_preset_id"],
        "presets": presets,
    }
    return result


def write_legacy_projection(value: "Secrets") -> None:
    """Best-effort rollback snapshot; never used as a new-version authority."""
    from . import secrets as legacy_secrets
    from .atomic_files import atomic_write_text

    payload = value.model_dump()
    payload["_storage_projection"] = {
        "kind": "adr-0017-derived-compatibility-snapshot",
        "schema_version": 1,
    }
    atomic_write_text(
        legacy_secrets.SECRETS_FILE,
        json.dumps(payload, ensure_ascii=False, indent=2),
        backup_dir=legacy_secrets.SECRETS_FILE.parent / "backups" / "secrets",
        keep_backups=3,
        validate_existing=legacy_secrets._validate_secrets_bytes,
        mode=0o600,
    )


def project_current_best_effort() -> None:
    try:
        write_legacy_projection(load())
    except Exception:
        _LOGGER.warning("failed to update legacy secrets projection", exc_info=True)


def save(value: "Secrets") -> "Secrets":
    """Persist non-LLM settings and credentials without rewriting preset files."""
    with _CONFIG_LOCK:
        values, records = split_legacy_settings(
            value, include_empty_credentials=True
        )
        settings_store.save(settings_store.AppSettingsDocument(values=values))
        for record in records:
            credentials.put(
                record.id,
                label=record.label,
                secret=record.secret,
                kind=record.kind,
            )
        result = load()
        try:
            write_legacy_projection(result)
        except Exception:
            _LOGGER.warning("failed to update legacy secrets projection", exc_info=True)
        return result


def update(patch: dict[str, Any]) -> "Secrets":
    from . import secrets as legacy_secrets
    from ..services import llm_presets

    with _CONFIG_LOCK:
        patch = deepcopy(patch)
        llm_patch = patch.pop("llm_tagger", None)
        if llm_patch is not None:
            if not isinstance(llm_patch, dict):
                raise ValueError("llm_tagger patch must be an object")
            if set(llm_patch) - {"current_preset"}:
                raise ValueError(
                    "LLM presets must be changed through the preset resource API"
                )
            if "current_preset" in llm_patch:
                llm_presets.set_default_preset(str(llm_patch["current_preset"]))
        current = load()
        current_values = current.model_dump()
        _strip_compat_computed(current_values)
        merged = legacy_secrets._deep_merge(current_values, patch)
        return save(legacy_secrets.Secrets.model_validate(merged))
