"""Disposable model-discovery cache for LLM preset connections."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .atomic_files import atomic_write_text
from .paths import STUDIO_DATA

LLM_MODEL_CACHE_DIR = STUDIO_DATA / "cache" / "llm_models"


def _path(preset_id: str) -> Path:
    digest = hashlib.sha256(preset_id.encode("utf-8")).hexdigest()[:24]
    return LLM_MODEL_CACHE_DIR / f"{digest}.json"


def _fingerprint(base_url: str, credential_ref: str) -> str:
    raw = f"{base_url.strip()}\0{credential_ref.strip().lower()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load(
    preset_id: str, *, base_url: str, credential_ref: str
) -> list[str]:
    path = _path(preset_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text("utf-8"))
        if payload.get("kind") != "anima-llm-model-cache":
            return []
        if payload.get("fingerprint") != _fingerprint(base_url, credential_ref):
            return []
        items = payload.get("items")
        if not isinstance(items, list):
            return []
        return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))
    except (OSError, ValueError, TypeError):
        return []


def save(
    preset_id: str,
    items: list[str],
    *,
    base_url: str,
    credential_ref: str,
) -> None:
    normalized = list(
        dict.fromkeys(str(item).strip() for item in items if str(item).strip())
    )
    payload = {
        "kind": "anima-llm-model-cache",
        "schema_version": 1,
        "preset_id": preset_id,
        "fingerprint": _fingerprint(base_url, credential_ref),
        "items": normalized,
    }
    atomic_write_text(
        _path(preset_id), json.dumps(payload, ensure_ascii=False, indent=2)
    )


def discard(preset_id: str) -> None:
    _path(preset_id).unlink(missing_ok=True)
