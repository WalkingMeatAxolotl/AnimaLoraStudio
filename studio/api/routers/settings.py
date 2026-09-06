"""Public non-secret settings API backed by the split storage facade."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ...domain.errors import DomainError
from ...infrastructure import config_store
from ...infrastructure import secrets

router = APIRouter()


@router.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return config_store.public_snapshot()


@router.patch("/api/settings")
def patch_settings(body: dict[str, Any]) -> dict[str, Any]:
    if "llm_tagger" in body:
        raise DomainError(
            "LLM defaults and presets must use the preset resource API",
            code="settings.llm_resource_required",
            http_status=409,
        )
    secrets.update(body)
    return config_store.public_snapshot()
