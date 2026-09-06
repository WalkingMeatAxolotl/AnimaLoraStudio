from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from studio.api.exception_handlers import register_exception_handlers
from studio.api.routers import settings as settings_router
from studio.infrastructure import config_store
from studio.infrastructure import credentials
from studio.infrastructure import llm_model_cache
from studio.infrastructure import llm_preset_store
from studio.infrastructure import secrets
from studio.infrastructure import settings_store
from studio.infrastructure import storage_layout


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "credentials.json")
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(llm_preset_store, "LLM_PRESETS_DIR", tmp_path / "llm_presets")
    monkeypatch.setattr(llm_model_cache, "LLM_MODEL_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(storage_layout, "is_split_complete", lambda: True)

    legacy = secrets.Secrets()
    legacy.gelbooru.user_id = "alice"
    legacy.gelbooru.api_key = "private-key"
    values, records = config_store.split_legacy_settings(legacy)
    settings_store.save(settings_store.AppSettingsDocument(values=values))
    credentials.import_records(records)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(settings_router.router)
    return TestClient(app)


def test_settings_patch_preserves_separate_credentials(client: TestClient) -> None:
    fetched = client.get("/api/settings")
    assert fetched.status_code == 200
    assert fetched.json()["gelbooru"]["api_key"] == secrets.MASK
    assert fetched.json()["llm_tagger"]["presets"][0]["etag"].startswith("sha256:")
    assert "private-key" not in fetched.text

    updated = client.patch("/api/settings", json={"gelbooru": {"user_id": "bob"}})
    assert updated.status_code == 200, updated.text
    assert updated.json()["gelbooru"]["user_id"] == "bob"
    assert updated.json()["gelbooru"]["api_key"] == secrets.MASK
    assert settings_store.load().values["gelbooru"]["api_key"] == ""
    assert "private-key" in {item.secret for item in credentials.load().items.values()}
    selected = client.patch(
        "/api/settings",
        json={"models": {"selected": {"krea2": "raw_fp8"}}},
    )
    assert selected.status_code == 200
    assert selected.json()["models"]["selected"]["krea2"] == "raw_fp8"
    assert "selected_anima" not in settings_store.load().values["models"]


def test_settings_api_rejects_llm_aggregate_mutation(client: TestClient) -> None:
    response = client.patch(
        "/api/settings",
        json={"llm_tagger": {"presets": []}},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "settings.llm_resource_required"
