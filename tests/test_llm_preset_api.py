from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from studio.api.exception_handlers import register_exception_handlers
from studio.api.routers import credentials as credentials_router
from studio.api.routers import llm_presets as presets_router
from studio.infrastructure import credentials
from studio.infrastructure import llm_preset_store as store


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(store, "LLM_PRESETS_DIR", tmp_path / "llm_presets")
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "credentials.json")
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(credentials_router.router)
    app.include_router(presets_router.router)
    return TestClient(app)


def _preset_payload(label: str = "Recipe") -> dict:
    return {
        "label": label,
        "base_url": "https://example.test/v1",
        "model": "vision-model",
        "messages": [
            {"type": "text", "role": "system", "content": "Describe it"},
            {"type": "image", "role": "user", "content": ""},
        ],
    }


def _create_credential(client: TestClient) -> dict:
    response = client.post(
        "/api/credentials",
        json={"label": "Primary", "kind": "api_key", "secret": "secret-value"},
    )
    assert response.status_code == 201, response.text
    assert "secret" not in response.text
    return response.json()


def _create_preset(client: TestClient, credential_id: str = "") -> dict:
    response = client.post(
        "/api/llm-tagger/presets",
        json={"preset": _preset_payload(), "credential_ref": credential_id},
    )
    assert response.status_code == 201, response.text
    assert response.headers["etag"]
    return response.json()


def test_crud_uses_etag_and_never_returns_secret(client: TestClient) -> None:
    credential = _create_credential(client)
    created = _create_preset(client, credential["id"])

    fetched = client.get(f"/api/llm-tagger/presets/{created['id']}")
    assert fetched.status_code == 200
    etag = fetched.headers["etag"]
    assert fetched.json()["credential_configured"] is True
    assert "secret-value" not in fetched.text
    assert "api_key" not in fetched.json()

    missing_precondition = client.patch(
        f"/api/llm-tagger/presets/{created['id']}",
        json={"temperature": 0.8},
    )
    assert missing_precondition.status_code == 428

    updated = client.patch(
        f"/api/llm-tagger/presets/{created['id']}",
        headers={"If-Match": etag},
        json={"temperature": 0.8},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["temperature"] == pytest.approx(0.8)

    stale = client.patch(
        f"/api/llm-tagger/presets/{created['id']}",
        headers={"If-Match": etag},
        json={"temperature": 0.9},
    )
    assert stale.status_code == 412
    assert stale.json()["error"]["details"]["current_etag"] == updated.json()["etag"]


def test_model_refresh_is_read_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = _create_credential(client)
    created = _create_preset(client, credential["id"])
    path = store.LLM_PRESETS_DIR / f"{created['id']}.json"
    before = path.read_bytes()
    captured: dict = {}

    def _fetch(base_url: str, api_key: str, *, timeout: int, session=None):
        captured.update(
            {"base_url": base_url, "api_key": api_key, "timeout": timeout}
        )
        return ["model-a", "model-b"]

    monkeypatch.setattr(
        presets_router.llm_tagger_service,
        "fetch_openai_compatible_models",
        _fetch,
    )

    response = client.post(
        f"/api/llm-tagger/presets/{created['id']}/models/refresh",
        json={"timeout": 12},
    )

    assert response.status_code == 200, response.text
    assert response.json()["items"] == ["model-a", "model-b"]
    assert captured == {
        "base_url": "https://example.test/v1",
        "api_key": "secret-value",
        "timeout": 12,
    }
    assert path.read_bytes() == before


def test_credential_delete_checks_preset_references(client: TestClient) -> None:
    credential = _create_credential(client)
    created = _create_preset(client, credential["id"])

    blocked = client.delete(
        f"/api/credentials/{credential['id']}",
        headers={"If-Match": credential["etag"]},
    )
    assert blocked.status_code == 409
    assert blocked.json()["error"]["details"]["referenced_by"] == [created["id"]]

    forced = client.delete(
        f"/api/credentials/{credential['id']}?force=true",
        headers={"If-Match": credential["etag"]},
    )
    assert forced.status_code == 200
    fetched = client.get(f"/api/llm-tagger/presets/{created['id']}")
    assert fetched.json()["credential_configured"] is False


def test_credential_stale_write_returns_current_etag(client: TestClient) -> None:
    credential = _create_credential(client)
    updated = client.put(
        f"/api/credentials/{credential['id']}/secret",
        headers={"If-Match": credential["etag"]},
        json={"secret": "rotated"},
    )
    assert updated.status_code == 200

    stale = client.put(
        f"/api/credentials/{credential['id']}/secret",
        headers={"If-Match": credential["etag"]},
        json={"secret": "lost-update"},
    )

    assert stale.status_code == 412
    assert stale.json()["error"]["details"]["current_etag"] == updated.json()["etag"]


def test_preset_reads_survive_degraded_credential_store(client: TestClient) -> None:
    credential = _create_credential(client)
    created = _create_preset(client, credential["id"])
    credentials.CREDENTIALS_FILE.write_text("{broken", encoding="utf-8")

    listed = client.get("/api/llm-tagger/presets")
    fetched = client.get(f"/api/llm-tagger/presets/{created['id']}")
    connection = client.post(
        f"/api/llm-tagger/presets/{created['id']}/models/refresh",
        json={},
    )

    assert listed.status_code == 200
    item = next(row for row in listed.json()["items"] if row["id"] == created["id"])
    assert item["credential_status"] == "degraded"
    assert item["credential_configured"] is False
    assert fetched.status_code == 200
    assert fetched.json()["credential_status"] == "degraded"
    assert connection.status_code == 503


def test_portable_export_and_import_remove_local_binding(client: TestClient) -> None:
    credential = _create_credential(client)
    created = _create_preset(client, credential["id"])

    exported = client.get(f"/api/llm-tagger/presets/{created['id']}/export")

    assert exported.status_code == 200
    payload = exported.json()
    assert payload["credential_ref"] == ""
    assert payload["base_url"] == ""
    assert "api_key" not in payload
    assert "model_ids" not in payload

    imported = client.post(
        "/api/llm-tagger/presets/import",
        files={"file": ("portable.json", json.dumps(payload), "application/json")},
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["id"] != created["id"]
    assert imported.json()["credential_ref"] == ""
    assert imported.json()["credential_configured"] is False


def test_builtin_override_reset_requires_current_etag(client: TestClient) -> None:
    fetched = client.get("/api/llm-tagger/presets/style_json")
    etag = fetched.headers["etag"]
    overridden = client.patch(
        "/api/llm-tagger/presets/style_json",
        headers={"If-Match": etag},
        json={"label": "Local Style"},
    )
    assert overridden.status_code == 200
    assert overridden.json()["origin"] == "builtin_override"

    reset = client.post(
        "/api/llm-tagger/presets/style_json/reset",
        headers={"If-Match": f'"{overridden.json()["etag"]}"'},
    )
    assert reset.status_code == 200, reset.text
    assert reset.json()["origin"] == "builtin"
    assert reset.json()["label"] != "Local Style"
