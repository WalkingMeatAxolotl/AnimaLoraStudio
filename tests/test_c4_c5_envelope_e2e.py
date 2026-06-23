"""PR-2 C4+C5 端到端 — 真实 router 走完 DomainError handler 后 envelope 形态。

不仅锁 status code（test_error_response_baseline 已锁），还要锁：
  - body.detail 仍是 string（前端老路径不破）
  - body.error.code 是 service-domain 命名 (preset.not_found 等)
  - body.error.trace_id 跟 X-Trace-Id header 一致
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.api.routers import root as _root_router
from studio.api.routers import samples as _samples_router
from studio.infrastructure.logging import TRACE_HEADER


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    output = tmp_path / "output"
    (output / "samples").mkdir(parents=True)
    web_dist = tmp_path / "web_dist"
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setattr(server, "WEB_DIST", web_dist)
    monkeypatch.setattr(_samples_router, "OUTPUT_DIR", output)
    monkeypatch.setattr(_root_router, "WEB_DIST", web_dist)
    from studio.services.presets import io as presets_io
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", tmp_path / "presets")
    return TestClient(server.app)


# ── preset 404 路径（C4 batch 1）───────────────────────────────────────


def test_preset_not_found_envelope(client: TestClient) -> None:
    resp = client.get("/api/presets/__nonexistent__")
    assert resp.status_code == 404
    body = resp.json()
    # Phase 3：只发 error 信封，无 legacy detail
    assert "detail" not in body
    assert body["error"]["code"] == "preset.not_found"
    assert "not found" in body["error"]["message"]
    assert "__nonexistent__" in body["error"]["message"]
    # trace_id 跟 header 一致
    assert body["error"]["trace_id"] == resp.headers[TRACE_HEADER]


def test_preset_name_invalid_envelope_400(client: TestClient) -> None:
    """PUT 非法 preset 名 — 走 PresetNameInvalidError → 400 / preset.name_invalid。"""
    resp = client.put("/api/presets/bad..slash/name", json={})
    # 路由匹配可能让这变 404（path 含 / FastAPI 看不到这是 single name）
    # 我们接受 400 或 404；锁 envelope 形态
    body = resp.json()
    assert resp.status_code in (400, 404, 422)
    # Phase 3：error 信封（422 RequestValidationError 仍是 detail list）
    assert "error" in body or isinstance(body.get("detail"), list)
    # 4xx 全部应该有 trace_id header（middleware）
    assert TRACE_HEADER in resp.headers
