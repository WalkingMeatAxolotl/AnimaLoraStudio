"""PR-2 C2 — exception handler + dual-write envelope 验证。

覆盖：
  - DomainError → dual-write envelope {detail, error: {code,message,trace_id}}
  - HTTPException 不被新 handler 截胡（starlette 默认 → {detail} 保形）
  - Exception fallback → 500 + 脱敏 envelope（不 leak traceback）
  - RequestValidationError → 保 list[dict] 形状不变
  - 4xx vs 5xx logger level（4xx info / 5xx exception）
  - trace_id 进 body.error.trace_id 跟 header X-Trace-Id 一致
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from studio.domain.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    PresetNotFoundError,
    ValidationError,
)
from studio.infrastructure.logging import TRACE_HEADER


@pytest.fixture
def app() -> FastAPI:
    """Bare FastAPI app + TraceIdMiddleware + exception handlers（不带业务 router）。"""
    from studio.api.exception_handlers import register_exception_handlers
    from studio.api.trace_middleware import TraceIdMiddleware

    a = FastAPI()
    a.add_middleware(TraceIdMiddleware)
    register_exception_handlers(a)

    @a.get("/raise_domain")
    def _raise_domain():
        raise PresetNotFoundError("preset 'foo' does not exist",
                                   details={"name": "foo"})

    @a.get("/raise_validation")
    def _raise_validation():
        raise ValidationError("epoch must be > 0", details={"field": "epoch"})

    @a.get("/raise_conflict")
    def _raise_conflict():
        raise ConflictError("slug 'x' already exists")

    @a.get("/raise_http")
    def _raise_http():
        raise HTTPException(status_code=400, detail="legacy http err string")

    @a.get("/raise_http_dict")
    def _raise_http_dict():
        raise HTTPException(status_code=400,
                            detail={"error": "running_tasks_present", "count": 3})

    @a.get("/raise_uncaught")
    def _raise_uncaught():
        raise RuntimeError("oops something raw broke")

    @a.get("/raise_domain_500")
    def _raise_domain_500():
        raise DomainError("upstream timed out", code="upstream.timeout",
                          http_status=503)

    return a


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ── DomainError → dual-write envelope ──────────────────────────────────


def test_domain_error_returns_subclass_http_status(client: TestClient) -> None:
    resp = client.get("/raise_domain")
    assert resp.status_code == 404


def test_domain_error_body_has_detail_legacy(client: TestClient) -> None:
    """前端 client.ts 老路径读 body.detail 是 string — dual-write 保这条。"""
    resp = client.get("/raise_domain")
    body = resp.json()
    assert "detail" in body
    assert body["detail"] == "preset 'foo' does not exist", (
        "detail 必须是 message 字符串（前端 client.ts:1233-1238 兜底）"
    )


def test_domain_error_body_has_error_struct(client: TestClient) -> None:
    """新 envelope: body.error.{code,message,trace_id,details}。"""
    resp = client.get("/raise_domain")
    body = resp.json()
    assert "error" in body
    err = body["error"]
    assert err["code"] == "preset.not_found"
    assert err["message"] == "preset 'foo' does not exist"
    assert err["trace_id"] is not None
    assert len(err["trace_id"]) == 24
    assert err["details"] == {"name": "foo"}


def test_domain_error_trace_id_matches_header(client: TestClient) -> None:
    """body.error.trace_id 必须等于 X-Trace-Id header — 前端两条路径取一处即可。"""
    resp = client.get("/raise_domain")
    assert resp.headers[TRACE_HEADER] == resp.json()["error"]["trace_id"]


def test_domain_error_no_details_when_empty(client: TestClient) -> None:
    """details 空时不输出该字段（不污染 JSON）。"""
    resp = client.get("/raise_conflict")
    body = resp.json()
    assert body["error"]["code"] == "conflict"
    assert "details" not in body["error"]


def test_validation_error_returns_422(client: TestClient) -> None:
    resp = client.get("/raise_validation")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation"
    assert resp.json()["error"]["details"] == {"field": "epoch"}


def test_domain_error_5xx_logs_exception(
    client: TestClient, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="studio.api.exception_handlers"):
        resp = client.get("/raise_domain_500")
    assert resp.status_code == 503
    errors = [r for r in caplog.records
              if r.name == "studio.api.exception_handlers" and r.levelname == "ERROR"]
    assert errors, "5xx DomainError 应 logger.exception (ERROR level)"


def test_domain_error_4xx_logs_info_not_exception(
    client: TestClient, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="studio.api.exception_handlers"):
        client.get("/raise_domain")
    errors = [r for r in caplog.records
              if r.name == "studio.api.exception_handlers" and r.levelname == "ERROR"]
    assert errors == [], "4xx 不应 logger.exception（业务正常路径，不该 ERROR 噪音）"


# ── HTTPException 形态保护（前端 client.ts 老路径不破）────────────────


def test_http_exception_string_detail_preserved(client: TestClient) -> None:
    """raise HTTPException(detail='str') 仍返 {detail: str}，handler 不截胡。"""
    resp = client.get("/raise_http")
    assert resp.status_code == 400
    body = resp.json()
    assert body == {"detail": "legacy http err string"}, (
        f"HTTPException 形状必须不变，实际 {body}"
    )
    # 应该 *没有* error 字段（不被 DomainError handler 处理）
    assert "error" not in body


def test_http_exception_dict_detail_preserved(client: TestClient) -> None:
    """raise HTTPException(detail={"error": "x", ...}) 仍 {detail: {...}}。

    test_studio_server.py 7 处断言 body['detail']['error'] 走这条；不能变。
    """
    resp = client.get("/raise_http_dict")
    body = resp.json()
    assert body == {"detail": {"error": "running_tasks_present", "count": 3}}


def test_http_exception_still_has_trace_id_header(client: TestClient) -> None:
    """HTTPException 也带 X-Trace-Id（middleware 在外层自动加）。"""
    resp = client.get("/raise_http")
    assert TRACE_HEADER in resp.headers
    assert len(resp.headers[TRACE_HEADER]) == 24


# ── Exception fallback ────────────────────────────────────────────────


def test_uncaught_exception_returns_500_dual_envelope(client: TestClient) -> None:
    resp = client.get("/raise_uncaught")
    assert resp.status_code == 500
    body = resp.json()
    assert "detail" in body
    assert "error" in body
    assert body["error"]["code"] == "internal.server_error"
    assert body["error"]["trace_id"] is not None


def test_uncaught_exception_body_does_not_leak_traceback(client: TestClient) -> None:
    """脱敏 — body 不能含 RuntimeError 字面 / "oops" 字面 / 任何 stack 关键字。"""
    resp = client.get("/raise_uncaught")
    body_text = resp.text
    for forbidden in ("RuntimeError", "oops", "Traceback", "File ", "line "):
        assert forbidden not in body_text, (
            f"500 body 不应 leak {forbidden!r}（开发者按 trace_id 查 studio.log）"
        )


def test_uncaught_exception_logs_with_traceback(
    client: TestClient, caplog: pytest.LogCaptureFixture,
) -> None:
    """开发者通过 trace_id 在 server log 找 traceback — 必须打 logger.exception。"""
    with caplog.at_level(logging.ERROR, logger="studio.api.exception_handlers"):
        client.get("/raise_uncaught")
    errors = [r for r in caplog.records
              if r.name == "studio.api.exception_handlers" and r.levelname == "ERROR"]
    assert errors, "fallback handler 必须 logger.exception"
    # 至少一条 record 带 exc_info
    assert any(r.exc_info for r in errors), "logger.exception 必须带 exc_info"


# ── 测 webui server 真实 router 注册后 baseline 不破 ────────────────


def test_existing_preset_404_path_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跑现有 preset 404 endpoint — handler 注册后 detail 形状保持。"""
    from studio import db, server
    from studio.api.routers import root as _root_router
    from studio.api.routers import samples as _samples_router
    from studio.services.presets import io as presets_io

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
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", tmp_path / "presets")

    c = TestClient(server.app)
    resp = c.get("/api/presets/__nonexistent__")
    assert resp.status_code == 404
    body = resp.json()
    # 现状（PR-2 C3 前）：presets.py 还在 raise HTTPException —— 形状保持 {detail: str}
    assert "detail" in body
    assert isinstance(body["detail"], str)
