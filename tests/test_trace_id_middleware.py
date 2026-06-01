"""PR-1 C5 — trace_id ContextVar + Middleware + Filter 验证。

覆盖：
  - new_trace_id() 格式与稳定性
  - bind / get / reset 基本 API
  - ContextFilter 注入 trace_id 到 LogRecord
  - TraceIdMiddleware：读 X-Trace-Id / 生成新 / 写回 response header
  - 跨 endpoint 的 contextvar 隔离（一个请求 bind 不污染其它）
  - HTTP 错误响应也带 X-Trace-Id（middleware 在所有响应外层）
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.infrastructure.logging import (
    ContextFilter,
    TRACE_HEADER,
    _reset_for_tests,
    bind_trace_id,
    get_trace_id,
    new_trace_id,
    reset_trace_id,
)


# ── new_trace_id ──────────────────────────────────────────────────────────


def test_new_trace_id_is_24_hex_chars() -> None:
    tid = new_trace_id()
    assert len(tid) == 24
    assert all(c in "0123456789abcdef" for c in tid)


def test_new_trace_id_is_unique() -> None:
    ids = {new_trace_id() for _ in range(100)}
    assert len(ids) == 100, "uuid4-based id 应几乎不冲突"


# ── bind / get / reset ───────────────────────────────────────────────────


def test_get_trace_id_default_none() -> None:
    _reset_for_tests()
    assert get_trace_id() is None


def test_bind_and_get_trace_id() -> None:
    _reset_for_tests()
    token = bind_trace_id("test-trace-001")
    try:
        assert get_trace_id() == "test-trace-001"
    finally:
        reset_trace_id(token)
    assert get_trace_id() is None, "reset 后应回到 None"


# ── ContextFilter ─────────────────────────────────────────────────────────


def test_context_filter_injects_trace_id_into_record() -> None:
    _reset_for_tests()
    token = bind_trace_id("filter-test-tid")
    try:
        f = ContextFilter()
        record = logging.LogRecord(
            name="studio.x", level=logging.INFO, pathname="/x.py", lineno=1,
            msg="hi", args=(), exc_info=None,
        )
        f.filter(record)
        assert record.trace_id == "filter-test-tid"
    finally:
        reset_trace_id(token)


def test_context_filter_does_not_overwrite_existing_trace_id() -> None:
    """如果 caller 显式 logger.info(..., extra={"trace_id": "X"}) 已经设了，filter 不改。"""
    f = ContextFilter()
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname="/x.py", lineno=1,
        msg="", args=(), exc_info=None,
    )
    record.trace_id = "explicit-override"
    f.filter(record)
    assert record.trace_id == "explicit-override"


def test_context_filter_when_no_trace_bound() -> None:
    _reset_for_tests()
    f = ContextFilter()
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname="/x.py", lineno=1,
        msg="", args=(), exc_info=None,
    )
    f.filter(record)
    assert record.trace_id is None


# ── TraceIdMiddleware ────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    output = tmp_path / "output"
    samples_dir = output / "samples"
    web_dist = tmp_path / "web_dist"
    samples_dir.mkdir(parents=True)
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setattr(server, "WEB_DIST", web_dist)
    from studio.api.routers import root as _root_router
    from studio.api.routers import samples as _samples_router
    monkeypatch.setattr(_samples_router, "OUTPUT_DIR", output)
    monkeypatch.setattr(_root_router, "WEB_DIST", web_dist)
    return TestClient(server.app)


def test_middleware_adds_trace_id_header_to_response(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert TRACE_HEADER in resp.headers, (
        f"middleware 应给所有响应加 {TRACE_HEADER}；实际 headers: {dict(resp.headers)}"
    )
    tid = resp.headers[TRACE_HEADER]
    assert len(tid) == 24


def test_middleware_echoes_client_trace_id(client: TestClient) -> None:
    """前端传 X-Trace-Id → 后端用该值，response 回带同样 id。"""
    custom_tid = "client-provided-trace-id-x"
    resp = client.get("/api/health", headers={TRACE_HEADER: custom_tid})
    assert resp.headers[TRACE_HEADER] == custom_tid


def test_middleware_generates_new_when_client_omits(client: TestClient) -> None:
    """client 不传 → 后端生成新 24 字符 hex。"""
    resp = client.get("/api/health")
    tid = resp.headers[TRACE_HEADER]
    assert len(tid) == 24
    assert all(c in "0123456789abcdef" for c in tid)


def test_middleware_isolates_trace_id_between_requests(client: TestClient) -> None:
    """连续两个请求 trace_id 不同 — 验证 ContextVar 不 leak。"""
    r1 = client.get("/api/health")
    r2 = client.get("/api/health")
    assert r1.headers[TRACE_HEADER] != r2.headers[TRACE_HEADER]


def test_middleware_adds_trace_id_on_4xx_response(client: TestClient,
                                                    tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """4xx 错误响应也必须带 trace_id（debug 价值最大的场景）。"""
    from studio.services.presets import io as presets_io
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", tmp_path / "presets")
    resp = client.get("/api/presets/__nonexistent_for_trace_test__")
    assert resp.status_code == 404
    assert TRACE_HEADER in resp.headers, "4xx 也必须带 trace_id 给用户截图"
    assert len(resp.headers[TRACE_HEADER]) == 24


# ── Filter 装到 setup_logging 后 record 自动带 ──────────────────────────


def test_setup_logging_filter_picks_up_contextvar(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """setup_logging 后 logger.x 调用自动带 trace_id 进 JSON line。"""
    monkeypatch.delenv("ANIMA_LOGGING_NO_BOOTSTRAP", raising=False)
    _reset_for_tests()
    from studio.infrastructure.logging import setup_logging, STUDIO_LOG_NAME
    import json
    setup_logging("pytest-c5", log_dir=tmp_path, console=False)

    token = bind_trace_id("ctx-via-setup-tid12345678")
    try:
        logging.getLogger("studio.test_c5").info("hello with trace")
    finally:
        reset_trace_id(token)

    # flush
    for h in logging.getLogger().handlers:
        h.flush()
    line = (tmp_path / STUDIO_LOG_NAME).read_text(encoding="utf-8").strip()
    out = json.loads(line)
    assert out["trace_id"] == "ctx-via-setup-tid12345678"
    assert out["msg"] == "hello with trace"

    # cleanup
    saved_handlers = list(logging.getLogger().handlers)
    saved_filters = list(logging.getLogger().filters)
    _reset_for_tests()
    logging.getLogger().handlers = []
    for f in saved_filters:
        logging.getLogger().removeFilter(f)
