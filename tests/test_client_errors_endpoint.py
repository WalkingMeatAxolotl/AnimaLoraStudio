"""PR-3 C1 — /api/client-errors endpoint 测试。

覆盖：
  - POST 合法 body → 204 + logger.error 一条到 studio.client
  - 非 JSON / 空 body → 204 silently swallow
  - per-IP 10/min 限流，第 11 次 silently drop
  - stack / componentStack 字段截断
  - logger.error 带 extra dict 含识别字段
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """裸 FastAPI app + client_errors router（不挂业务，速度快）。"""
    from studio.api.routers.client_errors import _reset_rate_limit_for_tests, router

    _reset_rate_limit_for_tests()
    a = FastAPI()
    a.include_router(router)
    return TestClient(a)


# ── 基本上报 ──────────────────────────────────────────────────────────


def test_post_valid_body_returns_204(client: TestClient,
                                       caplog: pytest.LogCaptureFixture) -> None:
    body = {
        "kind": "react.boundary",
        "message": "Cannot read properties of undefined",
        "stack": "TypeError: ... at TrainingMonitor (foo.tsx:42)",
        "componentStack": "\n at TrainingMonitor\n at App",
        "url": "http://localhost:8765/studio/monitor/42",
        "user_agent": "Mozilla/5.0 ...",
        "client_ts": "2026-05-28T14:33:01.090Z",
        "app_version": "0.12.0",
        "build_hash": "abc1234",
    }
    with caplog.at_level(logging.ERROR, logger="studio.client"):
        resp = client.post("/api/client-errors", json=body)

    assert resp.status_code == 204
    assert resp.content == b""

    records = [r for r in caplog.records if r.name == "studio.client"]
    assert len(records) == 1
    rec = records[0]
    assert rec.levelname == "ERROR"
    assert "Cannot read properties" in rec.getMessage()
    assert "[react.boundary]" in rec.getMessage()
    # extra 字段
    assert getattr(rec, "client_kind", None) == "react.boundary"
    assert getattr(rec, "client_url", None) == "http://localhost:8765/studio/monitor/42"
    assert getattr(rec, "client_app_version", None) == "0.12.0"
    assert getattr(rec, "client_build_hash", None) == "abc1234"
    assert "TypeError" in getattr(rec, "client_stack", "")
    assert "TrainingMonitor" in getattr(rec, "client_componentStack", "")


def test_post_minimal_body_still_204(client: TestClient) -> None:
    """只传 message 也工作（kind 默认 manual）。"""
    resp = client.post("/api/client-errors", json={"message": "test"})
    assert resp.status_code == 204


def test_post_empty_dict_returns_204(client: TestClient) -> None:
    """空 dict 也不挂 — message 默认 "(no message)"。"""
    resp = client.post("/api/client-errors", json={})
    assert resp.status_code == 204


def test_post_non_json_body_silently_swallows(
    client: TestClient, caplog: pytest.LogCaptureFixture,
) -> None:
    """malformed JSON → 204；后端 warning 记一行不报错给前端。"""
    with caplog.at_level(logging.WARNING, logger="studio.client"):
        resp = client.post("/api/client-errors", content=b"not json at all")
    assert resp.status_code == 204
    assert any("malformed" in r.getMessage() for r in caplog.records
               if r.name == "studio.client")


def test_post_non_dict_json_returns_204(client: TestClient) -> None:
    """body 是 list/string 而非 dict → 204 silently swallow。"""
    resp = client.post("/api/client-errors", json=["a", "b"])
    assert resp.status_code == 204
    resp2 = client.post("/api/client-errors", json="just a string")
    assert resp2.status_code == 204


# ── 字段截断 ──────────────────────────────────────────────────────────


def test_long_fields_truncated(client: TestClient,
                                 caplog: pytest.LogCaptureFixture) -> None:
    huge_stack = "x" * 10000
    huge_msg = "y" * 5000
    with caplog.at_level(logging.ERROR, logger="studio.client"):
        client.post("/api/client-errors", json={
            "message": huge_msg, "stack": huge_stack,
        })
    rec = [r for r in caplog.records if r.name == "studio.client"][0]
    assert len(rec.getMessage()) < len(huge_msg) + 100  # truncated to 1000
    assert len(getattr(rec, "client_stack", "")) <= 4000


# ── 限流 ──────────────────────────────────────────────────────────────


def test_rate_limit_kicks_in_after_10_per_ip(client: TestClient,
                                                caplog: pytest.LogCaptureFixture) -> None:
    from studio.api.routers.client_errors import _reset_rate_limit_for_tests
    _reset_rate_limit_for_tests()

    with caplog.at_level(logging.INFO, logger="studio.client"):
        # 前 10 个都 200 + ERROR record
        for i in range(10):
            resp = client.post("/api/client-errors", json={"message": f"err {i}"})
            assert resp.status_code == 204
        error_records = [r for r in caplog.records
                         if r.name == "studio.client" and r.levelname == "ERROR"]
        assert len(error_records) == 10

        # 第 11 个仍 204 但不进 ERROR；进 INFO "rate-limit drop"
        caplog.clear()
        resp = client.post("/api/client-errors", json={"message": "over the limit"})
        assert resp.status_code == 204
        error_after = [r for r in caplog.records
                       if r.name == "studio.client" and r.levelname == "ERROR"]
        assert error_after == [], "rate-limit 后不应产生 ERROR record"
        info_records = [r for r in caplog.records
                        if r.name == "studio.client" and r.levelname == "INFO"]
        assert any("rate-limit drop" in r.getMessage() for r in info_records)


def test_rate_limit_per_ip_isolated(client: TestClient) -> None:
    """同一 TestClient 默认 X-Forwarded-For 不变 IP 共享；显式传不同 IP 验隔离。"""
    from studio.api.routers.client_errors import _reset_rate_limit_for_tests
    _reset_rate_limit_for_tests()

    # IP A 用满 10
    for i in range(10):
        client.post("/api/client-errors",
                    json={"message": f"a {i}"},
                    headers={"X-Forwarded-For": "1.2.3.4"})
    # IP B 仍能上报
    resp = client.post("/api/client-errors",
                        json={"message": "b"},
                        headers={"X-Forwarded-For": "5.6.7.8"})
    assert resp.status_code == 204


def test_rate_limit_window_recovers(client: TestClient,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """60s 窗口后旧 timestamp 出窗，limit 自动恢复。"""
    from studio.api.routers import client_errors as ce
    ce._reset_rate_limit_for_tests()

    fake_t = [1000.0]
    monkeypatch.setattr(ce.time, "monotonic", lambda: fake_t[0])

    # 用满 10
    for _ in range(10):
        client.post("/api/client-errors", json={"message": "x"})

    # 第 11 拒
    assert ce._rate_limit_ok("testclient") is False

    # 时间推进 61s 窗口外
    fake_t[0] += 61.0
    assert ce._rate_limit_ok("testclient") is True


# ── 真实 webui app 端到端 ─────────────────────────────────────────────


def test_endpoint_registered_on_webui_app() -> None:
    """app.py 必须 include_router(client_errors)，route 在 app.routes 内。

    FastAPI 0.137+ 把 include_router 包成 `_IncludedRouter` wrapper —— 顶层
    iter 看不到子 path，必须递归展开（详 tests/_route_helpers.py）。
    """
    from studio.api.app import app

    from ._route_helpers import iter_leaf_routes

    paths = {r.path for r in iter_leaf_routes(app.routes) if hasattr(r, "path")}
    assert "/api/client-errors" in paths
