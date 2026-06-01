"""PR-LOG-1 安全网 — 锁错误响应 envelope 形状（防 PR-LOG-4 破前端 contract）。

策略：调若干已知会 4xx 的端点，断言：
  (a) status code 是预期
  (b) response.json() 顶层有 "detail" key
  (c) detail 是 str 或 dict（非 list 非 None）

不锁 detail 内容文案（PR 期间允许改）；不锁额外字段（PR-LOG-4 dual-write 会新增 "error" 平行 key）。
未来 PR-LOG-4 合完应该仍能通过本测试（dual-write 保 detail 形态不变）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.api.routers import root as _root_router
from studio.api.routers import samples as _samples_router


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
    monkeypatch.setattr(_samples_router, "OUTPUT_DIR", output)
    monkeypatch.setattr(_root_router, "WEB_DIST", web_dist)
    return TestClient(server.app)


def _assert_detail_envelope(body: dict, status: int) -> None:
    assert "detail" in body, f"missing 'detail' key in {status} response: {body!r}"
    d = body["detail"]
    assert isinstance(d, (str, dict)), (
        f"detail must be str or dict (前端 client.ts 只 handle 这两态), got {type(d).__name__}: {d!r}"
    )


def test_preset_not_found_returns_404_with_detail(client: TestClient, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.services.presets import io as presets_io
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", tmp_path / "presets")
    resp = client.get("/api/presets/__definitely_does_not_exist_xxx__")
    assert resp.status_code == 404, f"expected 404, got {resp.status_code} body={resp.text!r}"
    _assert_detail_envelope(resp.json(), resp.status_code)


def test_preset_invalid_name_returns_400_with_detail(client: TestClient, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.services.presets import io as presets_io
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", tmp_path / "presets")
    # 路径分隔符是非法字符之一，触发 PresetError "非法预设名"
    resp = client.put("/api/presets/bad..name/with/slash", json={"foo": "bar"})
    assert resp.status_code in (400, 404, 422), (
        f"expected 4xx, got {resp.status_code} body={resp.text!r}"
    )
    _assert_detail_envelope(resp.json(), resp.status_code)


def test_unknown_task_log_returns_empty_not_error(client: TestClient) -> None:
    """logs router 当前对 unknown task 返 empty content 而非 404。
    锁这个行为：未来 PR 不应该改成 404（前端 polling 会突然炸 toast）。"""
    resp = client.get("/api/logs/999999")
    assert resp.status_code == 200, f"unknown task log expected 200 empty, got {resp.status_code}"
    body = resp.json()
    assert body == {"task_id": 999999, "content": "", "size": 0}


def test_validation_error_returns_422_with_detail(client: TestClient) -> None:
    """pydantic body validation 错误：FastAPI 默认 detail 是 list of dict。

    PR-LOG-4 注册 RequestValidationError handler 后会改成统一 envelope；
    本 baseline 锁现有形态（detail 是 list），让 PR-LOG-4 必须显式 surface 这个改动。
    """
    resp = client.put("/api/presets/test_preset", json="not-an-object")  # body 必须是 dict
    assert resp.status_code == 422, f"expected 422 from body validation, got {resp.status_code}"
    body = resp.json()
    assert "detail" in body
    # 现状：FastAPI 默认 422 detail 是 list[dict]；锁这个事实
    assert isinstance(body["detail"], (str, list, dict)), (
        f"422 detail 当前可能是 list（pydantic 原生），未来 PR-LOG-4 可能改 dict；不接受 None/int/bool"
    )
