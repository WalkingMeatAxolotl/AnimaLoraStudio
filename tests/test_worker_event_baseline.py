"""PR-LOG-1 安全网 — 锁 `__EVENT__:` IPC 协议契约（防 PR-LOG-2/3 破 worker→supervisor 通信）。

`__EVENT__:` 是 worker 子进程通过 stdout 给 supervisor 传结构化事件的协议（ADR-0006）。
前端通过 `GET /api/logs/{task_id}` 拿 task log 时，logs router 显式过滤掉这些行
（用户在 UI 看的是人读日志，不是 IPC 协议）。

PR-LOG-5 worker 接入 logger 时必须保留这层契约：
  - cmd_builder._EVENT_MARKER 字串还是 "__EVENT__:"
  - logs router 仍过滤该前缀
  - worker 输出此前缀的行必须走 print（不走 logger，否则前缀被 format 污染）
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.api.routers import logs as _logs_router
from studio.paths import LOGS_DIR
from studio.supervisor.cmd_builder import _EVENT_MARKER


def test_event_marker_string_is_double_underscore_event_colon() -> None:
    """字面值锁死。改这串就要同步改 runtime/training/snapshot.py + workers/preprocess_worker.py + logs.py。"""
    assert _EVENT_MARKER == "__EVENT__:", (
        f"_EVENT_MARKER 变了：{_EVENT_MARKER!r}。改这个值会让 worker IPC 全炸；"
        "必须同步 runtime/training/snapshot.py:EVENT_MARKER + workers/preprocess_worker.py + api/routers/logs.py"
    )


def test_logs_router_filters_event_lines(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """logs router 必须去掉 `__EVENT__:` 行，只返人读日志。"""
    monkeypatch.setattr(_logs_router, "LOGS_DIR", tmp_path)

    # 写一个混合 log：业务行 + EVENT 行 + 业务行
    fake_log = tmp_path / "42.log"
    fake_log.write_text(
        "[start] tagging 100 images\n"
        "__EVENT__:pause_state:{\"is_pausable\":true}\n"
        "tagged 50/100\n"
        "__EVENT__:progress:{\"step\":50}\n"
        "[done] tagged 100 images\n",
        encoding="utf-8",
    )

    # 用 monkeypatched server.app
    monkeypatch.setattr(server.db, "STUDIO_DB", tmp_path / "studio.db")
    db.init_db(tmp_path / "studio.db")
    client = TestClient(server.app)

    resp = client.get("/api/logs/42")
    assert resp.status_code == 200
    body = resp.json()
    content = body["content"]
    assert "[start] tagging 100 images" in content
    assert "tagged 50/100" in content
    assert "[done] tagged 100 images" in content
    assert "__EVENT__:" not in content, "logs router 必须过滤 EVENT 行；前端用户看到 IPC 协议字串就是 bug"
    assert "pause_state" not in content
    assert "progress" not in content


def test_event_marker_unused_task_returns_empty_not_error(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """unknown task_id 返 200 empty，不是 404；前端 polling 依赖这个不抛错。"""
    monkeypatch.setattr(_logs_router, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(server.db, "STUDIO_DB", tmp_path / "studio.db")
    db.init_db(tmp_path / "studio.db")
    client = TestClient(server.app)

    resp = client.get("/api/logs/9999")
    assert resp.status_code == 200
    assert resp.json() == {"task_id": 9999, "content": "", "size": 0}
