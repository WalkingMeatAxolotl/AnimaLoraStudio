"""PR-1 C7 — supervisor error_msg 回写 + malformed event SSE + event_bus warn 验证。

3 个 B audit P1 修复：
  - B-1.6: _tail_log_for_error_msg + _finish_slot 拼到 db.tasks.error_msg
  - B-4.4: _on_task_log malformed event → SSE event_malformed
  - B-1.5: event_bus._safe_put QueueFull → logger.warning
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import pytest


# ── B-1.6: _tail_log_for_error_msg ─────────────────────────────────────


def test_tail_log_missing_file_returns_empty(tmp_path: Path) -> None:
    from studio.supervisor.core import _tail_log_for_error_msg
    assert _tail_log_for_error_msg(tmp_path / "nope.log") == ""


def test_tail_log_picks_traceback_section(tmp_path: Path) -> None:
    """末段有 Traceback 时优先取那一段（含完整 stack）。"""
    from studio.supervisor.core import _tail_log_for_error_msg
    log = tmp_path / "42.log"
    log.write_text(
        "[start] tagging 100 images\n"
        "[progress] 50/100\n"
        "[error] inference crashed on image 47\n"
        'Traceback (most recent call last):\n'
        '  File "wd14.py", line 184, in _infer_one\n'
        '    out = session.run(...)\n'
        'RuntimeError: ONNX op crash\n',
        encoding="utf-8",
    )
    out = _tail_log_for_error_msg(log)
    assert "Traceback" in out
    assert "RuntimeError" in out
    assert "ONNX op crash" in out


def test_tail_log_no_traceback_uses_last_lines(tmp_path: Path) -> None:
    """无 Traceback 字串 → 取末 N 行（默认 12）。"""
    from studio.supervisor.core import _tail_log_for_error_msg
    log = tmp_path / "x.log"
    lines = [f"line {i}" for i in range(30)]
    log.write_text("\n".join(lines), encoding="utf-8")
    out = _tail_log_for_error_msg(log, max_lines=5)
    out_lines = out.strip().splitlines()
    assert out_lines == ["line 25", "line 26", "line 27", "line 28", "line 29"]


def test_tail_log_truncates_to_max_chars(tmp_path: Path) -> None:
    from studio.supervisor.core import _tail_log_for_error_msg
    log = tmp_path / "x.log"
    log.write_text("Traceback (most recent call last):\n" + ("x" * 2000), encoding="utf-8")
    out = _tail_log_for_error_msg(log, max_chars=400)
    assert len(out) <= 400
    assert out.startswith("...")


# ── B-1.5: event_bus _safe_put QueueFull warn ──────────────────────────


def test_safe_put_logs_warning_on_queue_full(caplog: pytest.LogCaptureFixture) -> None:
    """QueueFull 不再静默丢；记 WARNING 行带 event type。"""
    from studio.infrastructure.event_bus import _safe_put

    async def _run():
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        q.put_nowait({"type": "first"})
        with caplog.at_level(logging.WARNING, logger="studio.infrastructure.event_bus"):
            _safe_put(q, {"type": "task_state_changed", "task_id": 42})
        warnings = [r for r in caplog.records
                    if r.name == "studio.infrastructure.event_bus" and r.levelname == "WARNING"]
        assert warnings, "QueueFull 必须 logger.warning 不能静默"
        assert "task_state_changed" in warnings[-1].getMessage()
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())


# ── B-4.4: malformed event SSE warn ────────────────────────────────────


def test_malformed_event_publishes_sse_event_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """worker 写错 __EVENT__: payload → supervisor _on_task_log catch + publish
    event_malformed 让前端可见（之前静默丢导致 UI 暂停按钮永远灰）。"""
    from studio.supervisor.core import Supervisor
    from unittest.mock import MagicMock

    events = []
    sup = Supervisor(on_event=events.append, db_path=tmp_path / "studio.db")
    sup._logs_dir = tmp_path
    slot = MagicMock()
    slot.id = 42

    callback = sup._make_task_log_callback(slot, 42)
    # 喂一行 malformed event（payload 不是合法 JSON）
    callback('__EVENT__:pause_state:{not json}')

    malformed = [e for e in events if e["type"] == "event_malformed"]
    assert malformed, f"应 publish event_malformed；实际 events: {events}"
    assert malformed[0]["task_id"] == 42
    assert "pause_state" in malformed[0]["raw_preview"]


def test_well_formed_event_does_not_publish_event_malformed(
    tmp_path: Path,
) -> None:
    """正常 event 不应触发 event_malformed。"""
    from studio.supervisor.core import Supervisor
    from unittest.mock import MagicMock

    events = []
    sup = Supervisor(on_event=events.append, db_path=tmp_path / "studio.db")
    sup._logs_dir = tmp_path
    slot = MagicMock()
    slot.id = 7
    slot.pause_state_path = None

    callback = sup._make_task_log_callback(slot, 7)
    callback('__EVENT__:pause_state:{"state_path": "/x.bin", "step": 100}')

    malformed = [e for e in events if e["type"] == "event_malformed"]
    assert not malformed
