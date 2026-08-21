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
    callback('__EVENT__:pause_state:{not json}', 0)

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
    callback('__EVENT__:pause_state:{"state_path": "/x.bin", "step": 100}', 0)

    malformed = [e for e in events if e["type"] == "event_malformed"]
    assert not malformed


# ── 日志目标态刀 2：error_msg 取最后一个错误块；SSE 形状统一 ────────────────


_H = "2026-08-19 15:00:00.000 "


def test_tail_log_prefers_last_error_record_block(tmp_path: Path) -> None:
    """行契约落地后：最后一条 ERROR 记录（行头去前缀 + 续行）优先于末 N 行。"""
    from studio.supervisor.core import _tail_log_for_error_msg
    log = tmp_path / "run.log"
    log.write_text(
        _H + "INFO  training.loop: step=1\n"
        + _H + "ERROR training.bootstrap: 配置校验失败（2 处）:\n"
        + "  epochs: must be > 0\n"
        + "  lr: must be > 0\n"
        + _H + "INFO  training.loop: bye\n",
        encoding="utf-8",
    )
    out = _tail_log_for_error_msg(log)
    assert out.splitlines() == [
        "配置校验失败（2 处）:", "  epochs: must be > 0", "  lr: must be > 0",
    ]


def test_tail_log_error_record_with_traceback_continuation(tmp_path: Path) -> None:
    """logger.exception 的 traceback 是该记录的续行，整块进 error_msg。"""
    from studio.supervisor.core import _tail_log_for_error_msg
    log = tmp_path / "run.log"
    log.write_text(
        _H + "ERROR studio.workers.tag_worker: job crashed\n"
        + "Traceback (most recent call last):\n"
        + '  File "x.py", line 1\n'
        + "RuntimeError: boom\n",
        encoding="utf-8",
    )
    out = _tail_log_for_error_msg(log)
    assert out.startswith("job crashed\nTraceback")
    assert out.endswith("RuntimeError: boom")


def test_tail_log_raw_traceback_after_error_record_wins(tmp_path: Path) -> None:
    """早先一条可恢复的 ERROR 记录，之后子进程裸崩（traceback 不经 logger）：取后者。"""
    from studio.supervisor.core import _tail_log_for_error_msg
    log = tmp_path / "run.log"
    log.write_text(
        _H + "ERROR training.loop: recoverable thing\n"
        + _H + "INFO  training.loop: continuing\n"
        + "Traceback (most recent call last):\n"
        + "ValueError: real crash\n",
        encoding="utf-8",
    )
    out = _tail_log_for_error_msg(log)
    assert out.startswith("Traceback") and out.endswith("ValueError: real crash")


def test_tail_log_reads_only_file_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """上百 MB 的 run.log 不整读：只看尾部窗口。"""
    from studio.supervisor import core as core_mod
    monkeypatch.setattr(core_mod, "_ERROR_MSG_TAIL_BYTES", 128)
    log = tmp_path / "run.log"
    body = (_H + "ERROR training.loop: too-old-to-see\n") + ("x" * 500 + "\n") + (_H + "ERROR training.loop: recent\n")
    log.write_text(body, encoding="utf-8")
    assert core_mod._tail_log_for_error_msg(log) == "recent"


def test_task_and_job_log_events_carry_seq_and_end_offset(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from studio.supervisor.core import Supervisor
    events: list = []
    sup = Supervisor(on_event=events.append, db_path=tmp_path / "studio.db")
    sup._make_task_log_callback(MagicMock(), 1)("hello", 123)
    sup._make_job_log_callback(2, 3, None, "tag")("world", 456)
    t = next(e for e in events if e["type"] == "task_log_appended")
    j = next(e for e in events if e["type"] == "job_log_appended")
    assert t["end_offset"] == 123 and isinstance(t["seq"], int)
    assert j["end_offset"] == 456 and isinstance(j["seq"], int)
    assert t["seq"] < j["seq"]


def test_job_malformed_event_publishes_event_malformed(tmp_path: Path) -> None:
    """job 路径之前只 logger.exception 不 emit；现与 task 路径对齐。"""
    from studio.supervisor.core import Supervisor
    events: list = []
    sup = Supervisor(on_event=events.append, db_path=tmp_path / "studio.db")
    sup._make_job_log_callback(9, 1, None, "download")("__EVENT__:progress:{nope", 0)
    m = [e for e in events if e["type"] == "event_malformed"]
    assert m and m[0]["job_id"] == 9 and "progress" in m[0]["raw_preview"]
    assert not [e for e in events if e["type"] == "job_log_appended"]


def test_event_broadcast_error_is_not_reported_as_malformed(tmp_path: Path) -> None:
    """广播自身抛错不能被误报成 malformed marker（try 只包解析）。"""
    from studio.supervisor.core import Supervisor

    def boom(evt):
        if evt["type"] == "progress":
            raise RuntimeError("sse down")

    sup = Supervisor(on_event=boom, db_path=tmp_path / "studio.db")
    with pytest.raises(RuntimeError, match="sse down"):
        sup._make_job_log_callback(9, 1, None, "download")('__EVENT__:progress:{"a":1}', 0)
