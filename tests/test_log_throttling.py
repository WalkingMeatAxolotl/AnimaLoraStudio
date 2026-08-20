"""刀 1 机制修复：_ErrorThrottle / event_bus 丢弃状态化 / system_stats 熔断。"""
from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace

import pytest


# ── _ErrorThrottle ───────────────────────────────────────────────────
class TestErrorThrottle:
    def _make(self, monkeypatch, now: list[float]):
        from studio.supervisor.core import _ErrorThrottle
        import studio.supervisor.core as core
        monkeypatch.setattr(core.time, "monotonic", lambda: now[0])
        return _ErrorThrottle("test site")

    def test_first_full_then_silent_within_window(self, monkeypatch, caplog):
        now = [1000.0]
        th = self._make(monkeypatch, now)
        with caplog.at_level(logging.DEBUG, logger="studio.supervisor.core"):
            for _ in range(5):
                try:
                    raise ValueError("x")
                except ValueError:
                    th.failed()
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR  # exception

    def test_window_summary_and_recovery(self, monkeypatch, caplog):
        now = [1000.0]
        th = self._make(monkeypatch, now)
        with caplog.at_level(logging.DEBUG, logger="studio.supervisor.core"):
            for _ in range(3):
                try:
                    raise ValueError("x")
                except ValueError:
                    th.failed()
            now[0] += 61.0
            try:
                raise ValueError("x")
            except ValueError:
                th.failed()
            th.recovered()
        levels = [r.levelno for r in caplog.records]
        assert levels == [logging.ERROR, logging.WARNING, logging.INFO]
        assert "count=4" in caplog.records[1].message

    def test_exception_type_change_relogs_full(self, monkeypatch, caplog):
        now = [1000.0]
        th = self._make(monkeypatch, now)
        with caplog.at_level(logging.DEBUG, logger="studio.supervisor.core"):
            try:
                raise ValueError("x")
            except ValueError:
                th.failed()
            try:
                raise KeyError("y")
            except KeyError:
                th.failed()
        assert [r.levelno for r in caplog.records] == [logging.ERROR, logging.ERROR]


# ── event_bus 丢弃状态化 ─────────────────────────────────────────────
class TestEventBusDropState:
    def test_one_congestion_two_lines(self, caplog):
        from studio.infrastructure.event_bus import _safe_put

        async def scenario():
            q: asyncio.Queue = asyncio.Queue(maxsize=1)
            with caplog.at_level(logging.WARNING, logger="studio.infrastructure.event_bus"):
                _safe_put(q, {"type": "a"})          # 放进去
                for i in range(10):                   # 连续丢 10 个
                    _safe_put(q, {"type": f"drop{i}"})
                q.get_nowait()                        # 消费者恢复
                _safe_put(q, {"type": "b"})          # 恢复 → 汇总条
        asyncio.run(scenario())
        msgs = [r.message for r in caplog.records]
        assert len(msgs) == 2
        assert "dropping events" in msgs[0]
        assert "dropped 10 event(s)" in msgs[1]
        assert "drop9" in msgs[1]  # last_type


# ── system_stats 熔断 ────────────────────────────────────────────────
@pytest.fixture()
def reset_stats_state():
    import studio.services.system_stats as ss
    ss._GPU_FAILS[0] = 0
    ss._GPU_DISABLED[0] = False
    ss._PSUTIL_FAILS[0] = 0
    ss._PSUTIL_DISABLED[0] = False
    yield ss
    ss._GPU_FAILS[0] = 0
    ss._GPU_DISABLED[0] = False
    ss._PSUTIL_FAILS[0] = 0
    ss._PSUTIL_DISABLED[0] = False


class TestSystemStatsFuse:
    def test_gpu_fuse_after_three_failures(self, reset_stats_state, monkeypatch, caplog):
        ss = reset_stats_state
        monkeypatch.setattr(ss, "_ensure_nvml", lambda: True)
        # 无 nvmlDeviceGetCount 属性 → 函数体 AttributeError 走 except
        monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace())
        with caplog.at_level(logging.DEBUG, logger="studio.services.system_stats"):
            for _ in range(5):
                assert ss._collect_gpu() is None
        assert ss._GPU_DISABLED[0] is True
        msgs = [r.message for r in caplog.records]
        assert len(msgs) == 2  # 首条 + 熔断条，之后零日志
        assert "sampling disabled" in msgs[1]

    def test_psutil_fuse_reports_zeros(self, reset_stats_state, monkeypatch, caplog):
        ss = reset_stats_state
        def boom(*a, **kw):
            raise RuntimeError("psutil down")
        monkeypatch.setattr(ss.psutil, "cpu_percent", boom)
        monkeypatch.setattr(ss, "_collect_gpu", lambda: None)
        with caplog.at_level(logging.DEBUG, logger="studio.services.system_stats"):
            for _ in range(5):
                s = ss.collect_stats()
        assert s.cpu_pct == 0.0
        assert ss._PSUTIL_DISABLED[0] is True
        msgs = [r.message for r in caplog.records]
        assert len(msgs) == 2
        assert "reporting zeros" in msgs[1]
