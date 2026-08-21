"""TaskLog 通道（R7 收编）：级别转发、旧签名兼容、协议一致性。"""
from __future__ import annotations

import logging

import pytest

from studio.infrastructure.task_log import (
    NULL_LOG,
    CallbackTaskLog,
    TaskLog,
    TaskLogLike,
)


@pytest.fixture()
def log_and_records(caplog):
    logger = logging.getLogger("test.task_log")
    caplog.set_level(logging.DEBUG, logger="test.task_log")
    return TaskLog(logger), caplog


class TestTaskLog:
    def test_call_is_info(self, log_and_records):
        tl, caplog = log_and_records
        tl("plain line")
        (rec,) = caplog.records
        assert rec.levelno == logging.INFO
        assert rec.message == "plain line"

    def test_level_methods_forward(self, log_and_records):
        tl, caplog = log_and_records
        tl.debug("d %s", 1)
        tl.info("i %s", 2)
        tl.warning("w %s", 3)
        tl.error("e %s", 4)
        levels = [(r.levelno, r.message) for r in caplog.records]
        assert levels == [
            (logging.DEBUG, "d 1"),
            (logging.INFO, "i 2"),
            (logging.WARNING, "w 3"),
            (logging.ERROR, "e 4"),
        ]

    def test_exc_info_attaches_traceback(self, log_and_records):
        tl, caplog = log_and_records
        try:
            raise ValueError("boom")
        except ValueError:
            tl.warning("failed", exc_info=True)
            tl.error("failed hard", exc_info=True)
        assert all(r.exc_info for r in caplog.records)

    def test_satisfies_protocol(self, log_and_records):
        tl, _ = log_and_records
        assert isinstance(tl, TaskLogLike)


class TestCallbackTaskLog:
    def test_all_levels_degrade_to_callback(self):
        lines: list[str] = []
        cb = CallbackTaskLog(lines.append)
        cb("a")
        cb.debug("b %s", 1)
        cb.info("c")
        cb.warning("d", exc_info=True)
        cb.error("e %s %s", 1, 2)
        assert lines == ["a", "b 1", "c", "d", "e 1 2"]

    def test_satisfies_protocol(self):
        assert isinstance(CallbackTaskLog(lambda _l: None), TaskLogLike)

    def test_null_log_swallows(self):
        NULL_LOG("x")
        NULL_LOG.error("y", exc_info=True)
