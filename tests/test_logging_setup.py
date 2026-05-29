"""PR-1 C3 — setup_logging 完整实现测试。

覆盖：
  - JsonLineFormatter 10 字段输出
  - HumanConsoleFormatter 格式
  - setup_logging 幂等性
  - 第三方库 silence list
  - uvicorn logger 接管
  - sys.excepthook 注入
  - reconfigure_console_utf8 不 crash
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from studio.infrastructure.logging import (
    HumanConsoleFormatter,
    JsonLineFormatter,
    STUDIO_LOG_NAME,
    _NOISY_LOGGERS,
    _reset_for_tests,
    reconfigure_console_utf8,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_logging():
    """每个测试前后 reset，防 sentinel 累加 + 防污染。"""
    _reset_for_tests()
    saved_handlers = list(logging.getLogger().handlers)
    saved_level = logging.getLogger().level
    saved_excepthook = sys.excepthook
    yield
    _reset_for_tests()
    logging.getLogger().handlers = saved_handlers
    logging.getLogger().level = saved_level
    sys.excepthook = saved_excepthook


# ── JsonLineFormatter ─────────────────────────────────────────────────────


def test_json_formatter_emits_10_required_fields() -> None:
    fmt = JsonLineFormatter("webui")
    rec = logging.LogRecord(
        name="studio.test", level=logging.INFO, pathname="/x.py", lineno=1,
        msg="hello %s", args=("world",), exc_info=None,
    )
    out = json.loads(fmt.format(rec))
    assert set(out.keys()) >= {"ts", "level", "process", "pid", "trace_id", "logger", "msg"}
    assert out["level"] == "INFO"
    assert out["process"] == "webui"
    assert out["logger"] == "studio.test"
    assert out["msg"] == "hello world"
    assert out["trace_id"] is None  # C5 还没注入 ContextVar


def test_json_formatter_includes_exception_when_present() -> None:
    fmt = JsonLineFormatter("worker:tag/42")
    try:
        raise ValueError("boom")
    except ValueError:
        rec = logging.LogRecord(
            name="studio.x", level=logging.ERROR, pathname="/x.py", lineno=1,
            msg="failed", args=(), exc_info=sys.exc_info(),
        )
    out = json.loads(fmt.format(rec))
    assert "exc" in out
    assert out["exc"]["type"] == "ValueError"
    assert out["exc"]["message"] == "boom"
    assert "Traceback" in out["exc"]["traceback"]


def test_json_formatter_includes_extra_user_fields() -> None:
    fmt = JsonLineFormatter("webui")
    rec = logging.LogRecord(
        name="studio.x", level=logging.INFO, pathname="/x.py", lineno=1,
        msg="tagged", args=(), exc_info=None,
    )
    rec.image_path = "/path/img.png"
    rec.image_idx = 47
    out = json.loads(fmt.format(rec))
    assert "extra" in out
    assert out["extra"]["image_path"] == "/path/img.png"
    assert out["extra"]["image_idx"] == 47


def test_json_formatter_ts_is_iso_with_z() -> None:
    fmt = JsonLineFormatter("webui")
    rec = logging.LogRecord(
        name="x", level=logging.INFO, pathname="/x.py", lineno=1,
        msg="", args=(), exc_info=None,
    )
    out = json.loads(fmt.format(rec))
    assert out["ts"].endswith("Z")
    assert "T" in out["ts"]


# ── HumanConsoleFormatter ─────────────────────────────────────────────────


def test_human_formatter_includes_level_logger_msg() -> None:
    fmt = HumanConsoleFormatter()
    rec = logging.LogRecord(
        name="studio.api.foo", level=logging.WARNING, pathname="/x.py", lineno=1,
        msg="warn msg", args=(), exc_info=None,
    )
    out = fmt.format(rec)
    assert "WARNI" in out  # %(levelname)-5s = WARNI (truncated)
    assert "studio.api.foo" in out
    assert "warn msg" in out


# ── setup_logging ─────────────────────────────────────────────────────────


def test_setup_logging_writes_to_studio_log(tmp_path: Path) -> None:
    setup_logging("webui", log_dir=tmp_path, console=False)
    logger = logging.getLogger("studio.test_setup_smoke")
    logger.info("smoke")
    # flush 所有 handler
    for h in logging.getLogger().handlers:
        h.flush()

    log_file = tmp_path / STUDIO_LOG_NAME
    assert log_file.exists()
    line = log_file.read_text(encoding="utf-8").strip()
    out = json.loads(line)
    assert out["process"] == "webui"
    assert out["logger"] == "studio.test_setup_smoke"
    assert out["msg"] == "smoke"


def test_setup_logging_is_idempotent_for_same_process(tmp_path: Path) -> None:
    setup_logging("webui", log_dir=tmp_path, console=False)
    handlers_count = len(logging.getLogger().handlers)
    setup_logging("webui", log_dir=tmp_path, console=False)
    setup_logging("webui", log_dir=tmp_path, console=False)
    assert len(logging.getLogger().handlers) == handlers_count, (
        "同 process 重复调 setup_logging 不应累加 handler"
    )


def test_setup_logging_different_process_replaces_handlers(tmp_path: Path) -> None:
    """不同 process 名调（罕见 — pytest reload / worker 进程入口）替换 handler。"""
    setup_logging("webui", log_dir=tmp_path, console=False)
    count_a = len(logging.getLogger().handlers)
    setup_logging("worker:tag/1", log_dir=tmp_path, console=False)
    count_b = len(logging.getLogger().handlers)
    assert count_b == count_a, "不同 process 名也只装一套 handler（清掉再装）"


def test_setup_logging_silences_noisy_libs(tmp_path: Path) -> None:
    """root level=INFO 时第三方库被静音到 WARNING。"""
    # 先把所有 noisy logger reset 到 NOTSET，setup_logging 应该改成 WARNING
    for n in _NOISY_LOGGERS:
        logging.getLogger(n).setLevel(logging.NOTSET)
    setup_logging("webui", log_dir=tmp_path, console=False, level="INFO")
    for n in _NOISY_LOGGERS:
        assert logging.getLogger(n).level == logging.WARNING, (
            f"{n} 应被静音到 WARNING，实际 {logging.getLogger(n).level}"
        )


def test_setup_logging_takes_over_uvicorn_loggers(tmp_path: Path) -> None:
    """uvicorn.* logger handler 被清空 + propagate=True，让 root JSON handler 接管。"""
    # 模拟 uvicorn 启动后挂了自己 handler
    uv = logging.getLogger("uvicorn.access")
    fake_h = logging.StreamHandler()
    uv.handlers = [fake_h]
    uv.propagate = False

    setup_logging("webui", log_dir=tmp_path, console=False)

    assert uv.handlers == [], "uvicorn 自带 handler 应被清空"
    assert uv.propagate is True, "uvicorn logger 应 propagate 让 root 接管"


def test_setup_logging_console_false_no_console_handler(tmp_path: Path) -> None:
    setup_logging("webui", log_dir=tmp_path, console=False)
    handlers = logging.getLogger().handlers
    stream_handlers = [h for h in handlers if isinstance(h, logging.StreamHandler)
                       and not isinstance(h, logging.handlers.RotatingFileHandler)]
    # RotatingFileHandler 是 StreamHandler 子类 — 排除它
    from concurrent_log_handler import ConcurrentRotatingFileHandler
    pure_stream = [h for h in stream_handlers if not isinstance(h, ConcurrentRotatingFileHandler)]
    assert pure_stream == [], "console=False 不应装任何 stderr handler"


def test_setup_logging_installs_sys_excepthook(tmp_path: Path) -> None:
    """sys.excepthook 被替换为路由到 logger 的版本。"""
    original = sys.excepthook
    setup_logging("webui", log_dir=tmp_path, console=False)
    assert sys.excepthook is not original, "sys.excepthook 应被替换"


def test_setup_logging_excepthook_preserves_keyboardinterrupt(tmp_path: Path,
                                                                caplog: pytest.LogCaptureFixture) -> None:
    """Ctrl+C 不应被吞进 logger（用户体验）。"""
    setup_logging("webui", log_dir=tmp_path, console=False)
    # excepthook 拿到 KeyboardInterrupt 应该走原始 hook 不进 logger
    with caplog.at_level(logging.CRITICAL, logger="studio.unhandled"):
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            etype, evalue, etb = sys.exc_info()
            sys.excepthook(etype, evalue, etb)
    critical_records = [r for r in caplog.records if r.name == "studio.unhandled"]
    assert critical_records == [], "KeyboardInterrupt 不应路由到 logger.critical"


def test_setup_logging_extra_handlers_attached(tmp_path: Path) -> None:
    extra = logging.StreamHandler()
    setup_logging("worker:tag/1", log_dir=tmp_path, console=False, extra_handlers=[extra])
    assert extra in logging.getLogger().handlers


def test_reconfigure_console_utf8_does_not_crash() -> None:
    """无论 stdout/stderr 是何种 stream 都不应 crash（包括测试下的 pipe）。"""
    reconfigure_console_utf8()  # 不抛即通过
