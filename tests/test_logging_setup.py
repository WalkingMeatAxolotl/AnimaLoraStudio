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
    LOG_LEVEL_ENV,
    LOG_LINE_RE,
    OWN_LOGGER_NAMESPACES,
    STUDIO_LOG_NAME,
    _NOISY_LOGGERS,
    _reset_for_tests,
    reconfigure_console_utf8,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_logging(monkeypatch: pytest.MonkeyPatch):
    """每个测试前后 reset，防 sentinel 累加 + 防污染。

    conftest session fixture 设 ANIMA_LOGGING_NO_BOOTSTRAP=1 让业务代码
    setup_logging 全部 noop。本文件直接测 setup_logging 行为，必须 unset。
    """
    monkeypatch.delenv("ANIMA_LOGGING_NO_BOOTSTRAP", raising=False)
    _reset_for_tests()
    saved_handlers = list(logging.getLogger().handlers)
    saved_level = logging.getLogger().level
    saved_excepthook = sys.excepthook
    saved_levels = {
        n: logging.getLogger(n).level
        for n in (*OWN_LOGGER_NAMESPACES, *_NOISY_LOGGERS, "uvicorn", "uvicorn.error", "uvicorn.access")
    }
    yield
    _reset_for_tests()
    logging.getLogger().handlers = saved_handlers
    logging.getLogger().level = saved_level
    for n, lv in saved_levels.items():
        logging.getLogger(n).setLevel(lv)
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


# ── 级别模型（docs/design/logging-target-state.md §3.1）──────────────────


def _console_handlers() -> list[logging.Handler]:
    return [
        h for h in logging.getLogger().handlers
        if type(h) is logging.StreamHandler
    ]


def test_own_namespaces_debug_root_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """自家命名空间恒 DEBUG（记录不过滤），root 保持 INFO 防第三方 debug 洪水。"""
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    for n in OWN_LOGGER_NAMESPACES:
        logging.getLogger(n).setLevel(logging.NOTSET)
    setup_logging("webui", log_dir=tmp_path, console=False)
    assert logging.getLogger().level == logging.INFO
    for n in OWN_LOGGER_NAMESPACES:
        assert logging.getLogger(n).getEffectiveLevel() == logging.DEBUG, n
    # 未列入的第三方 logger 跟 root 走 INFO
    assert logging.getLogger("some_third_party_lib").getEffectiveLevel() == logging.INFO


def test_file_handler_records_own_debug(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    setup_logging("webui", log_dir=tmp_path, console=False)
    logging.getLogger("studio.test_dbg").debug("dbg-line")
    for h in logging.getLogger().handlers:
        h.flush()
    text = (tmp_path / STUDIO_LOG_NAME).read_text(encoding="utf-8")
    assert "dbg-line" in text


def test_console_level_defaults_info_and_reads_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    setup_logging("cli:x", log_dir=tmp_path, console=True, file=False)
    (h,) = _console_handlers()
    assert h.level == logging.INFO

    _reset_for_tests()
    monkeypatch.setenv(LOG_LEVEL_ENV, "debug")
    setup_logging("cli:y", log_dir=tmp_path, console=True, file=False)
    (h,) = _console_handlers()
    assert h.level == logging.DEBUG

    _reset_for_tests()
    monkeypatch.setenv(LOG_LEVEL_ENV, "bogus")
    setup_logging("cli:z", log_dir=tmp_path, console=True, file=False)
    (h,) = _console_handlers()
    assert h.level == logging.INFO, "非法 env 值回落 INFO"


def test_console_level_param_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "DEBUG")
    setup_logging("cli:w", log_dir=tmp_path, console=True, file=False, level="WARNING")
    (h,) = _console_handlers()
    assert h.level == logging.WARNING


def test_console_auto_is_human_even_when_piped(tmp_path: Path) -> None:
    """pipe 下不再输出 JSON（与 studio.log 重复且终端不可读）。测试进程 stderr 即非 tty。"""
    assert not sys.stderr.isatty()
    setup_logging("webui", log_dir=tmp_path, console="auto")
    (h,) = _console_handlers()
    assert isinstance(h.formatter, HumanConsoleFormatter)


def test_console_json_explicit(tmp_path: Path) -> None:
    setup_logging("webui", log_dir=tmp_path, console="json")
    (h,) = _console_handlers()
    assert isinstance(h.formatter, JsonLineFormatter)


def test_uvicorn_access_silenced_others_kept(tmp_path: Path) -> None:
    for n in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(n).setLevel(logging.NOTSET)
    setup_logging("webui", log_dir=tmp_path, console=False)
    assert logging.getLogger("uvicorn.access").level == logging.WARNING
    assert logging.getLogger("uvicorn.error").getEffectiveLevel() == logging.INFO


def test_reconfigure_console_utf8_does_not_crash() -> None:
    """无论 stdout/stderr 是何种 stream 都不应 crash（包括测试下的 pipe）。"""
    reconfigure_console_utf8()  # 不抛即通过


# ── 行契约（docs/design/logging-target-state.md §3.2）────────────────────────


@pytest.mark.parametrize("level,expect", [
    (logging.DEBUG, "DEBUG"), (logging.INFO, "INFO"),
    (logging.WARNING, "WARNING"), (logging.ERROR, "ERROR"), (logging.CRITICAL, "CRITICAL"),
])
def test_human_line_matches_contract_regex(level: int, expect: str) -> None:
    fmt = HumanConsoleFormatter()
    rec = logging.LogRecord(
        name="training.progress", level=level, pathname="/x.py", lineno=1,
        msg="epoch=%d step=%d", args=(0, 50), exc_info=None,
    )
    line = fmt.format(rec)
    m = LOG_LINE_RE.match(line)
    assert m, line
    assert m["level"] == expect
    assert m["logger"] == "training.progress"
    assert m["msg"] == "epoch=0 step=50"


def test_contract_regex_rejects_continuation_lines() -> None:
    for cont in ("Traceback (most recent call last):", '  File "x.py", line 1', "ValueError: boom", ""):
        assert LOG_LINE_RE.match(cont) is None, cont
