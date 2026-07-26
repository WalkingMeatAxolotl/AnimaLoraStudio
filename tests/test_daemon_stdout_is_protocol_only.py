"""daemon 的 stdout 是协议流，第三方 logger 不许往里写。

`runtime/anima_daemon.py` 的契约是「stdout 仅协议；日志全走 stderr」。lycoris 在
import 时给 `LyCORIS` logger 挂 `StreamHandler(sys.stdout)` 且 `propagate=False`
（`lycoris/logging.py`）—— 每注入一次 LoRA 就有 5 行日志掉进协议流，server 侧
reader 逐行记「daemon stdout non-JSON」warning。

这里不 import daemon 模块本身（它会拉 torch + transformers，几十秒），只测那个
掰 handler 的函数 —— 它是纯 logging 操作，没有别的依赖。
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

_DAEMON = Path(__file__).resolve().parents[1] / "runtime" / "anima_daemon.py"


def _load_guard():
    """从 daemon 源码里取出 `_keep_third_party_logs_off_stdout`，不执行模块其余部分。"""
    src = _DAEMON.read_text(encoding="utf-8")
    marker = "def _keep_third_party_logs_off_stdout() -> None:"
    assert marker in src, "daemon 里没有这个函数了 —— stdout 保护是不是被删了？"
    start = src.index(marker)
    end = src.index("\n_keep_third_party_logs_off_stdout()", start)
    mod = ModuleType("daemon_guard")
    mod.__dict__.update({"logging": logging, "sys": sys})
    exec(compile(src[start:end], str(_DAEMON), "exec"), mod.__dict__)
    return mod._keep_third_party_logs_off_stdout


@pytest.fixture
def clean_lycoris_logger():
    lg = logging.getLogger("LyCORIS")
    saved = list(lg.handlers)
    lg.handlers.clear()
    yield lg
    lg.handlers.clear()
    lg.handlers.extend(saved)


def test_preinstalled_stderr_handler_makes_lycoris_skip_stdout(clean_lycoris_logger):
    """抢先挂 stderr handler → lycoris 的 `if not logger.handlers:` 整段跳过。

    这是主路径：本函数在 daemon 顶部跑，远早于 lycoris 被 import。
    """
    _load_guard()()

    assert clean_lycoris_logger.handlers, "该挂上一个 handler"
    assert all(
        getattr(h, "stream", None) is sys.stderr
        for h in clean_lycoris_logger.handlers
    )
    # 模拟 lycoris 那段守卫：有 handler 就不会再加 stdout 的
    assert clean_lycoris_logger.handlers  # → lycoris 的 `if not handlers` 为 False


def test_existing_stdout_handler_is_removed(clean_lycoris_logger):
    """兜底：万一 import 顺序变了、lycoris 先挂上了 stdout，也要掰回来。"""
    stdout_handler = logging.StreamHandler(sys.stdout)
    clean_lycoris_logger.addHandler(stdout_handler)

    _load_guard()()

    streams = [getattr(h, "stream", None) for h in clean_lycoris_logger.handlers]
    assert sys.stdout not in streams
    assert sys.stderr in streams


def test_unrelated_handlers_are_left_alone(clean_lycoris_logger):
    """只掰 stdout 的；别人挂的文件 / 内存 handler 不动。"""
    import io

    memory_handler = logging.StreamHandler(io.StringIO())
    clean_lycoris_logger.addHandler(memory_handler)

    _load_guard()()

    assert memory_handler in clean_lycoris_logger.handlers
    # 已有非 stdout handler → 不再多加一个 stderr 的
    assert len(clean_lycoris_logger.handlers) == 1
