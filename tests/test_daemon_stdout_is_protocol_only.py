"""daemon 的 stdout 是协议流，第三方 logger 不许往里写。

`runtime/anima_daemon.py` 的契约是「stdout 仅协议；日志全走 stderr」。lycoris 在
import 时给 `LyCORIS` logger 挂 `StreamHandler(sys.stdout)` 且 `propagate=False`
（`lycoris/logging.py`）—— 每注入一次 LoRA 就有 5 行日志掉进协议流，server 侧
reader 逐行记「daemon stdout non-JSON」warning。

这里不 import daemon 模块本身（它会拉 torch + transformers，几十秒），只测那个掰
handler 的函数 —— 它是纯 logging 操作，没有别的依赖。

**测试用自己的 logger 名，绝不碰真的 `LyCORIS`**：pytest 的 logging 插件会往真实
logger 上挂 `LogCaptureHandler`，断言「有几个 handler / 是不是 stderr」会随 pytest
版本和是否有别的测试用过 caplog 而变 —— 本地绿 CI 红的经典形态。
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

_DAEMON = Path(__file__).resolve().parents[1] / "runtime" / "anima_daemon.py"
# 专属名，不与任何真实库重名；propagate 保持默认 True，pytest 只会挂到 root 上
_PROBE = "anima_test.stdout_guard_probe"


def _load_guard():
    """从 daemon 源码里取出保护函数 + 名单，不执行模块其余部分。"""
    src = _DAEMON.read_text(encoding="utf-8")
    marker = "# 会往 stdout 写日志的第三方 logger。"
    assert marker in src, "daemon 里没有这段了 —— stdout 保护是不是被删了？"
    start = src.index(marker)
    end = src.index("\n_keep_third_party_logs_off_stdout()", start)
    mod = ModuleType("daemon_guard")
    mod.__dict__.update({"logging": logging, "sys": sys, "Iterable": list})
    exec(compile(src[start:end], str(_DAEMON), "exec"), mod.__dict__)
    return mod


@pytest.fixture
def probe() -> logging.Logger:
    lg = logging.getLogger(_PROBE)
    lg.handlers.clear()
    yield lg
    lg.handlers.clear()


def test_lycoris_is_in_the_guarded_list():
    """名单本身是这条保护的全部意义 —— 漏了 LyCORIS 等于没保护。"""
    assert "LyCORIS" in _load_guard()._STDOUT_NOISY_LOGGERS


def test_installs_a_stderr_handler_so_lycoris_skips_its_own(probe):
    """主路径：本函数在 lycoris import 之前跑。

    lycoris 那段是 `if not logger.handlers:` 守卫的，所以只要先占住 handler 位，
    它就不会再挂 stdout 的。
    """
    _load_guard()._keep_third_party_logs_off_stdout([_PROBE])

    assert len(probe.handlers) == 1
    assert probe.handlers[0].stream is sys.stderr


def test_existing_stdout_handler_is_removed(probe):
    """兜底：万一 import 顺序变了、lycoris 先挂上了 stdout，也要掰回来。"""
    probe.addHandler(logging.StreamHandler(sys.stdout))

    _load_guard()._keep_third_party_logs_off_stdout([_PROBE])

    streams = [getattr(h, "stream", None) for h in probe.handlers]
    assert sys.stdout not in streams
    assert streams == [sys.stderr]


def test_unrelated_handlers_are_left_alone(probe):
    """只掰 stdout 的；别人挂的文件 / 内存 handler 不动，也不再多加一个。"""
    memory_handler = logging.StreamHandler(io.StringIO())
    probe.addHandler(memory_handler)

    _load_guard()._keep_third_party_logs_off_stdout([_PROBE])

    assert probe.handlers == [memory_handler]


def test_stdout_handler_removed_even_when_others_remain(probe):
    """混合情形：stdout 的拿掉、别的留着、不补 stderr（已有出口）。"""
    memory_handler = logging.StreamHandler(io.StringIO())
    probe.addHandler(logging.StreamHandler(sys.stdout))
    probe.addHandler(memory_handler)

    _load_guard()._keep_third_party_logs_off_stdout([_PROBE])

    assert probe.handlers == [memory_handler]
