"""LoKr/LoHa normal-dropout 告警刷屏收敛（utils.lycoris_adapter）。

lycoris 的 LokrModule/LohaModule 在 `dropout>0` 时**每个模块实例**都 print 一行
"[WARN]LoHa/LoKr haven't implemented normal dropout yet."，280 层就是 280 行。
注入期把 stdout 按行过滤：含 marker 的整行吞掉并计数，其余原样透传，退出时汇总成
一条 logger 记录。

覆盖：整行丢弃 + 计数、非 marker 行原样透传、跨 write 的半行、无换行尾巴的 flush、
上下文退出（含异常）后 sys.stdout 复原、属性透传。
"""
from __future__ import annotations

import io
import sys

import pytest

from utils.lycoris_adapter import (
    _LOKR_DROPOUT_MARKER,
    _LineFilteredStdout,
    _suppress_lokr_dropout_spam,
)

SPAM = f"[WARN]LoHa/LoKr {_LOKR_DROPOUT_MARKER}"


def test_drops_marker_lines_and_counts_them() -> None:
    sink = io.StringIO()
    flt = _LineFilteredStdout(sink, _LOKR_DROPOUT_MARKER)

    for _ in range(280):
        flt.write(SPAM + "\n")

    assert flt.dropped == 280
    assert sink.getvalue() == ""


def test_passes_other_lines_through_unchanged() -> None:
    sink = io.StringIO()
    flt = _LineFilteredStdout(sink, _LOKR_DROPOUT_MARKER)

    flt.write("training step 1\n")
    flt.write(SPAM + "\n")
    flt.write("training step 2\n")

    assert flt.dropped == 1
    assert sink.getvalue() == "training step 1\ntraining step 2\n"


def test_filters_across_write_boundaries() -> None:
    """print 不保证一行一次 write —— 半行也要按行判定，不能漏过 marker。"""
    sink = io.StringIO()
    flt = _LineFilteredStdout(sink, _LOKR_DROPOUT_MARKER)

    head, tail = SPAM[:10], SPAM[10:]
    flt.write(head)
    flt.write(tail + "\nkeep me\n")

    assert flt.dropped == 1
    assert sink.getvalue() == "keep me\n"


def test_write_returns_input_length() -> None:
    """IO 协议：write 返回写入的字符数，调用方（print）据此判断。"""
    flt = _LineFilteredStdout(io.StringIO(), _LOKR_DROPOUT_MARKER)
    assert flt.write(SPAM + "\n") == len(SPAM) + 1
    assert flt.write("plain\n") == 6


def test_flush_handles_trailing_partial_line() -> None:
    """最后一行没有换行时，flush 也要按 marker 判定，不能把它当普通输出漏出去。"""
    sink = io.StringIO()
    flt = _LineFilteredStdout(sink, _LOKR_DROPOUT_MARKER)
    flt.write(SPAM)  # 无换行
    flt.flush()

    assert flt.dropped == 1
    assert sink.getvalue() == ""

    sink2 = io.StringIO()
    flt2 = _LineFilteredStdout(sink2, _LOKR_DROPOUT_MARKER)
    flt2.write("half line")  # 无换行、非 marker
    flt2.flush()

    assert flt2.dropped == 0
    assert sink2.getvalue() == "half line"


def test_forwards_unknown_attributes_to_wrapped() -> None:
    """encoding / isatty 等由被包装对象提供 —— 训练进程里 stdout 是真 TextIO。"""
    flt = _LineFilteredStdout(sys.__stdout__, _LOKR_DROPOUT_MARKER)
    assert flt.encoding == sys.__stdout__.encoding
    assert flt.isatty() == sys.__stdout__.isatty()


def test_context_restores_stdout_and_reports_count(capsys) -> None:
    original = sys.stdout
    with _suppress_lokr_dropout_spam() as flt:
        assert sys.stdout is not original
        print(SPAM)
        print(SPAM)
        print("real output")
    assert sys.stdout is original
    assert flt.dropped == 2
    assert capsys.readouterr().out == "real output\n"


def test_context_restores_stdout_on_exception() -> None:
    """注入抛错时也必须把 stdout 换回去，否则后续输出全走过滤器。"""
    original = sys.stdout
    with pytest.raises(RuntimeError):
        with _suppress_lokr_dropout_spam():
            print(SPAM)
            raise RuntimeError("injection blew up")
    assert sys.stdout is original
