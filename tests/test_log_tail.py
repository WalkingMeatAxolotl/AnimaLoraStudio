"""PP2 — LogTailer 增量推送行 callback。"""
from __future__ import annotations

import time
from pathlib import Path

from studio.log_tail import LogTailer


def _wait_lines(received: list[str], n: int, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while len(received) < n and time.time() < deadline:
        time.sleep(0.05)


def test_tailer_picks_up_appended_lines(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.touch()
    received: list[str] = []
    tailer = LogTailer(log, received.append, poll_interval=0.05)
    tailer.start()
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write("line one\n")
            f.write("line two\n")
        _wait_lines(received, 2)
        with open(log, "a", encoding="utf-8") as f:
            f.write("line three\n")
        _wait_lines(received, 3)
    finally:
        tailer.stop()
    assert received[:3] == ["line one", "line two", "line three"]


def test_tailer_handles_missing_file(tmp_path: Path) -> None:
    """文件还没出现时不应抛错；出现后正常 tail。"""
    log = tmp_path / "later.log"
    received: list[str] = []
    tailer = LogTailer(log, received.append, poll_interval=0.05)
    tailer.start()
    try:
        time.sleep(0.1)  # 文件不存在的轮询周期
        log.write_text("hello\n", encoding="utf-8")
        _wait_lines(received, 1)
    finally:
        tailer.stop()
    assert received == ["hello"]


def test_tailer_flushes_partial_line_on_stop(tmp_path: Path) -> None:
    """没有换行的尾部内容也应在 stop 时被 flush。"""
    log = tmp_path / "p.log"
    log.write_text("abc", encoding="utf-8")
    received: list[str] = []
    tailer = LogTailer(log, received.append, poll_interval=0.05)
    tailer.start()
    time.sleep(0.1)
    tailer.stop()
    assert received == ["abc"]
