"""PR-6 — model_downloader._on_log per-line print。

验证 download 后台 thread 里调 on_log 时，日志既进 UI ring buffer，也回显到
backend stdout（让 studio_*.log / 终端能完整 grep）。
"""
from __future__ import annotations

import threading
import time

import pytest

from studio.services import model_downloader


@pytest.fixture
def reset_downloads():
    """每个测试用例独立，避免 _DOWNLOADS 全局状态污染。"""
    with model_downloader._LOCK:
        before = dict(model_downloader._DOWNLOADS)
        model_downloader._DOWNLOADS.clear()
    yield
    with model_downloader._LOCK:
        model_downloader._DOWNLOADS.clear()
        model_downloader._DOWNLOADS.update(before)


def _wait_done(key: str, timeout: float = 2.0) -> None:
    """轮询等任务结束（避免依赖 bus / 线程加入）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with model_downloader._LOCK:
            ds = model_downloader._DOWNLOADS.get(key)
        if ds and ds.status in ("done", "failed"):
            return
        time.sleep(0.01)
    raise AssertionError(f"download '{key}' didn't complete in {timeout}s")


def test_on_log_writes_to_ring_buffer_and_stdout(
    reset_downloads, capfd: pytest.CaptureFixture
) -> None:
    """on_log 同时写：(1) ring buffer ds.log，(2) stdout（print(line, flush=True)）。"""
    lines_to_emit = ["downloading file 1", "downloading file 2", "✓ done"]

    def fake_fn(on_log):
        for line in lines_to_emit:
            on_log(line)
        return True

    model_downloader.start_download_async("test-key", fake_fn)
    _wait_done("test-key")

    # ring buffer 完整保留
    with model_downloader._LOCK:
        ds = model_downloader._DOWNLOADS["test-key"]
        assert ds.status == "done"
        assert ds.log == lines_to_emit

    # stdout 也都拿到（用 capfd 抓 fd 级 stdout，覆盖跨线程 print）
    out = capfd.readouterr().out
    for line in lines_to_emit:
        assert line in out


def test_on_log_does_not_hold_lock_during_print(
    reset_downloads,
) -> None:
    """print 在锁外：on_log 调用本身不应让 _LOCK 在 I/O 期间被占着。

    检测方式：让 print 阻塞（替成 sleep 兼带计时），同时另一线程尝试拿锁；
    若锁外执行，并发 acquire 应当能立刻成功。
    """
    import builtins

    print_started = threading.Event()
    can_finish_print = threading.Event()

    real_print = builtins.print

    def slow_print(*args, **kwargs):
        print_started.set()
        can_finish_print.wait(timeout=2.0)
        return real_print(*args, **kwargs)

    def fake_fn(on_log):
        builtins.print = slow_print
        try:
            on_log("first")
        finally:
            builtins.print = real_print
        return True

    model_downloader.start_download_async("test-lock", fake_fn)

    assert print_started.wait(timeout=2.0), "fake_fn 没进 print"

    # 此时 _on_log 应已离开 with _LOCK 块（先写 ring buffer 再 print），
    # 主线程能在 100ms 内拿到锁
    acquired = model_downloader._LOCK.acquire(timeout=0.1)
    assert acquired, "_on_log 在 print 期间持锁，违反 PR-6 设计"
    model_downloader._LOCK.release()

    can_finish_print.set()
    _wait_done("test-lock")
