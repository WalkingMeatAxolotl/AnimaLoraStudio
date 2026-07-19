"""进程树终止的跨平台、无机器状态单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

from studio.supervisor import process


def test_windows_taskkill_success_does_not_use_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(
        process.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stderr=b"", stdout=b"",
        ),
    )
    monkeypatch.setattr(
        process, "_kill_process_tree_psutil", lambda pid: calls.append(pid),
    )

    process._kill_process_tree_windows(123)

    assert calls == []


def test_windows_taskkill_nonzero_uses_psutil_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(
        process.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr=b"ERROR: Access denied", stdout=b"",
        ),
    )
    monkeypatch.setattr(
        process, "_kill_process_tree_psutil", lambda pid: calls.append(pid),
    )

    process._kill_process_tree_windows(456)

    assert calls == [456]


def test_windows_taskkill_exception_uses_psutil_fallback(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        raise FileNotFoundError("taskkill missing")

    monkeypatch.setattr(process.subprocess, "run", fail)
    monkeypatch.setattr(
        process, "_kill_process_tree_psutil", lambda pid: calls.append(pid),
    )

    process._kill_process_tree_windows(789)

    assert calls == [789]
