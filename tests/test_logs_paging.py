"""`GET /api/logs/{task_id}` 分页读取（docs/design/logging-target-state.md §3.4）。

覆盖：tail 默认 / tail 跨块 / before 往前翻 / after 断线补拉 / EVENT 行剥离 /
末尾半行不返回 / 单行超块 / offset 与 SSE（LogTailer）同坐标系 / raw 下载 /
tail·before·after 互斥。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.api.routers import logs as logs_mod
from studio.infrastructure.log_tail import LogTailer


@pytest.fixture
def log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from studio.infrastructure import paths as _paths
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(logs_mod, "LOGS_DIR", tmp_path / "legacy")
    p = _paths.task_log_path(7)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_lines(p: Path, n: int, *, trailing_newline: bool = True, width: int = 0) -> list[bytes]:
    lines = [f"line-{i:05d}".encode() + (b"x" * width) for i in range(n)]
    data = b"\n".join(lines) + (b"\n" if trailing_newline else b"")
    p.write_bytes(data)
    return lines


def test_missing_returns_empty_page() -> None:
    page = logs_mod.read_task_log_page(999)
    assert page["lines"] == [] and page["size"] == 0 and page["has_more_before"] is False


def test_tail_default_and_offsets(log_file: Path) -> None:
    lines = _write_lines(log_file, 10)
    page = logs_mod.read_task_log_page(7, tail=3, limit=3)
    assert [l["text"] for l in page["lines"]] == ["line-00007", "line-00008", "line-00009"]
    # offset = 行起始字节；end_offset = 文件末尾（最后一行含换行）
    assert page["lines"][0]["offset"] == sum(len(x) + 1 for x in lines[:7])
    assert page["end_offset"] == log_file.stat().st_size == page["size"]
    assert page["has_more_before"] is True


def test_tail_more_than_file_returns_all(log_file: Path) -> None:
    _write_lines(log_file, 4)
    page = logs_mod.read_task_log_page(7, limit=500)
    assert len(page["lines"]) == 4
    assert page["start_offset"] == 0 and page["has_more_before"] is False


def test_tail_spans_multiple_read_chunks(log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logs_mod, "_READ_CHUNK", 64)  # 每块只装几行，逼出跨块拼接
    lines = _write_lines(log_file, 200)
    page = logs_mod.read_task_log_page(7, limit=50)
    assert [l["text"].encode() for l in page["lines"]] == lines[-50:]
    assert page["lines"][0]["offset"] == sum(len(x) + 1 for x in lines[:150])


def test_before_pages_backwards_without_overlap(log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logs_mod, "_READ_CHUNK", 64)
    lines = _write_lines(log_file, 30)
    p1 = logs_mod.read_task_log_page(7, limit=10)
    p2 = logs_mod.read_task_log_page(7, before=p1["start_offset"], limit=10)
    p3 = logs_mod.read_task_log_page(7, before=p2["start_offset"], limit=10)
    p4 = logs_mod.read_task_log_page(7, before=p3["start_offset"], limit=10)
    got = [l["text"].encode() for l in p3["lines"] + p2["lines"] + p1["lines"]]
    assert got == lines
    assert p3["has_more_before"] is False and p4["lines"] == []


def test_after_returns_new_lines_and_cursor(log_file: Path) -> None:
    lines = _write_lines(log_file, 5)
    page = logs_mod.read_task_log_page(7, limit=500)
    cursor = page["end_offset"]
    with open(log_file, "ab") as f:
        f.write(b"new-a\nnew-b\npartial")
    more = logs_mod.read_task_log_page(7, after=cursor, limit=500)
    assert [l["text"] for l in more["lines"]] == ["new-a", "new-b"]
    # 半行不返回，游标停在它前面
    assert more["end_offset"] == cursor + len(b"new-a\nnew-b\n")
    assert more["lines"][0]["offset"] == sum(len(x) + 1 for x in lines)
    # 半行补齐换行后再拉，只拿到它
    with open(log_file, "ab") as f:
        f.write(b"-done\n")
    rest = logs_mod.read_task_log_page(7, after=more["end_offset"], limit=500)
    assert [l["text"] for l in rest["lines"]] == ["partial-done"]


def test_after_limit_cuts_and_cursor_precise(log_file: Path) -> None:
    lines = _write_lines(log_file, 10)
    page = logs_mod.read_task_log_page(7, after=0, limit=3)
    assert [l["text"].encode() for l in page["lines"]] == lines[:3]
    assert page["end_offset"] == sum(len(x) + 1 for x in lines[:3])
    nxt = logs_mod.read_task_log_page(7, after=page["end_offset"], limit=3)
    assert [l["text"].encode() for l in nxt["lines"]] == lines[3:6]


def test_event_lines_stripped_in_all_modes(log_file: Path) -> None:
    log_file.write_bytes(
        b"a\n__EVENT__:progress:{\"step\":1}\nb\n__EVENT__:pause_state:{}\nc\n"
    )
    tail = logs_mod.read_task_log_page(7, limit=500)
    assert [l["text"] for l in tail["lines"]] == ["a", "b", "c"]
    after = logs_mod.read_task_log_page(7, after=0, limit=500)
    assert [l["text"] for l in after["lines"]] == ["a", "b", "c"]
    # 剥掉 EVENT 行后仍凑满 limit（不是先截 limit 再剥）
    two = logs_mod.read_task_log_page(7, limit=2)
    assert [l["text"] for l in two["lines"]] == ["b", "c"]


def test_trailing_partial_line_excluded_from_tail(log_file: Path) -> None:
    _write_lines(log_file, 3, trailing_newline=False)
    page = logs_mod.read_task_log_page(7, limit=500)
    assert [l["text"] for l in page["lines"]] == ["line-00000", "line-00001"]
    assert page["end_offset"] == len(b"line-00000\nline-00001\n")


def test_single_line_larger_than_chunk(log_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logs_mod, "_READ_CHUNK", 64)
    lines = _write_lines(log_file, 3, width=300)
    page = logs_mod.read_task_log_page(7, limit=2)
    assert [l["text"].encode() for l in page["lines"]] == lines[-2:]
    assert page["lines"][0]["offset"] == len(lines[0]) + 1


def test_offsets_match_log_tailer_cursor(log_file: Path) -> None:
    """SSE 的 end_offset 与 API 的 after 游标是同一坐标系：tailer 推到哪，after 就从哪续。"""
    _write_lines(log_file, 5)
    seen: list[tuple[str, int]] = []
    tailer = LogTailer(log_file, lambda line, off: seen.append((line, off)), poll_interval=0.01)
    tailer._read_chunk()
    assert [t for t, _ in seen] == [f"line-{i:05d}" for i in range(5)]
    last_end = seen[-1][1]
    assert last_end == log_file.stat().st_size
    with open(log_file, "ab") as f:
        f.write(b"tail-1\n")
    page = logs_mod.read_task_log_page(7, after=last_end, limit=500)
    assert [l["text"] for l in page["lines"]] == ["tail-1"]


def test_query_modes_are_mutually_exclusive(log_file: Path) -> None:
    from fastapi.testclient import TestClient
    from studio import server
    client = TestClient(server.app)
    _write_lines(log_file, 2)
    r = client.get("/api/logs/7?tail=1&after=0")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "log.query_conflict"


def test_raw_download_and_404(log_file: Path) -> None:
    from fastapi.testclient import TestClient
    from studio import server
    client = TestClient(server.app)
    _write_lines(log_file, 2)
    r = client.get("/api/logs/7/raw")
    assert r.status_code == 200
    assert r.content == log_file.read_bytes()
    assert "task-7.log" in r.headers.get("content-disposition", "")
    r404 = client.get("/api/logs/8/raw")
    assert r404.status_code == 404
    assert r404.json()["error"]["code"] == "log.not_found"
