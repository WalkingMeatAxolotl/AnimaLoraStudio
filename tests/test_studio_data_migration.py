"""studio_data 自定义位置：指针解析 + 扫描 + 迁移复制线程。

全部用 tmp_path，不碰真 studio_data（测试卫生：不依赖机器状态）。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from studio.infrastructure.paths import resolve_studio_data, DEFAULT_STUDIO_DATA
from studio.services import studio_data as svc


# ---------------------------------------------------------------------------
# resolve_studio_data —— 指针文件解析
# ---------------------------------------------------------------------------

def test_resolve_no_pointer_returns_default(tmp_path: Path) -> None:
    assert resolve_studio_data(tmp_path / "absent.json") == DEFAULT_STUDIO_DATA


def test_resolve_valid_pointer(tmp_path: Path) -> None:
    target = tmp_path / "custom_data"
    target.mkdir()
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"path": str(target)}), encoding="utf-8")
    assert resolve_studio_data(ptr) == target


def test_resolve_broken_json_falls_back(tmp_path: Path) -> None:
    ptr = tmp_path / "ptr.json"
    ptr.write_text("{not json", encoding="utf-8")
    assert resolve_studio_data(ptr) == DEFAULT_STUDIO_DATA


def test_resolve_missing_target_dir_falls_back(tmp_path: Path) -> None:
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"path": str(tmp_path / "gone")}), encoding="utf-8")
    assert resolve_studio_data(ptr) == DEFAULT_STUDIO_DATA


def test_resolve_relative_path_falls_back(tmp_path: Path) -> None:
    ptr = tmp_path / "ptr.json"
    ptr.write_text(json.dumps({"path": "relative/dir"}), encoding="utf-8")
    assert resolve_studio_data(ptr) == DEFAULT_STUDIO_DATA


# ---------------------------------------------------------------------------
# scan_studio_data
# ---------------------------------------------------------------------------

def _make_tree(root: Path) -> None:
    (root / "tasks" / "1").mkdir(parents=True)
    (root / "tasks" / "1" / "run.log").write_bytes(b"x" * 10)
    (root / "presets").mkdir()
    (root / "presets" / "a.yaml").write_bytes(b"y" * 20)
    (root / "secrets.json").write_bytes(b"{}")
    # sqlite 伴生文件不计入 / 不复制
    (root / "studio.db-wal").write_bytes(b"w" * 100)
    (root / "studio.db-shm").write_bytes(b"s" * 100)


def test_scan_counts_files_and_bytes(tmp_path: Path) -> None:
    root = tmp_path / "sd"
    root.mkdir()
    _make_tree(root)
    r = svc.scan_studio_data(root)
    assert r["total_files"] == 3
    assert r["total_bytes"] == 32
    names = {e["name"] for e in r["entries"]}
    assert names == {"tasks", "presets", "secrets.json"}


def test_scan_missing_dir_returns_zero(tmp_path: Path) -> None:
    r = svc.scan_studio_data(tmp_path / "absent")
    assert r == {"total_files": 0, "total_bytes": 0, "entries": []}


# ---------------------------------------------------------------------------
# validate_target
# ---------------------------------------------------------------------------

def test_validate_rejects_relative(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="绝对路径"):
        svc.validate_target(Path("relative"), source=tmp_path)


def test_validate_rejects_same_path(tmp_path: Path) -> None:
    # 当前位置就是 tmp_path/studio_data，再选 tmp_path 当目标 → 落地目录撞上自己
    src = tmp_path / "studio_data"
    src.mkdir()
    with pytest.raises(ValueError, match="相同"):
        svc.validate_target(tmp_path, source=src)


def test_validate_rejects_nested_both_ways(tmp_path: Path) -> None:
    # 落地目录（target/studio_data）嵌进当前位置
    src = tmp_path / "sd"
    src.mkdir()
    with pytest.raises(ValueError, match="嵌套"):
        svc.validate_target(src / "inner", source=src)
    # 当前位置嵌进落地目录
    nested_src = tmp_path / "studio_data" / "inner"
    nested_src.mkdir(parents=True)
    with pytest.raises(ValueError, match="嵌套"):
        svc.validate_target(tmp_path, source=nested_src)


def test_validate_accepts_nonempty_target(tmp_path: Path) -> None:
    # 目标本身非空没关系 —— 数据落 target/studio_data/ 子目录
    src = tmp_path / "sd"
    src.mkdir()
    tgt = tmp_path / "tgt"
    tgt.mkdir()
    (tgt / "junk.txt").write_text("x")
    assert svc.validate_target(tgt, source=src) == tgt / "studio_data"


def test_validate_rejects_nonempty_destination(tmp_path: Path) -> None:
    src = tmp_path / "sd"
    src.mkdir()
    tgt = tmp_path / "tgt"
    (tgt / "studio_data").mkdir(parents=True)
    (tgt / "studio_data" / "junk.txt").write_text("x")
    with pytest.raises(ValueError, match="非空"):
        svc.validate_target(tgt, source=src)


def test_validate_rejects_file_target(tmp_path: Path) -> None:
    src = tmp_path / "sd"
    src.mkdir()
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(ValueError, match="不是目录"):
        svc.validate_target(f, source=src)


def test_validate_accepts_empty_or_absent(tmp_path: Path) -> None:
    src = tmp_path / "sd"
    src.mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert svc.validate_target(empty, source=src) == empty / "studio_data"
    assert svc.validate_target(tmp_path / "absent", source=src) == tmp_path / "absent" / "studio_data"


# ---------------------------------------------------------------------------
# 迁移复制线程（_run_migration 直接同步调，避免依赖线程时序）
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_status():
    """模块级状态单例在测试间互相污染，逐测试重置。"""
    yield
    svc._status = svc.MigrationStatus()


def test_migration_copies_tree_and_writes_pointer(tmp_path: Path) -> None:
    src = tmp_path / "sd"
    src.mkdir()
    _make_tree(src)
    # 真 sqlite：迁移走 backup API，产物必须能打开且数据一致
    with sqlite3.connect(str(src / "studio.db")) as conn:
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES ('hello')")
    dst = tmp_path / "moved"
    ptr = tmp_path / "ptr.json"
    events: list[dict] = []
    svc._run_migration(src, dst, events.append, ptr)

    assert (dst / "tasks" / "1" / "run.log").read_bytes() == b"x" * 10
    assert (dst / "presets" / "a.yaml").exists()
    assert (dst / "secrets.json").exists()
    assert not (dst / "studio.db-wal").exists()
    assert not (dst / "studio.db-shm").exists()
    with sqlite3.connect(str(dst / "studio.db")) as conn:
        assert conn.execute("SELECT v FROM t").fetchone() == ("hello",)

    assert json.loads(ptr.read_text("utf-8")) == {"path": str(dst)}
    assert resolve_studio_data(ptr) == dst

    done = [e for e in events if e["type"] == "studio_data_migrate_done"]
    assert len(done) == 1 and done[0]["ok"] is True
    assert svc.migration_status()["state"] == "done"


def test_migration_failure_cleans_target_and_keeps_pointer_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "sd"
    src.mkdir()
    _make_tree(src)
    dst = tmp_path / "moved"
    ptr = tmp_path / "ptr.json"

    def boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(svc.shutil, "copy2", boom)

    events: list[dict] = []
    svc._run_migration(src, dst, events.append, ptr)

    assert not dst.exists()          # 半截目标已清掉
    assert not ptr.exists()          # 指针没写 —— 重启后仍用旧位置
    done = [e for e in events if e["type"] == "studio_data_migrate_done"]
    assert len(done) == 1 and done[0]["ok"] is False
    status = svc.migration_status()
    assert status["state"] == "error"
    assert "disk full" in status["error"]


def test_start_migration_rejects_concurrent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "sd"
    src.mkdir()
    (src / "f.txt").write_text("x")
    release = threading.Event()
    started = threading.Event()

    def slow_run(*a, **kw):
        started.set()
        release.wait(5)
        svc._set_status(state="done")
    monkeypatch.setattr(svc, "_run_migration", slow_run)

    svc.start_migration(tmp_path / "t1", source=src, publish=lambda e: None,
                        pointer_file=tmp_path / "ptr.json")
    assert started.wait(5)
    with pytest.raises(RuntimeError, match="正在进行"):
        svc.start_migration(tmp_path / "t2", source=src, publish=lambda e: None,
                            pointer_file=tmp_path / "ptr.json")
    release.set()


def test_start_migration_lands_in_studio_data_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目标传父目录（可非空），复制线程拿到的落地目录是 target/studio_data/。"""
    src = tmp_path / "sd"
    src.mkdir()
    (src / "f.txt").write_text("x")
    got: dict[str, Path] = {}
    finished = threading.Event()

    def fake_run(s: Path, d: Path, pub, ptr) -> None:
        got["dst"] = d
        svc._set_status(state="done")
        finished.set()
    monkeypatch.setattr(svc, "_run_migration", fake_run)

    tgt = tmp_path / "tgt"
    tgt.mkdir()
    (tgt / "existing.bin").write_text("y")  # 非空目标也接受
    svc.start_migration(tgt, source=src, publish=lambda e: None,
                        pointer_file=tmp_path / "ptr.json")
    assert finished.wait(5)
    assert got["dst"] == tgt / "studio_data"
    assert svc.migration_status()["target"] == str(tgt / "studio_data")
