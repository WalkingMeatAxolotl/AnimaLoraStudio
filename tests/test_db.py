"""任务队列 SQLite DAO 测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db


@pytest.fixture
def dbfile(tmp_path: Path) -> Path:
    p = tmp_path / "studio.db"
    db.init_db(p)
    return p


def test_create_and_get(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        tid = db.create_task(conn, name="t1", config_name="run1")
        task = db.get_task(conn, tid)
    assert task is not None
    assert task["name"] == "t1"
    assert task["config_name"] == "run1"
    assert task["status"] == "pending"
    assert task["priority"] == 0
    assert task["created_at"] > 0


def test_list_ordering(dbfile: Path) -> None:
    """priority desc, created_at asc."""
    with db.connection_for(dbfile) as conn:
        a = db.create_task(conn, name="a", config_name="x", priority=0)
        b = db.create_task(conn, name="b", config_name="x", priority=10)
        c = db.create_task(conn, name="c", config_name="x", priority=10)
        items = db.list_tasks(conn)
    # b、c 优先级高，b 早于 c；a 排最后
    assert [t["id"] for t in items] == [b, c, a]


def test_filter_by_status(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        a = db.create_task(conn, name="a", config_name="x")
        db.create_task(conn, name="b", config_name="x")
        db.update_task(conn, a, status="done")
        pending = db.list_tasks(conn, status="pending")
        done = db.list_tasks(conn, status="done")
    assert len(pending) == 1 and pending[0]["status"] == "pending"
    assert len(done) == 1 and done[0]["id"] == a


def test_next_pending(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        db.create_task(conn, name="low", config_name="x", priority=0)
        high = db.create_task(conn, name="high", config_name="x", priority=10)
        nxt = db.next_pending(conn)
    assert nxt is not None and nxt["id"] == high


def test_next_pending_none_when_empty(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        assert db.next_pending(conn) is None


def test_update_task(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        tid = db.create_task(conn, name="t", config_name="x")
        db.update_task(conn, tid, status="running", pid=1234)
        task = db.get_task(conn, tid)
    assert task["status"] == "running"
    assert task["pid"] == 1234


def test_delete_task(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        tid = db.create_task(conn, name="t", config_name="x")
        affected = db.delete_task(conn, tid)
        assert affected == 1
        assert db.get_task(conn, tid) is None


def test_reorder_only_pending(dbfile: Path) -> None:
    """完成的任务不应被 reorder 影响 priority。"""
    with db.connection_for(dbfile) as conn:
        a = db.create_task(conn, name="a", config_name="x")
        b = db.create_task(conn, name="b", config_name="x")
        c = db.create_task(conn, name="c", config_name="x")
        # c 已完成
        db.update_task(conn, c, status="done", priority=100)
        c_priority_before = db.get_task(conn, c)["priority"]
        # 把 a、b 重排，b 优先
        db.reorder(conn, [b, a])
        a_after = db.get_task(conn, a)
        b_after = db.get_task(conn, b)
        c_after = db.get_task(conn, c)
    assert b_after["priority"] > a_after["priority"]
    assert c_after["priority"] == c_priority_before  # 已完成的不变


def test_invalid_status_constants() -> None:
    assert "pending" in db.VALID_STATUSES
    assert "done" in db.TERMINAL_STATUSES
    assert "running" not in db.TERMINAL_STATUSES
