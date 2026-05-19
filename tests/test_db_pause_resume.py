"""db.py + migration v6 (ADR 0006 PR-2) — paused 状态、queue_held kv。"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db


@pytest.fixture
def dbfile(tmp_path: Path) -> Path:
    p = tmp_path / "studio.db"
    db.init_db(p)
    return p


# ---- VALID_STATUSES ----------------------------------------------------------


def test_paused_in_valid_statuses() -> None:
    assert "paused" in db.VALID_STATUSES


def test_paused_not_terminal() -> None:
    """paused 可被 resume，不是终态。"""
    assert "paused" not in db.TERMINAL_STATUSES


# ---- migration v6 ------------------------------------------------------------


def test_migration_adds_pause_columns(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
    for col in ("paused_state_path", "paused_config_path", "paused_step", "paused_at"):
        assert col in cols, f"missing column {col}"


def test_migration_creates_queue_settings_table(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='queue_settings'"
        ).fetchone()
    assert row is not None


def test_migration_idempotent_on_existing_db(dbfile: Path) -> None:
    """重复 init_db 不应失败（migrations 已有 user_version 防重跑）。"""
    db.init_db(dbfile)  # second time
    with db.connection_for(dbfile) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "paused_state_path" in cols


def test_pause_columns_nullable_by_default(dbfile: Path) -> None:
    """新跑 task 默认 paused_* 全 NULL（没暂停过）。"""
    with db.connection_for(dbfile) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        task = db.get_task(conn, tid)
    assert task is not None
    for col in ("paused_state_path", "paused_config_path", "paused_step", "paused_at"):
        assert task[col] is None


# ---- get_queue_held / set_queue_held ----------------------------------------


def test_get_queue_held_default_false(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        assert db.get_queue_held(conn) is False


def test_set_queue_held_true_then_read(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        db.set_queue_held(conn, True)
        assert db.get_queue_held(conn) is True


def test_set_queue_held_flip(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        db.set_queue_held(conn, True)
        db.set_queue_held(conn, False)
        assert db.get_queue_held(conn) is False


def test_queue_held_persists_across_connections(dbfile: Path) -> None:
    """ADR §3.2: 跨 server 重启保留。"""
    with db.connection_for(dbfile) as conn:
        db.set_queue_held(conn, True)
    # 重开 connection 模拟 process 重启
    with db.connection_for(dbfile) as conn2:
        assert db.get_queue_held(conn2) is True


def test_queue_held_upsert_no_duplicate_rows(dbfile: Path) -> None:
    """ON CONFLICT 应该 update，不该再插一条。"""
    with db.connection_for(dbfile) as conn:
        for _ in range(5):
            db.set_queue_held(conn, True)
        row_count = conn.execute(
            "SELECT COUNT(*) FROM queue_settings WHERE key = 'queue.held'"
        ).fetchone()[0]
    assert row_count == 1


# ---- update_task 可写 paused 字段 -------------------------------------------


def test_update_task_can_set_paused_fields(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            status="paused",
            paused_state_path="/x/pause_step_100.pt",
            paused_config_path="/x/pause_step_100.config.json",
            paused_step=100,
            paused_at=1234567890.5,
        )
        task = db.get_task(conn, tid)
    assert task is not None
    assert task["status"] == "paused"
    assert task["paused_state_path"] == "/x/pause_step_100.pt"
    assert task["paused_step"] == 100
    assert task["paused_at"] == pytest.approx(1234567890.5)


def test_paused_task_not_in_pending_or_running_lists(dbfile: Path) -> None:
    """paused task 不该被 list_tasks(status='pending') 拉到，dispatcher 不会派它。"""
    with db.connection_for(dbfile) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(conn, tid, status="paused")
        pending = db.list_tasks(conn, status="pending")
        running = db.list_tasks(conn, status="running")
        paused = db.list_tasks(conn, status="paused")
    assert tid not in [t["id"] for t in pending]
    assert tid not in [t["id"] for t in running]
    assert tid in [t["id"] for t in paused]


def test_next_pending_skips_paused(dbfile: Path) -> None:
    with db.connection_for(dbfile) as conn:
        paused_id = db.create_task(conn, name="paused", config_name="c", priority=100)
        db.update_task(conn, paused_id, status="paused")
        pending_id = db.create_task(conn, name="pending", config_name="c", priority=1)
        nxt = db.next_pending(conn)
    assert nxt is not None and nxt["id"] == pending_id
