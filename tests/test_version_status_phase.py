"""ADR-0007 §11.3-B: versions.status / phase 双字段 + v8 migration 测试。"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from studio import db, versions
from studio.migrations import current_version


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# schema 加列 + 默认值
# ---------------------------------------------------------------------------


def test_v8_adds_status_phase_columns(tmp_path: Path) -> None:
    """新库 init 后 versions 含 status / phase / last_failure_reason 三列。"""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(versions)")}
        assert {"status", "phase", "last_failure_reason"} <= cols


def test_v8_default_values_for_new_versions(tmp_path: Path) -> None:
    """新建 version 不显式指定 status/phase → DEFAULT 'preparing' / 'curating'。"""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        c.execute(
            "INSERT INTO projects(slug, title, stage, created_at, updated_at) "
            "VALUES ('p', 'P', 'created', ?, ?)",
            (time.time(), time.time()),
        )
        pid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        c.execute(
            "INSERT INTO versions(project_id, label, stage, created_at) "
            "VALUES (?, 'v', 'curating', ?)",
            (pid, time.time()),
        )
        c.commit()

        row = c.execute(
            "SELECT status, phase, last_failure_reason FROM versions WHERE label='v'"
        ).fetchone()
        assert row["status"] == "preparing"
        assert row["phase"] == "curating"
        assert row["last_failure_reason"] is None


def test_v8_apply_all_idempotent(tmp_path: Path) -> None:
    """重复跑 init_db 不应重复添加列或丢数据。"""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    db.init_db(dbfile)
    with _open(dbfile) as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(versions)")}
        assert {"status", "phase", "last_failure_reason"} <= cols
        assert current_version(c) == 8


# ---------------------------------------------------------------------------
# v8 backfill: 静态映射
# ---------------------------------------------------------------------------


def test_v8_backfill_static_stages(tmp_path: Path) -> None:
    """老 5 stage 静态映射：curating/tagging/regularizing/ready → preparing+phase；done → completed+ready。"""
    from studio.migrations._v8_version_status_phase import _backfill

    dbfile = tmp_path / "test.db"
    db.init_db(dbfile)

    with _open(dbfile) as c:
        c.execute(
            "INSERT INTO projects(slug, title, stage, created_at, updated_at) "
            "VALUES ('p', 'P', 'created', ?, ?)",
            (time.time(), time.time()),
        )
        pid = c.execute("SELECT last_insert_rowid()").fetchone()[0]

        # 插入老 stage 5 种，模拟 v8 跑之前的状态
        for stage in ("curating", "tagging", "regularizing", "ready", "done"):
            c.execute(
                "INSERT INTO versions(project_id, label, stage, created_at) "
                "VALUES (?, ?, ?, ?)",
                (pid, stage, stage, time.time()),
            )
        c.commit()

        # 再跑一次 backfill 强制重算（默认 v8 跑过已经设了，这里验证函数本身）
        _backfill(c)

        rows = {
            r["label"]: (r["status"], r["phase"])
            for r in c.execute("SELECT label, status, phase FROM versions")
        }
        assert rows["curating"] == ("preparing", "curating")
        assert rows["tagging"] == ("preparing", "tagging")
        assert rows["regularizing"] == ("preparing", "regularizing")
        assert rows["ready"] == ("preparing", "ready")
        assert rows["done"] == ("completed", "ready")


# ---------------------------------------------------------------------------
# v8 backfill: training stage 按 latest task 推
# ---------------------------------------------------------------------------


def test_v8_backfill_training_by_latest_task(tmp_path: Path) -> None:
    """stage=training 时按 latest task.status 派生 version.status。"""
    from studio.migrations._v8_version_status_phase import _backfill

    dbfile = tmp_path / "test.db"
    db.init_db(dbfile)

    with _open(dbfile) as c:
        c.execute(
            "INSERT INTO projects(slug, title, stage, created_at, updated_at) "
            "VALUES ('p', 'P', 'created', ?, ?)",
            (time.time(), time.time()),
        )
        pid = c.execute("SELECT last_insert_rowid()").fetchone()[0]

        cases = [
            ("v_done",     "done",     "completed"),
            ("v_failed",   "failed",   "failed"),
            ("v_canceled", "canceled", "canceled"),
            ("v_running",  "running",  "training"),
            ("v_pending",  "pending",  "training"),
            ("v_paused",   "paused",   "training"),
        ]
        for label, task_status, _ in cases:
            c.execute(
                "INSERT INTO versions(project_id, label, stage, created_at) "
                "VALUES (?, ?, 'training', ?)",
                (pid, label, time.time()),
            )
            vid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
            c.execute(
                "INSERT INTO tasks(name, config_name, status, project_id, version_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (label, "cfg", task_status, pid, vid, time.time()),
            )
        c.commit()
        _backfill(c)

        rows = {
            r["label"]: r["status"]
            for r in c.execute("SELECT label, status FROM versions")
        }
        for label, _, expected in cases:
            assert rows[label] == expected, f"{label} → expected {expected}, got {rows[label]}"


def test_v8_backfill_training_without_task_fallback(tmp_path: Path) -> None:
    """脏数据：stage=training 但 task 不存在 → fallback preparing+ready。"""
    from studio.migrations._v8_version_status_phase import _backfill

    dbfile = tmp_path / "test.db"
    db.init_db(dbfile)

    with _open(dbfile) as c:
        c.execute(
            "INSERT INTO projects(slug, title, stage, created_at, updated_at) "
            "VALUES ('p', 'P', 'created', ?, ?)",
            (time.time(), time.time()),
        )
        pid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        c.execute(
            "INSERT INTO versions(project_id, label, stage, created_at) "
            "VALUES (?, 'orphan', 'training', ?)",
            (pid, time.time()),
        )
        c.commit()
        _backfill(c)

        row = c.execute(
            "SELECT status, phase FROM versions WHERE label='orphan'"
        ).fetchone()
        assert row["status"] == "preparing"
        assert row["phase"] == "ready"


def test_v8_backfill_training_uses_latest_task(tmp_path: Path) -> None:
    """同 version 多 task 时，按 created_at 最新的 task.status 派生。"""
    from studio.migrations._v8_version_status_phase import _backfill

    dbfile = tmp_path / "test.db"
    db.init_db(dbfile)

    with _open(dbfile) as c:
        c.execute(
            "INSERT INTO projects(slug, title, stage, created_at, updated_at) "
            "VALUES ('p', 'P', 'created', ?, ?)",
            (time.time(), time.time()),
        )
        pid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        c.execute(
            "INSERT INTO versions(project_id, label, stage, created_at) "
            "VALUES (?, 'v', 'training', ?)",
            (pid, time.time()),
        )
        vid = c.execute("SELECT last_insert_rowid()").fetchone()[0]

        # 老 task failed，新 task done —— 应取新的（completed）
        now = time.time()
        c.execute(
            "INSERT INTO tasks(name, config_name, status, project_id, version_id, created_at) "
            "VALUES ('old', 'cfg', 'failed', ?, ?, ?)",
            (pid, vid, now - 100),
        )
        c.execute(
            "INSERT INTO tasks(name, config_name, status, project_id, version_id, created_at) "
            "VALUES ('new', 'cfg', 'done', ?, ?, ?)",
            (pid, vid, now),
        )
        c.commit()
        _backfill(c)

        row = c.execute("SELECT status FROM versions WHERE label='v'").fetchone()
        assert row["status"] == "completed"


# ---------------------------------------------------------------------------
# VersionStatus / VersionPhase enum + helper
# ---------------------------------------------------------------------------


def test_version_status_enum_values() -> None:
    assert versions.VersionStatus.PREPARING == "preparing"
    assert versions.VersionStatus.TRAINING == "training"
    assert versions.VersionStatus.COMPLETED == "completed"
    assert versions.VersionStatus.FAILED == "failed"
    assert versions.VersionStatus.CANCELED == "canceled"
    assert len(versions.VersionStatus.VALUES) == 5


def test_version_phase_order_and_skippable() -> None:
    assert versions.VersionPhase.ORDER == (
        "curating", "tagging", "editing", "regularizing", "ready",
    )
    assert len(versions.VersionPhase.VALUES) == 5
    assert versions.VersionPhase.SKIPPABLE == frozenset({"regularizing"})


def test_get_status_fallback_to_preparing() -> None:
    assert versions.get_status({}) == "preparing"
    assert versions.get_status({"status": None}) == "preparing"
    assert versions.get_status({"status": ""}) == "preparing"
    assert versions.get_status({"status": "training"}) == "training"


def test_get_phase_fallback_to_curating() -> None:
    assert versions.get_phase({}) == "curating"
    assert versions.get_phase({"phase": None}) == "curating"
    assert versions.get_phase({"phase": "editing"}) == "editing"
