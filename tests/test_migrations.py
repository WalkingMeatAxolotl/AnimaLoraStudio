"""PP1 — schema migration framework: v1 → v2 升级保留旧 tasks 数据。"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from studio import db
from studio.migrations import MIGRATIONS, apply_all, current_version


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def test_fresh_db_lands_at_latest_version(tmp_path: Path) -> None:
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        v = current_version(c)
    assert v == len(MIGRATIONS) + 1


def test_v2_creates_tables_and_extends_tasks(tmp_path: Path) -> None:
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        names = {
            r["name"]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"tasks", "projects", "versions", "project_jobs"} <= names
        cols = {r["name"] for r in c.execute("PRAGMA table_info(tasks)")}
        assert {"project_id", "version_id"} <= cols


def test_v1_db_upgrades_in_place_preserving_tasks(tmp_path: Path) -> None:
    """模拟 PP0 之前留下来的 v1 库（只有 tasks 表）：执行 init_db 应升到 v2 且数据不丢。"""
    dbfile = tmp_path / "legacy.db"
    legacy = sqlite3.connect(str(dbfile))
    legacy.executescript(
        """
        CREATE TABLE tasks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            config_name  TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            priority     INTEGER NOT NULL DEFAULT 0,
            created_at   REAL NOT NULL,
            started_at   REAL,
            finished_at  REAL,
            pid          INTEGER,
            exit_code    INTEGER,
            output_dir   TEXT,
            error_msg    TEXT
        );
        """
    )
    legacy.execute(
        "INSERT INTO tasks(name, config_name, created_at) VALUES (?, ?, ?)",
        ("legacy_task", "legacy_cfg", time.time()),
    )
    legacy.commit()
    legacy.close()

    db.init_db(dbfile)

    with _open(dbfile) as c:
        rows = list(c.execute("SELECT * FROM tasks"))
        assert len(rows) == 1
        assert rows[0]["name"] == "legacy_task"
        # 新增列默认 NULL，能直接查出来
        assert rows[0]["project_id"] is None
        assert rows[0]["version_id"] is None
        assert current_version(c) == len(MIGRATIONS) + 1


def test_apply_all_is_idempotent(tmp_path: Path) -> None:
    dbfile = tmp_path / "x.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        first = apply_all(c)
        again = apply_all(c)
    assert first == again == len(MIGRATIONS) + 1
