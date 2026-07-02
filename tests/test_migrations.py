"""PP1 — schema migration framework: v1 → v2 升级保留旧 tasks 数据。"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from studio import db
from studio.infrastructure.migrations import MIGRATIONS, apply_all, current_version


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


def test_v5_adds_task_type_column(tmp_path: Path) -> None:
    """v5: tasks 加 task_type 列，旧 task 自动 fallback 到 'train'。"""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(tasks)")}
        assert "task_type" in cols
        # 新建一个 task，default 应该是 'train'
        c.execute(
            "INSERT INTO tasks(name, config_name, status, priority, created_at) "
            "VALUES (?, ?, 'pending', 0, ?)",
            ("legacy_task", "cfg", time.time()),
        )
        c.commit()
        row = c.execute("SELECT task_type FROM tasks").fetchone()
        assert row["task_type"] == "train"


def test_v14_adds_generate_meta_columns(tmp_path: Path) -> None:
    """v14: tasks 加 generate_params / generate_cover（0.17 P-I forward-write），全 nullable。"""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(tasks)")}
        assert "generate_params" in cols
        assert "generate_cover" in cols
        # 老 task（不写这两列）保持 NULL
        c.execute(
            "INSERT INTO tasks(name, config_name, status, priority, created_at) "
            "VALUES (?, ?, 'pending', 0, ?)",
            ("legacy", "cfg", time.time()),
        )
        c.commit()
        row = c.execute("SELECT generate_params, generate_cover FROM tasks").fetchone()
        assert row["generate_params"] is None
        assert row["generate_cover"] is None


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


# ---------------------------------------------------------------------------
# v11 — ADR 0010 加 preprocessing phase + 回填 curating+train 非空
# ---------------------------------------------------------------------------


def _setup_v11_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """init db + monkeypatch PROJECTS_DIR → 让 _v11 能找到 train/ 物理图。"""
    from studio.services.projects import projects
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    return dbfile


def test_v11_advances_curating_with_train_image_to_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """curating + train/{folder}/{image} 存在 → 推进到 preprocessing。"""
    from studio.infrastructure.migrations._v11_preprocessing_phase import migrate as v11
    from studio.services.projects import projects, versions
    dbfile = _setup_v11_env(tmp_path, monkeypatch)

    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    sub = projects.project_dir(p["id"], p["slug"]) / "versions" / "v1" / "train" / "1_data"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "X.png").write_bytes(b"img")
    with db.connection_for(dbfile) as conn:
        conn.execute("UPDATE versions SET phase = 'curating' WHERE id = ?", (v["id"],))
        conn.commit()
        v11(conn)
        new_phase = conn.execute(
            "SELECT phase FROM versions WHERE id = ?", (v["id"],)
        ).fetchone()[0]
    assert new_phase == "preprocessing"


def test_v11_keeps_curating_when_train_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """curating + train 空 → 保持 curating。"""
    from studio.infrastructure.migrations._v11_preprocessing_phase import migrate as v11
    from studio.services.projects import projects, versions
    dbfile = _setup_v11_env(tmp_path, monkeypatch)

    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
        conn.execute("UPDATE versions SET phase = 'curating' WHERE id = ?", (v["id"],))
        conn.commit()
        v11(conn)
        new_phase = conn.execute(
            "SELECT phase FROM versions WHERE id = ?", (v["id"],)
        ).fetchone()[0]
    assert new_phase == "curating"


def test_v11_ignores_other_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tagging / editing / regularizing / ready 不动，无论 train 状态。"""
    from studio.infrastructure.migrations._v11_preprocessing_phase import migrate as v11
    from studio.services.projects import projects, versions
    dbfile = _setup_v11_env(tmp_path, monkeypatch)

    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        for label, phase in [
            ("v1", "tagging"), ("v2", "editing"),
            ("v3", "regularizing"), ("v4", "ready"),
        ]:
            v = versions.create_version(conn, project_id=p["id"], label=label)
            conn.execute(
                "UPDATE versions SET phase = ? WHERE id = ?", (phase, v["id"])
            )
        conn.commit()
        # train 有图也不该被推
        for label in ["v1", "v2", "v3", "v4"]:
            sub = projects.project_dir(p["id"], p["slug"]) / "versions" / label / "train" / "1_data"
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "X.png").write_bytes(b"x")
        v11(conn)
        phases = dict(conn.execute(
            "SELECT label, phase FROM versions WHERE project_id = ?", (p["id"],)
        ).fetchall())
    assert phases == {
        "v1": "tagging", "v2": "editing",
        "v3": "regularizing", "v4": "ready",
    }


def test_v11_ignores_train_root_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train/ 根目录直接放的图忽略 → phase 保持 curating。"""
    from studio.infrastructure.migrations._v11_preprocessing_phase import migrate as v11
    from studio.services.projects import projects, versions
    dbfile = _setup_v11_env(tmp_path, monkeypatch)

    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
        conn.execute("UPDATE versions SET phase = 'curating' WHERE id = ?", (v["id"],))
        conn.commit()
    train_root = projects.project_dir(p["id"], p["slug"]) / "versions" / "v1" / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    (train_root / "stray.png").write_bytes(b"x")

    with db.connection_for(dbfile) as conn:
        v11(conn)
        new_phase = conn.execute(
            "SELECT phase FROM versions WHERE id = ?", (v["id"],)
        ).fetchone()[0]
    assert new_phase == "curating"


def test_v11_idempotent_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.infrastructure.migrations._v11_preprocessing_phase import migrate as v11
    from studio.services.projects import projects, versions
    dbfile = _setup_v11_env(tmp_path, monkeypatch)

    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    sub = projects.project_dir(p["id"], p["slug"]) / "versions" / "v1" / "train" / "1_data"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "X.png").write_bytes(b"x")
    with db.connection_for(dbfile) as conn:
        conn.execute("UPDATE versions SET phase = 'curating' WHERE id = ?", (v["id"],))
        conn.commit()
        v11(conn)
        v11(conn)
        new_phase = conn.execute(
            "SELECT phase FROM versions WHERE id = ?", (v["id"],)
        ).fetchone()[0]
    assert new_phase == "preprocessing"
