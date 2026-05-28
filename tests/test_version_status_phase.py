"""ADR-0007 §11.3-B: versions.status / phase 双字段 + enum / accessor 测试。

注：原 v8 backfill 测试已在 PR-5 v9 destructive 后删除 —— 那些测试需要 stage
列存在以塞入老数据，v9 已物理删除 projects.stage / versions.stage。
v8 backfill 函数的逻辑覆盖移到了 PR-2 review + ADR §11.3-B 文档；这里仅留
"v8 加列存在 + 默认值 + apply_all 幂等 + enum / accessor" 共 5 类测试。
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from studio import db
from studio.services.projects import versions
from studio.infrastructure.migrations import MIGRATIONS, current_version


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


def test_v9_drops_stage_column(tmp_path: Path) -> None:
    """ADR-0007 PR-5 v9 destructive：projects.stage / versions.stage 已不存在。"""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        v_cols = {r["name"] for r in c.execute("PRAGMA table_info(versions)")}
        p_cols = {r["name"] for r in c.execute("PRAGMA table_info(projects)")}
    assert "stage" not in v_cols
    assert "stage" not in p_cols


def test_default_values_for_new_versions(tmp_path: Path) -> None:
    """新建 version 不显式指定 status/phase → DEFAULT 'preparing' / 'curating'。"""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        c.execute(
            "INSERT INTO projects(slug, title, created_at, updated_at) "
            "VALUES ('p', 'P', ?, ?)",
            (time.time(), time.time()),
        )
        pid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        c.execute(
            "INSERT INTO versions(project_id, label, created_at) "
            "VALUES (?, 'v', ?)",
            (pid, time.time()),
        )
        c.commit()

        row = c.execute(
            "SELECT status, phase, last_failure_reason FROM versions WHERE label='v'"
        ).fetchone()
        assert row["status"] == "preparing"
        assert row["phase"] == "curating"
        assert row["last_failure_reason"] is None


def test_apply_all_idempotent(tmp_path: Path) -> None:
    """重复跑 init_db 不应破坏 schema。"""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    db.init_db(dbfile)
    with _open(dbfile) as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(versions)")}
        assert {"status", "phase", "last_failure_reason"} <= cols
        assert current_version(c) == len(MIGRATIONS) + 1


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
