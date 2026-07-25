"""旧模型 eval 作业行的一次性清理（issue #465 存量收尾）。

**代际判据**：清的是旧模型的 eval 子作业行（一次评估散成几百条），留的是新模型的
`eval_session` 行（一次评估一条 = 正常历史记录）。判据刻意不看「run 文件还在不在」——
那个口径会把正常历史当垃圾清掉。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studio import db, secrets
from studio.infrastructure import paths as infra_paths
from studio.services import eval_cleanup
from studio.services.projects import jobs as project_jobs, projects, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(infra_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


def _make_task(conn, *, kind: str, status: str = "done", params: dict[str, Any] | None = None) -> int:
    """建一条作业行 + 它的 tasks/<id>/run.log（模拟 supervisor 派发过）。"""
    tid = db.create_task(
        conn, name=kind, config_name=kind, task_type=kind, params=params or {},
    )
    if status != "pending":
        db.update_task(conn, tid, status=status)
    log = infra_paths.task_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("fake eval log\n" * 20, encoding="utf-8")
    return tid


# ---------------------------------------------------------------------------
# 代际判据
# ---------------------------------------------------------------------------

def test_scan_picks_up_all_legacy_eval_kinds(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        legacy = [
            _make_task(conn, kind=kind)
            for kind in ("eval_samples", "eval_clip", "eval_dino", "eval_tag", "eval_ccip")
        ]
        scan = eval_cleanup.scan_legacy_jobs(conn)

    assert scan["count"] == 5
    assert sorted(j["id"] for j in scan["jobs"]) == sorted(legacy)
    assert scan["dirs"] == 5
    assert scan["bytes"] > 0


def test_scan_never_touches_eval_session_rows(isolated) -> None:
    """新模型的 Session 行是正常历史记录 —— 无论多老都不清。"""
    with db.connection_for(isolated["db"]) as conn:
        session_task = _make_task(conn, kind="eval_session", params={"session_id": 1})
        legacy = _make_task(conn, kind="eval_clip")
        scan = eval_cleanup.scan_legacy_jobs(conn)

    assert [j["id"] for j in scan["jobs"]] == [legacy]
    assert session_task not in [j["id"] for j in scan["jobs"]]


def test_scan_ignores_non_eval_tasks(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        _make_task(conn, kind="train")
        _make_task(conn, kind="tag")
        _make_task(conn, kind="download")
        scan = eval_cleanup.scan_legacy_jobs(conn)

    assert scan["count"] == 0


def test_scan_skips_non_terminal_rows(isolated) -> None:
    """pending / running 不碰。_v19 已把升级瞬间的残留收成 canceled；真还有非终态的，
    留着让用户看见比默默删掉好。"""
    with db.connection_for(isolated["db"]) as conn:
        _make_task(conn, kind="eval_samples", status="pending")
        _make_task(conn, kind="eval_clip", status="running")
        done = _make_task(conn, kind="eval_dino", status="done")
        canceled = _make_task(conn, kind="eval_tag", status="canceled")
        scan = eval_cleanup.scan_legacy_jobs(conn)

    assert sorted(j["id"] for j in scan["jobs"]) == sorted([done, canceled])


def test_scan_does_not_care_whether_run_files_exist(isolated) -> None:
    """代际判据不看 run 是否存在 —— 旧模型的行一律清，不管它当年的结果还在不在。"""
    with db.connection_for(isolated["db"]) as conn:
        a = _make_task(conn, kind="eval_samples", params={"run_id": "run-alive"})
        b = _make_task(conn, kind="eval_clip", params={})  # 连 run_id 都没有
        scan = eval_cleanup.scan_legacy_jobs(conn)

    assert sorted(j["id"] for j in scan["jobs"]) == sorted([a, b])


def test_scan_without_size_skips_expensive_stat(isolated) -> None:
    """启动期自动清理不报数给用户 → 跳过递归 stat（几千个目录会明显拖慢）。"""
    with db.connection_for(isolated["db"]) as conn:
        _make_task(conn, kind="eval_samples")
        scan = eval_cleanup.scan_legacy_jobs(conn, with_size=False)

    assert scan["count"] == 1
    assert scan["bytes"] == 0
    assert scan["dirs"] == 1  # 目录存在性仍然报


# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------

def test_purge_removes_dirs_and_rows(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        legacy = _make_task(conn, kind="eval_samples")
        session_task = _make_task(conn, kind="eval_session", params={"session_id": 1})
        legacy_dir = infra_paths.task_dir(legacy)
        session_dir = infra_paths.task_dir(session_task)
        assert legacy_dir.exists() and session_dir.exists()

        result = eval_cleanup.purge_legacy_jobs(conn)

        assert result["removed_rows"] == 1
        assert result["removed_dirs"] == 1
        assert result["freed_bytes"] > 0
        assert not legacy_dir.exists()
        assert db.get_task(conn, legacy) is None
        # Session 的行和目录都不能动
        assert session_dir.exists()
        assert db.get_task(conn, session_task) is not None


def test_purge_intersects_requested_ids_with_fresh_scan(isolated) -> None:
    """传进来的 id 只做过滤 —— 拿 eval_session 的 id 来也删不掉。"""
    with db.connection_for(isolated["db"]) as conn:
        legacy = _make_task(conn, kind="eval_samples")
        other_legacy = _make_task(conn, kind="eval_dino")
        session_task = _make_task(conn, kind="eval_session", params={"session_id": 1})
        running = _make_task(conn, kind="eval_clip", status="running")

        result = eval_cleanup.purge_legacy_jobs(
            conn, ids=[legacy, session_task, running, 99999]
        )

        assert result["ids"] == [legacy]
        assert db.get_task(conn, session_task) is not None
        assert db.get_task(conn, running) is not None
        assert db.get_task(conn, other_legacy) is not None  # 没在 ids 里，不动
        assert infra_paths.task_dir(session_task).exists()


def test_purge_with_nothing_to_do_is_a_noop(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        _make_task(conn, kind="eval_session", params={"session_id": 1})
        result = eval_cleanup.purge_legacy_jobs(conn)

    assert result == {
        "removed_rows": 0, "removed_dirs": 0, "freed_bytes": 0, "ids": [],
    }


def test_purge_handles_row_without_directory(isolated) -> None:
    """DB 行在但目录已被手动删掉 → 行仍要清，不报错。"""
    with db.connection_for(isolated["db"]) as conn:
        legacy = _make_task(conn, kind="eval_samples")
        import shutil
        shutil.rmtree(infra_paths.task_dir(legacy))

        result = eval_cleanup.purge_legacy_jobs(conn)

        assert result["removed_rows"] == 1
        assert result["removed_dirs"] == 0
        assert db.get_task(conn, legacy) is None


def test_purge_batches_beyond_sqlite_variable_limit(isolated) -> None:
    """SQLite 默认变量上限 999 —— 上千条要分批 DELETE，不能一条 SQL 塞完。"""
    with db.connection_for(isolated["db"]) as conn:
        for _ in range(1200):
            db.create_task(
                conn, name="eval_clip", config_name="eval_clip",
                task_type="eval_clip", params={},
            )
        conn.execute("UPDATE tasks SET status = 'done'")
        conn.commit()

        result = eval_cleanup.purge_legacy_jobs(conn, with_size=False)

        assert result["removed_rows"] == 1200
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# 一次性标记
# ---------------------------------------------------------------------------

def test_startup_cleanup_runs_once_then_marks_done(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        legacy = _make_task(conn, kind="eval_samples")
        assert eval_cleanup.already_done(conn) is False

        first = eval_cleanup.cleanup_legacy_eval_on_startup(conn)

        assert first["skipped"] is False
        assert first["removed_rows"] == 1
        assert eval_cleanup.already_done(conn) is True
        assert db.get_task(conn, legacy) is None

        # 第二次直接跳过，连扫描都不做
        again = _make_task(conn, kind="eval_clip")
        second = eval_cleanup.cleanup_legacy_eval_on_startup(conn)

        assert second["skipped"] is True
        assert db.get_task(conn, again) is not None


def test_startup_cleanup_on_clean_db_still_marks_done(isolated) -> None:
    """干净库（没有存量）跑一次扫描就写标记退场，不会每次启动都扫。"""
    with db.connection_for(isolated["db"]) as conn:
        result = eval_cleanup.cleanup_legacy_eval_on_startup(conn)

        assert result["skipped"] is False
        assert result["removed_rows"] == 0
        assert eval_cleanup.already_done(conn) is True
