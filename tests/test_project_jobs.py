"""PP2 — project_jobs DAO + log_path + status transitions."""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db, project_jobs, projects


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(projects, "TRASH_DIR", tmp_path / "_trash")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    # 建一个父 project，FK 才能成立
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
    return {"db": dbfile, "project_id": p["id"]}


def test_create_job_assigns_log_path(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn,
            project_id=isolated['project_id'],
            kind="download",
            params={"tag": "x", "count": 5},
        )
    assert job["status"] == "pending"
    assert job["log_path"].endswith(f"{job['id']}.log")
    assert job["params_decoded"] == {"tag": "x", "count": 5}


def test_create_rejects_unknown_kind(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(project_jobs.JobError):
            project_jobs.create_job(
                conn, project_id=isolated['project_id'], kind="bogus", params={}
            )


def test_status_transitions(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=isolated['project_id'], kind="download", params={}
        )
        project_jobs.mark_running(conn, job["id"], pid=1234)
        running = project_jobs.get_job(conn, job["id"])
        assert running["status"] == "running"
        assert running["pid"] == 1234
        assert running["started_at"] is not None
        project_jobs.mark_done(conn, job["id"])
        done = project_jobs.get_job(conn, job["id"])
        assert done["status"] == "done"
        assert done["finished_at"] is not None


def test_mark_failed_sets_error_msg(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=isolated['project_id'], kind="download", params={}
        )
        project_jobs.mark_failed(conn, job["id"], "boom")
        got = project_jobs.get_job(conn, job["id"])
    assert got["status"] == "failed"
    assert got["error_msg"] == "boom"


def test_next_pending_picks_oldest(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        a = project_jobs.create_job(conn, project_id=isolated['project_id'], kind="download", params={})
        b = project_jobs.create_job(conn, project_id=isolated['project_id'], kind="download", params={})
        project_jobs.mark_running(conn, b["id"])
        nxt = project_jobs.next_pending(conn)
    assert nxt and nxt["id"] == a["id"]


def test_latest_for_returns_most_recent(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        a = project_jobs.create_job(conn, project_id=isolated['project_id'], kind="download", params={})
        b = project_jobs.create_job(conn, project_id=isolated['project_id'], kind="download", params={})
        latest = project_jobs.latest_for(conn, project_id=isolated['project_id'], kind="download")
    assert latest and latest["id"] == b["id"]


def test_cleanup_orphan_running(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=isolated['project_id'], kind="download", params={}
        )
        project_jobs.mark_running(conn, job["id"])
        n = project_jobs.cleanup_orphan_running(conn)
        got = project_jobs.get_job(conn, job["id"])
    assert n == 1
    assert got["status"] == "failed"
    assert "orphan" in (got["error_msg"] or "")


def test_list_jobs_filters(isolated) -> None:
    pid = isolated["project_id"]
    with db.connection_for(isolated["db"]) as conn:
        # 第二个 project，验证按 project_id 过滤
        other = projects.create_project(conn, title="Other")
        project_jobs.create_job(conn, project_id=pid, kind="download", params={})
        b = project_jobs.create_job(conn, project_id=pid, kind="download", params={})
        project_jobs.create_job(conn, project_id=other["id"], kind="download", params={})
        project_jobs.mark_done(conn, b["id"])
        pending = project_jobs.list_jobs(conn, project_id=pid, status="pending")
    assert len(pending) == 1
    assert pending[0]["status"] == "pending"
