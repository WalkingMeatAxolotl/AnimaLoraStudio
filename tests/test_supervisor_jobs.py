"""PP2 — supervisor 调度 project_jobs：优先级 > task；tail 推 SSE；取消。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from studio import db, project_jobs, projects
from studio.supervisor import Supervisor


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(projects, "TRASH_DIR", tmp_path / "_trash")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    return {"db": dbfile, "logs": tmp_path / "logs"}


def _wait_until(pred, timeout: float = 5.0, step: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


def _setup_project(isolated) -> dict:
    with db.connection_for(isolated["db"]) as conn:
        return projects.create_project(conn, title="P")


def test_jobs_take_priority_over_tasks(isolated) -> None:
    """同时入队 task 与 job：job 先跑。"""
    p = _setup_project(isolated)
    events: list[dict] = []
    job_cmd = lambda j: [sys.executable, "-c", "print('hi')"]
    task_cmd = lambda t, _cfg: [sys.executable, "-c", "print('task')"]
    sup = Supervisor(
        on_event=events.append,
        cmd_builder=task_cmd,
        job_cmd_builder=job_cmd,
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        # 任意 task（preset 路径不存在 → 标 failed，但不会先于 job 跑）
        db.create_task(conn, name="t1", config_name="not_exist")
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="download", params={}
        )
    sup.start()
    try:
        assert _wait_until(
            lambda: any(
                e.get("type") == "job_state_changed" and e.get("status") == "running"
                for e in events
            )
        )
    finally:
        sup.stop(timeout=5.0)
    # job_running 事件出现的位置应该早于 task 的任何事件出现
    job_run_idx = next(
        i for i, e in enumerate(events)
        if e.get("type") == "job_state_changed" and e.get("status") == "running"
    )
    task_evt_indices = [
        i for i, e in enumerate(events) if e.get("type") == "task_state_changed"
    ]
    if task_evt_indices:
        assert job_run_idx < min(task_evt_indices)


def test_job_lifecycle_done(isolated) -> None:
    p = _setup_project(isolated)
    events: list[dict] = []
    sup = Supervisor(
        on_event=events.append,
        job_cmd_builder=lambda j: [sys.executable, "-c", "print('hello'); print('bye')"],
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="download", params={}
        )
    sup.start()
    try:
        assert _wait_until(
            lambda: any(
                e.get("type") == "job_state_changed" and e.get("status") == "done"
                for e in events
            )
        )
    finally:
        sup.stop(timeout=5.0)

    with db.connection_for(isolated["db"]) as conn:
        finished = project_jobs.get_job(conn, job["id"])
    assert finished["status"] == "done"
    assert finished["finished_at"] is not None

    log_lines = [e for e in events if e.get("type") == "job_log_appended"]
    assert any("hello" in (e.get("text") or "") for e in log_lines)


def test_job_lifecycle_failed(isolated) -> None:
    p = _setup_project(isolated)
    events: list[dict] = []
    sup = Supervisor(
        on_event=events.append,
        job_cmd_builder=lambda j: [sys.executable, "-c", "import sys; sys.exit(2)"],
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="download", params={}
        )
    sup.start()
    try:
        assert _wait_until(
            lambda: any(
                e.get("type") == "job_state_changed" and e.get("status") == "failed"
                for e in events
            )
        )
    finally:
        sup.stop(timeout=5.0)
    with db.connection_for(isolated["db"]) as conn:
        finished = project_jobs.get_job(conn, job["id"])
    assert finished["status"] == "failed"
    assert "exit code 2" in (finished["error_msg"] or "")


def test_cancel_pending_job(isolated) -> None:
    p = _setup_project(isolated)
    sup = Supervisor(
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        poll_interval=10.0,  # 不让循环跑
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="download", params={}
        )
    assert sup.cancel_job(job["id"]) is True
    with db.connection_for(isolated["db"]) as conn:
        got = project_jobs.get_job(conn, job["id"])
    assert got["status"] == "canceled"
