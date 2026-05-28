"""PP2 — supervisor 调度 project_jobs：优先级 > task；tail 推 SSE；取消。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from studio import db
from studio.services.projects import jobs as project_jobs, projects
from studio.supervisor import Supervisor


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
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


def test_download_job_runs_in_parallel_with_training_task(isolated, tmp_path) -> None:
    """PP10.2.b：download job (IO-only) 应该跟训练 task 并行跑，不互相堵塞。"""
    p = _setup_project(isolated)
    events: list[dict] = []
    # task 走慢 sleep，download job 走慢 sleep；如果是串行，两者 running 不会重叠
    task_sleep = lambda t, _cfg: [
        sys.executable, "-c", "import time; time.sleep(0.5)"
    ]
    job_sleep = lambda j: [sys.executable, "-c", "import time; time.sleep(0.5)"]
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "fake.yaml").write_text("epochs: 1\n", encoding="utf-8")

    sup = Supervisor(
        on_event=events.append,
        cmd_builder=task_sleep,
        job_cmd_builder=job_sleep,
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        configs_dir=configs,
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="t1", config_name="fake")
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="download", params={}
        )
    sup.start()
    try:
        # 等到两边都进入 running
        assert _wait_until(
            lambda: any(
                e.get("type") == "task_state_changed" and e.get("status") == "running"
                for e in events
            )
            and any(
                e.get("type") == "job_state_changed" and e.get("status") == "running"
                for e in events
            ),
            timeout=5.0,
        ), "task 和 download job 没能同时进入 running（疑似仍串行）"
    finally:
        sup.stop(timeout=10.0)


def test_gpu_job_deferred_during_training_by_default(isolated, tmp_path) -> None:
    """PP10.2.b：训练 task 在跑时，tag / reg_build job 默认推迟（避免抢 GPU）。"""
    p = _setup_project(isolated)
    events: list[dict] = []
    task_sleep = lambda t, _cfg: [
        sys.executable, "-c", "import time; time.sleep(0.6)"
    ]
    job_quick = lambda j: [sys.executable, "-c", "print('tag done')"]
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "fake.yaml").write_text("epochs: 1\n", encoding="utf-8")

    sup = Supervisor(
        on_event=events.append,
        cmd_builder=task_sleep,
        job_cmd_builder=job_quick,
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        configs_dir=configs,
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="t1", config_name="fake")
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="tag", params={}
        )
    sup.start()
    try:
        # 等到训练 task running
        assert _wait_until(
            lambda: any(
                e.get("type") == "task_state_changed" and e.get("status") == "running"
                for e in events
            ),
            timeout=5.0,
        )
        # 给 tag job 一点时间被错误调度起来 — 它**不应**跑
        time.sleep(0.3)
        with db.connection_for(isolated["db"]) as conn:
            job_now = project_jobs.get_job(conn, job["id"])
        assert job_now["status"] == "pending", \
            f"tag job 不应在训练中起跑，当前状态={job_now['status']}"
        # 等训练结束 → tag job 应该自动跑起来
        assert _wait_until(
            lambda: project_jobs.get_job(
                db.connect(isolated["db"]), job["id"]
            )["status"] == "done",
            timeout=10.0,
        )
    finally:
        sup.stop(timeout=10.0)


def test_gpu_job_runs_during_training_when_allowed(isolated, tmp_path, monkeypatch) -> None:
    """PP10.2.b：开 secrets.queue.allow_gpu_during_train=True 后，tag job
    可以跟训练 task 并行（用户自己确认显存够）。"""
    from studio import secrets as _sec
    # 写一份 secrets 进 tmp，monkeypatch 切到这里
    secrets_file = tmp_path / "secrets.json"
    monkeypatch.setattr(_sec, "SECRETS_FILE", secrets_file)
    sec = _sec.Secrets()
    sec.queue.allow_gpu_during_train = True
    _sec.save(sec)

    p = _setup_project(isolated)
    events: list[dict] = []
    task_sleep = lambda t, _cfg: [
        sys.executable, "-c", "import time; time.sleep(0.5)"
    ]
    job_sleep = lambda j: [sys.executable, "-c", "import time; time.sleep(0.5)"]
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "fake.yaml").write_text("epochs: 1\n", encoding="utf-8")

    sup = Supervisor(
        on_event=events.append,
        cmd_builder=task_sleep,
        job_cmd_builder=job_sleep,
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        configs_dir=configs,
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="t1", config_name="fake")
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="tag", params={}
        )
    sup.start()
    try:
        # 应该两边都跑起来
        assert _wait_until(
            lambda: any(
                e.get("type") == "task_state_changed" and e.get("status") == "running"
                for e in events
            )
            and any(
                e.get("type") == "job_state_changed" and e.get("status") == "running"
                for e in events
            ),
            timeout=5.0,
        ), "开关打开后 tag job 仍被推迟"
    finally:
        sup.stop(timeout=10.0)


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


def test_worker_event_marker_lines_publish_typed_events(isolated) -> None:
    """Worker 写 `__EVENT__:type:json` → supervisor 解析后 publish typed SSE 事件，
    并且不让标记行进入 job_log。给前端"无轮询的实时进度"用。"""
    p = _setup_project(isolated)
    events: list[dict] = []
    # worker 模拟：先打一条普通日志，再打两条事件标记，再退出
    cmd = [
        sys.executable, "-c",
        'import json; print("hello"); '
        'print("__EVENT__:preprocess_progress:" + json.dumps({"idx":1,"total":3,"status":"done"})); '
        'print("__EVENT__:preprocess_progress:" + json.dumps({"idx":2,"total":3,"status":"done"}))',
    ]
    sup = Supervisor(
        on_event=events.append,
        job_cmd_builder=lambda _j: cmd,
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

    progress = [e for e in events if e.get("type") == "preprocess_progress"]
    assert len(progress) == 2
    assert progress[0]["idx"] == 1 and progress[0]["total"] == 3
    assert progress[1]["idx"] == 2
    # job_id / project_id 由 supervisor 自动注入，不依赖 worker 传
    assert progress[0]["job_id"] == job["id"]
    assert progress[0]["project_id"] == p["id"]

    # 标记行不进 job_log；只有"hello"应该出现
    log_texts = [e.get("text", "") for e in events if e.get("type") == "job_log_appended"]
    assert any("hello" in t for t in log_texts)
    assert not any("__EVENT__" in t for t in log_texts)
