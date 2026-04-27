"""Supervisor 端到端测试：用一个快速 sleep/exit 假 worker 替代 anima_train.py。

通过 cmd_builder 注入子进程命令，避免依赖真实训练栈。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest

from studio import db
from studio.supervisor import Supervisor


def _wait_for(predicate, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def env(tmp_path: Path):
    """初始化 db + 目录，并提供一个有效 config 文件。"""
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    logs.mkdir()
    configs.mkdir()
    (configs / "fake.yaml").write_text("epochs: 1\n", encoding="utf-8")
    return {"db": db_path, "logs": logs, "configs": configs}


def _events_collector():
    events: list[dict[str, Any]] = []
    def on_event(evt: dict[str, Any]) -> None:
        events.append(evt)
    return events, on_event


def test_pending_task_runs_to_completion(env) -> None:
    events, on_event = _events_collector()

    def fast_cmd(task, cfg):
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    sup = Supervisor(
        on_event=on_event, cmd_builder=fast_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "done", timeout=10
        ), f"timeout waiting for done; status={_task_status(env['db'], tid)}"
    finally:
        sup.stop()

    statuses = [e["status"] for e in events if e["task_id"] == tid]
    assert "running" in statuses
    assert "done" in statuses


def test_failed_task_marked_failed(env) -> None:
    events, on_event = _events_collector()

    def fail_cmd(task, cfg):
        return [sys.executable, "-c", "import sys; sys.exit(1)"]

    sup = Supervisor(
        on_event=on_event, cmd_builder=fail_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "failed", timeout=10
        )
    finally:
        sup.stop()

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task["exit_code"] == 1
    assert "exit code 1" in (task["error_msg"] or "")


def test_missing_config_marks_failed(env) -> None:
    """config 文件不存在时，supervisor 应立即把任务标 failed。"""
    events, on_event = _events_collector()

    sup = Supervisor(
        on_event=on_event,
        cmd_builder=lambda *_: [sys.executable, "-c", "pass"],
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="does_not_exist")

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "failed", timeout=5
        )
    finally:
        sup.stop()

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert "preset not found" in (task["error_msg"] or "")


def test_serial_execution(env) -> None:
    """两个任务排队，应先后串行执行。"""
    events, on_event = _events_collector()

    def slow_cmd(task, cfg):
        return [sys.executable, "-c", "import time; time.sleep(0.4)"]

    sup = Supervisor(
        on_event=on_event, cmd_builder=slow_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        a = db.create_task(conn, name="a", config_name="fake")
        b = db.create_task(conn, name="b", config_name="fake")

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], a) == "done"
                  and _task_status(env["db"], b) == "done",
            timeout=15,
        )
    finally:
        sup.stop()

    # a 的 finished_at 应早于 b 的 started_at
    with db.connection_for(env["db"]) as conn:
        ta = db.get_task(conn, a)
        tb = db.get_task(conn, b)
    assert ta["finished_at"] <= tb["started_at"] + 0.05


def test_cancel_pending(env) -> None:
    """pending 任务取消：直接标 canceled，不启动子进程。"""
    sup = Supervisor(
        cmd_builder=lambda *_: [sys.executable, "-c", "pass"],
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=10,  # 防止它真的拉起
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")
    assert sup.cancel(tid) is True
    assert _task_status(env["db"], tid) == "canceled"


def test_orphan_running_marked_failed_on_start(env) -> None:
    """启动时清理 status='running' 但 pid 已死的任务。"""
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="fake")
        db.update_task(conn, tid, status="running", pid=999999)

    sup = Supervisor(
        cmd_builder=lambda *_: [sys.executable, "-c", "pass"],
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "failed", timeout=5
        )
    finally:
        sup.stop()

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert "supervisor restart" in (task["error_msg"] or "")


def _task_status(dbfile: Path, tid: int) -> str:
    with db.connection_for(dbfile) as conn:
        task = db.get_task(conn, tid)
    return task["status"] if task else "missing"
