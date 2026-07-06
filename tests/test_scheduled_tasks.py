"""0.17 P-B 计划任务：DB 层 promote + supervisor tick 提升 / 取消。

端点层测试在 test_studio_queue_endpoints.py（enqueue scheduled_at / start_now /
cancel）与 test_train_endpoints.py（训练入队定时 + active 检查）。
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


def _task_status(db_path: Path, tid: int) -> str:
    with db.connection_for(db_path) as conn:
        task = db.get_task(conn, tid)
    return task["status"] if task else "?"


# ---------------------------------------------------------------------------
# DB 层
# ---------------------------------------------------------------------------


@pytest.fixture
def dbfile(tmp_path: Path) -> Path:
    path = tmp_path / "studio.db"
    db.init_db(path)
    return path


def test_create_task_with_scheduled_at(dbfile: Path) -> None:
    """带 scheduled_at → scheduled；不带 → pending（原行为）。"""
    with db.connection_for(dbfile) as conn:
        future = time.time() + 3600
        sid = db.create_task(conn, name="s", config_name="c", scheduled_at=future)
        pid = db.create_task(conn, name="p", config_name="c")
        s = db.get_task(conn, sid)
        p = db.get_task(conn, pid)
    assert s["status"] == "scheduled"
    assert s["scheduled_at"] == pytest.approx(future)
    assert p["status"] == "pending"
    assert p["scheduled_at"] is None


def test_promote_due_scheduled_only_promotes_due(dbfile: Path) -> None:
    """只提升到点的；scheduled_at 保留；第二次调用无事发生。"""
    with db.connection_for(dbfile) as conn:
        past = db.create_task(
            conn, name="due", config_name="c", scheduled_at=time.time() - 5
        )
        future = db.create_task(
            conn, name="later", config_name="c", scheduled_at=time.time() + 3600
        )
        promoted = db.promote_due_scheduled(conn)
        assert promoted == [past]
        assert db.get_task(conn, past)["status"] == "pending"
        assert db.get_task(conn, past)["scheduled_at"] is not None
        assert db.get_task(conn, future)["status"] == "scheduled"
        # 幂等：已提升的不再出现
        assert db.promote_due_scheduled(conn) == []


def test_scheduled_invisible_to_next_pending(dbfile: Path) -> None:
    """dispatcher 只看 pending —— scheduled 天然不被派活。"""
    with db.connection_for(dbfile) as conn:
        db.create_task(
            conn, name="s", config_name="c", scheduled_at=time.time() + 3600
        )
        assert db.next_pending(conn) is None
        db.promote_due_scheduled(conn, now=time.time() + 7200)
        assert db.next_pending(conn) is not None


def test_scheduled_in_live_statuses(dbfile: Path) -> None:
    """live 组（队列页数据源）包含 scheduled。"""
    with db.connection_for(dbfile) as conn:
        sid = db.create_task(
            conn, name="s", config_name="c", scheduled_at=time.time() + 3600
        )
        items = db.list_tasks_page(conn, statuses=db.LIVE_STATUSES)
    assert [i["id"] for i in items] == [sid]


# ---------------------------------------------------------------------------
# Supervisor：tick 提升 + 取消
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from studio.infrastructure import paths as _paths
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    logs.mkdir()
    configs.mkdir()
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(_paths, "LOGS_DIR", logs)
    (configs / "fake.yaml").write_text("epochs: 1\n", encoding="utf-8")
    return {"db": db_path, "logs": logs, "configs": configs}


def _events_collector():
    events: list[dict[str, Any]] = []
    def on_event(evt: dict[str, Any]) -> None:
        events.append(evt)
    return events, on_event


def test_due_scheduled_task_promoted_and_runs(env) -> None:
    """scheduled 到点 → tick 提升为 pending（带事件）→ 正常派活跑完。"""
    events, on_event = _events_collector()

    def fast_cmd(task, cfg):
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    sup = Supervisor(
        on_event=on_event, cmd_builder=fast_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(
            conn, name="t", config_name="fake", scheduled_at=time.time() - 1
        )

    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "done", timeout=10
        ), f"timeout; status={_task_status(env['db'], tid)}"
    finally:
        sup.stop()

    statuses = [e["status"] for e in events if e.get("task_id") == tid]
    # 提升事件（pending）在 running 之前
    assert "pending" in statuses
    assert statuses.index("pending") < statuses.index("running")


def test_future_scheduled_task_not_dispatched(env) -> None:
    """还没到点的 scheduled 不被提升 / 派活。"""
    events, on_event = _events_collector()

    def fast_cmd(task, cfg):
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    sup = Supervisor(
        on_event=on_event, cmd_builder=fast_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(
            conn, name="t", config_name="fake", scheduled_at=time.time() + 3600
        )

    sup.start()
    try:
        time.sleep(0.5)  # 若干个 tick
        assert _task_status(env["db"], tid) == "scheduled"
    finally:
        sup.stop()
    assert [e for e in events if e.get("task_id") == tid] == []


def test_cancel_scheduled_task(env) -> None:
    """scheduled 可直接取消（同 pending 路径）。"""
    events, on_event = _events_collector()
    sup = Supervisor(
        on_event=on_event,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(
            conn, name="t", config_name="fake", scheduled_at=time.time() + 3600
        )
    assert sup.cancel(tid) is True
    assert _task_status(env["db"], tid) == "canceled"
    assert [e["status"] for e in events if e.get("task_id") == tid] == ["canceled"]
