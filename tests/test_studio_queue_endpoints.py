"""/api/queue/* 端点测试。

不启动真正的 supervisor —— 用 monkeypatch 把 server 模块里的 db 路径、
presets 目录、logs 目录都指到 tmp_path，禁用 lifespan（跳过 supervisor 启动），
单独构造 Supervisor 注入到 app.state.supervisor。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """隔离 db / presets / logs 到 tmp_path。"""
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    presets = tmp_path / "presets"
    logs = tmp_path / "logs"
    presets.mkdir()
    logs.mkdir()
    (presets / "good.yaml").write_text("epochs: 1\n", encoding="utf-8")

    # server 端点引用 STUDIO_DB / USER_PRESETS_DIR / LOGS_DIR 三个常量
    monkeypatch.setattr(server, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server, "USER_PRESETS_DIR", presets)
    monkeypatch.setattr(server, "LOGS_DIR", logs)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)  # connect() 默认路径
    return tmp_path


class _StubSupervisor:
    """端点级测试用的取消器替身：避免真启子进程。"""
    def __init__(self) -> None:
        self.canceled: list[int] = []
        self.current_task_id: int | None = None
    def cancel(self, task_id: int) -> bool:
        with db.connection_for() as conn:
            task = db.get_task(conn, task_id)
            if not task or task["status"] not in ("pending", "running"):
                return False
            db.update_task(conn, task_id, status="canceled")
        self.canceled.append(task_id)
        return True


@pytest.fixture
def client(isolated: Path) -> TestClient:
    """绕过 lifespan：直接装一个 stub supervisor 到 app.state。"""
    server.app.state.supervisor = _StubSupervisor()
    # TestClient 不触发 lifespan，避免真的启动 supervisor 线程
    return TestClient(server.app)


# ---------------------------------------------------------------------------


def test_empty_queue(client: TestClient) -> None:
    resp = client.get("/api/queue")
    assert resp.status_code == 200
    assert resp.json()["items"] == []


def test_enqueue_and_get(client: TestClient) -> None:
    resp = client.post("/api/queue", json={"config_name": "good", "name": "task1"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["config_name"] == "good"
    assert data["status"] == "pending"
    tid = data["id"]

    got = client.get(f"/api/queue/{tid}")
    assert got.status_code == 200
    assert got.json()["id"] == tid


def test_enqueue_missing_config_404(client: TestClient) -> None:
    resp = client.post("/api/queue", json={"config_name": "ghost"})
    assert resp.status_code == 404


def test_filter_by_status(client: TestClient) -> None:
    client.post("/api/queue", json={"config_name": "good", "name": "a"})
    items = client.get("/api/queue?status=pending").json()["items"]
    assert len(items) == 1
    assert client.get("/api/queue?status=done").json()["items"] == []


def test_invalid_status_400(client: TestClient) -> None:
    resp = client.get("/api/queue?status=banana")
    assert resp.status_code == 400


def test_cancel_pending(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    resp = client.post(f"/api/queue/{tid}/cancel")
    assert resp.status_code == 200
    with db.connection_for() as conn:
        assert db.get_task(conn, tid)["status"] == "canceled"


def test_cancel_already_terminal_400(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="done")
    resp = client.post(f"/api/queue/{tid}/cancel")
    assert resp.status_code == 400


def test_retry_terminal_creates_new(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="failed")
    resp = client.post(f"/api/queue/{tid}/retry")
    assert resp.status_code == 200
    new_id = resp.json()["id"]
    assert new_id != tid
    assert resp.json()["status"] == "pending"


def test_retry_running_400(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="running")
    resp = client.post(f"/api/queue/{tid}/retry")
    assert resp.status_code == 400


def test_retry_copies_full_training_context(client: TestClient) -> None:
    """retry 必须复制 config_path / project_id / version_id，否则重试会
    走老降级路径用全局 preset 而不是 version 私有 config。"""
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    with db.connection_for() as conn:
        db.update_task(
            conn,
            tid,
            status="failed",
            config_path="/abs/path/to/version_private.yaml",
            project_id=42,
            version_id=99,
        )
    new = client.post(f"/api/queue/{tid}/retry").json()
    assert new["config_path"] == "/abs/path/to/version_private.yaml"
    assert new["project_id"] == 42
    assert new["version_id"] == 99
    # 「上次跑」的字段不应该带过来
    assert new["status"] == "pending"
    assert new.get("started_at") is None
    assert new.get("finished_at") is None
    assert new.get("error_msg") is None
    assert new.get("monitor_state_path") is None


def test_delete_only_terminal(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    # pending 状态不能删
    assert client.delete(f"/api/queue/{tid}").status_code == 400
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="done")
    assert client.delete(f"/api/queue/{tid}").status_code == 200
    assert client.get(f"/api/queue/{tid}").status_code == 404


def test_reorder(client: TestClient) -> None:
    a = client.post("/api/queue", json={"config_name": "good", "name": "a"}).json()["id"]
    b = client.post("/api/queue", json={"config_name": "good", "name": "b"}).json()["id"]
    resp = client.post("/api/queue/reorder", json={"ordered_ids": [b, a]})
    assert resp.status_code == 200
    items = client.get("/api/queue?status=pending").json()["items"]
    assert [i["id"] for i in items] == [b, a]


def test_logs_missing_returns_empty(client: TestClient) -> None:
    resp = client.get("/api/logs/9999")
    assert resp.status_code == 200
    assert resp.json()["content"] == ""


def test_logs_returns_content(client: TestClient, isolated: Path) -> None:
    log_path = isolated / "logs" / "42.log"
    log_path.write_text("hello world\n", encoding="utf-8")
    resp = client.get("/api/logs/42")
    assert resp.status_code == 200
    assert resp.json()["content"] == "hello world\n"
