"""0.17 P-G — GET /api/jobs 全局只读列表（数据作业区数据源）。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.services.projects import jobs as project_jobs, projects


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    return {"db": dbfile}


@pytest.fixture
def client(isolated) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


def _seed_job(pid: int, kind: str, status: str) -> int:
    with db.connection_for() as conn:
        job = project_jobs.create_job(
            conn, project_id=pid, kind=kind, params={"n": 1}
        )
        conn.execute(
            "UPDATE project_jobs SET status = ? WHERE id = ?", (status, job["id"])
        )
        conn.commit()
    return int(job["id"])


def _make_project(client: TestClient, title: str = "P") -> int:
    return int(client.post("/api/projects", json={"title": title}).json()["id"])


def test_group_live_returns_running_and_pending(client: TestClient) -> None:
    pid = _make_project(client)
    run = _seed_job(pid, "tag", "running")
    pend = _seed_job(pid, "download", "pending")
    _seed_job(pid, "reg_build", "done")
    items = client.get("/api/jobs?group=live").json()["items"]
    assert [i["id"] for i in items] == [pend, run]  # id DESC
    assert {i["status"] for i in items} == {"running", "pending"}


def test_group_history_paginates_with_total(client: TestClient) -> None:
    pid = _make_project(client)
    ids = [_seed_job(pid, "tag", "done") for _ in range(5)]
    body = client.get("/api/jobs?group=history&page=1&page_size=2").json()
    assert body["total"] == 5
    assert body["page_size"] == 2
    assert [i["id"] for i in body["items"]] == [ids[4], ids[3]]
    page3 = client.get("/api/jobs?group=history&page=3&page_size=2").json()
    assert [i["id"] for i in page3["items"]] == [ids[0]]


def test_kind_filter_applies_to_both_groups(client: TestClient) -> None:
    pid = _make_project(client)
    _seed_job(pid, "tag", "running")
    dl = _seed_job(pid, "download", "running")
    _seed_job(pid, "tag", "done")
    dl_done = _seed_job(pid, "download", "failed")

    live = client.get("/api/jobs?group=live&kind=download").json()["items"]
    assert [i["id"] for i in live] == [dl]
    hist = client.get("/api/jobs?group=history&kind=download").json()
    assert [i["id"] for i in hist["items"]] == [dl_done]
    assert hist["total"] == 1


def test_params_decoded_present(client: TestClient) -> None:
    pid = _make_project(client)
    _seed_job(pid, "tag", "running")
    items = client.get("/api/jobs?group=live").json()["items"]
    assert items[0]["params_decoded"] == {"n": 1}


def test_created_at_written_on_create(client: TestClient) -> None:
    """v16 — 新作业 create_job 时写入队时间。"""
    import time as _time
    pid = _make_project(client)
    before = _time.time()
    _seed_job(pid, "tag", "pending")
    items = client.get("/api/jobs?group=live").json()["items"]
    assert items[0]["created_at"] is not None
    assert before - 1 <= items[0]["created_at"] <= _time.time() + 1


def test_q_searches_project_title_and_slug(client: TestClient) -> None:
    """q 按所属项目 title/slug 搜（job 自身无 name），live+history 都生效且 total 准。"""
    pid_a = _make_project(client, title="usashiro")
    pid_b = _make_project(client, title="other")
    a_run = _seed_job(pid_a, "tag", "running")
    _seed_job(pid_b, "tag", "running")
    a_done = _seed_job(pid_a, "download", "done")
    _seed_job(pid_b, "download", "done")

    live = client.get("/api/jobs?group=live&q=usa").json()["items"]
    assert [i["id"] for i in live] == [a_run]
    hist = client.get("/api/jobs?group=history&q=usa").json()
    assert [i["id"] for i in hist["items"]] == [a_done]
    assert hist["total"] == 1
    # 未命中 → 空
    assert client.get("/api/jobs?group=live&q=zzz").json()["items"] == []


def test_invalid_group_and_kind_400(client: TestClient) -> None:
    assert client.get("/api/jobs?group=banana").status_code == 400
    assert client.get("/api/jobs?group=live&kind=banana").status_code == 400


def test_page_size_clamped(client: TestClient) -> None:
    pid = _make_project(client)
    _seed_job(pid, "tag", "done")
    body = client.get("/api/jobs?group=history&page_size=9999").json()
    assert body["page_size"] == 100
