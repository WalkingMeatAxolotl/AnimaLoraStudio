"""R-5 — /api/queue 档位视图（resource_class）：数据视图 / GPU 视图同源过滤。

（本文件原是 P-G 的 GET /api/jobs 测试；R-5 删掉该端点后改测 /api/queue 的
resource_class 参数与默认排除。）
"""
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
        db.update_task(conn, job["id"], status=status)
    return int(job["id"])


def _seed_train(name: str, status: str) -> int:
    with db.connection_for() as conn:
        tid = db.create_task(conn, name=name, config_name=name)
        db.update_task(conn, tid, status=status)
    return tid


def _make_project(client: TestClient, title: str = "P") -> int:
    return int(client.post("/api/projects", json={"title": title}).json()["id"])


def test_data_class_returns_light_and_io_only(client: TestClient) -> None:
    """数据视图 = light+io 档；eval_samples（exclusive）不在其中。"""
    pid = _make_project(client)
    tag = _seed_job(pid, "tag", "running")
    dl = _seed_job(pid, "download", "pending")
    _seed_job(pid, "eval_samples", "pending")
    _seed_train("t", "running")

    items = client.get("/api/queue?group=live&resource_class=data").json()["items"]
    assert sorted(i["id"] for i in items) == sorted([tag, dl])


def test_exclusive_class_includes_eval_samples(client: TestClient) -> None:
    """GPU 视图 = exclusive 档：train/reg_ai/generate + eval_samples（锚点 §4-2）。"""
    pid = _make_project(client)
    ev = _seed_job(pid, "eval_samples", "pending")
    tr = _seed_train("t", "running")
    _seed_job(pid, "tag", "pending")

    items = client.get(
        "/api/queue?group=live&resource_class=exclusive"
    ).json()["items"]
    assert sorted(i["id"] for i in items) == sorted([ev, tr])


def test_default_group_excludes_job_kinds(client: TestClient) -> None:
    """不带 resource_class/types：默认仍排除数据作业（旧调用方兼容）。"""
    pid = _make_project(client)
    _seed_job(pid, "tag", "pending")
    tr = _seed_train("t", "pending")
    items = client.get("/api/queue?group=live").json()["items"]
    assert [i["id"] for i in items] == [tr]


def test_history_pagination_with_class(client: TestClient) -> None:
    pid = _make_project(client)
    ids = [_seed_job(pid, "tag", "done") for _ in range(5)]
    body = client.get(
        "/api/queue?group=history&resource_class=data&page=1&page_size=2"
    ).json()
    assert body["total"] == 5
    assert [i["id"] for i in body["items"]] == [ids[4], ids[3]]  # id DESC


def test_types_filter_takes_precedence_over_class(client: TestClient) -> None:
    pid = _make_project(client)
    tag = _seed_job(pid, "tag", "pending")
    _seed_job(pid, "download", "pending")
    items = client.get(
        "/api/queue?group=live&resource_class=data&types=tag"
    ).json()["items"]
    assert [i["id"] for i in items] == [tag]


def test_q_searches_project_title_and_slug(client: TestClient) -> None:
    """q 命中所属项目 title/slug（数据作业 name=kind 无搜索价值）。"""
    pid_a = _make_project(client, title="usashiro")
    pid_b = _make_project(client, title="other")
    a = _seed_job(pid_a, "tag", "running")
    _seed_job(pid_b, "tag", "running")

    items = client.get(
        "/api/queue?group=live&resource_class=data&q=usa"
    ).json()["items"]
    assert [i["id"] for i in items] == [a]


def test_invalid_resource_class_400(client: TestClient) -> None:
    r = client.get("/api/queue?group=live&resource_class=banana")
    assert r.status_code == 400


def test_params_decoded_present(client: TestClient) -> None:
    pid = _make_project(client)
    _seed_job(pid, "tag", "running")
    items = client.get("/api/queue?group=live&resource_class=data").json()["items"]
    assert items[0]["params_decoded"] == {"n": 1}
