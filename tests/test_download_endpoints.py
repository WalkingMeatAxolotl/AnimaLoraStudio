"""PP2 — /api/projects/{pid}/download + /api/jobs/* HTTP。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, project_jobs, projects, secrets, server


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(projects, "TRASH_DIR", tmp_path / "_trash")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    secrets.update({"gelbooru": {"user_id": "u", "api_key": "k"}})
    return {"db": dbfile}


class _StubSupervisor:
    def __init__(self) -> None:
        self.canceled: list[int] = []

    def cancel_job(self, jid: int) -> bool:
        with db.connection_for() as conn:
            j = project_jobs.get_job(conn, jid)
            if not j or j["status"] in project_jobs.TERMINAL_STATUSES:
                return False
            project_jobs.mark_canceled(conn, jid)
        self.canceled.append(jid)
        return True


@pytest.fixture
def client(isolated) -> TestClient:
    server.app.state.supervisor = _StubSupervisor()
    return TestClient(server.app)


def _make_project(client: TestClient) -> dict:
    return client.post(
        "/api/projects", json={"title": "P", "initial_version_label": None}
    ).json()


# ---------------------------------------------------------------------------
# start_download
# ---------------------------------------------------------------------------


def test_start_download_creates_job_and_advances_stage(
    client: TestClient,
) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "char_x", "count": 5},
    )
    assert resp.status_code == 200, resp.text
    job = resp.json()
    assert job["status"] == "pending"
    assert job["kind"] == "download"
    # project stage 推到 downloading
    p2 = client.get(f"/api/projects/{p['id']}").json()
    assert p2["stage"] == "downloading"


def test_start_download_rejects_empty_tag(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download", json={"tag": "  ", "count": 1}
    )
    assert resp.status_code == 400


def test_start_download_requires_credentials(
    client: TestClient, isolated, monkeypatch
) -> None:
    # gelbooru 缺凭据：拒绝
    monkeypatch.setattr(
        secrets, "has_credentials_for",
        lambda src: False if src == "gelbooru" else True,
    )
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "x", "count": 1, "api_source": "gelbooru"},
    )
    assert resp.status_code == 400
    assert "gelbooru" in resp.json()["detail"]


def test_start_download_danbooru_does_not_require_credentials(
    client: TestClient, isolated, monkeypatch
) -> None:
    """Danbooru 匿名也能跑，端点不应在缺凭据时阻挡。"""
    monkeypatch.setattr(
        secrets, "has_credentials_for",
        lambda src: True if src == "danbooru" else False,
    )
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "x", "count": 1, "api_source": "danbooru"},
    )
    assert resp.status_code == 200, resp.text


def test_estimate_endpoint_returns_count(
    client: TestClient, isolated, monkeypatch
) -> None:
    """estimate 端点：通过 mock downloader.estimate 返回固定数量。"""
    from studio.services import downloader as dl
    monkeypatch.setattr(dl, "estimate", lambda opts: 42)
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download/estimate",
        json={"tag": "x", "api_source": "gelbooru"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 42
    assert body["effective_query"] == "x"


def test_estimate_includes_exclude_tags(
    client: TestClient, isolated, monkeypatch
) -> None:
    secrets.update({"download": {"exclude_tags": ["comic", "monochrome"]}})
    from studio.services import downloader as dl
    monkeypatch.setattr(dl, "estimate", lambda opts: 7)
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download/estimate",
        json={"tag": "x", "api_source": "gelbooru"},
    )
    body = resp.json()
    assert body["exclude_tags"] == ["comic", "monochrome"]
    assert "-comic" in body["effective_query"]
    assert "-monochrome" in body["effective_query"]


def test_start_download_rejects_bad_source(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "x", "count": 1, "api_source": "wat"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# status / log
# ---------------------------------------------------------------------------


def test_download_status_returns_latest(client: TestClient) -> None:
    p = _make_project(client)
    j1 = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "a", "count": 1},
    ).json()
    j2 = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "b", "count": 1},
    ).json()
    r = client.get(f"/api/projects/{p['id']}/download/status").json()
    assert r["job"]["id"] == j2["id"]


def test_download_status_no_jobs(client: TestClient) -> None:
    p = _make_project(client)
    r = client.get(f"/api/projects/{p['id']}/download/status").json()
    assert r["job"] is None


def test_get_job_log_returns_tail(client: TestClient, isolated) -> None:
    p = _make_project(client)
    job = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "a", "count": 1},
    ).json()
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    r = client.get(f"/api/jobs/{job['id']}/log?tail=2").json()
    assert r["content"].splitlines() == ["c", "d"]


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def test_list_files_empty(client: TestClient) -> None:
    p = _make_project(client)
    r = client.get(f"/api/projects/{p['id']}/files").json()
    assert r == {"items": [], "count": 0}


def test_list_files_returns_images(client: TestClient) -> None:
    p = _make_project(client)
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    (pdir / "1.png").write_bytes(b"x")
    (pdir / "2.jpg").write_bytes(b"x")
    (pdir / "ignored.txt").write_bytes(b"x")
    r = client.get(f"/api/projects/{p['id']}/files").json()
    names = sorted(i["name"] for i in r["items"])
    assert names == ["1.png", "2.jpg"]


def test_thumb_serves_image(client: TestClient) -> None:
    p = _make_project(client)
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    (pdir / "1.png").write_bytes(b"\x89PNG fake")
    r = client.get(
        f"/api/projects/{p['id']}/thumb?bucket=download&name=1.png"
    )
    assert r.status_code == 200
    assert r.content == b"\x89PNG fake"


def test_thumb_rejects_path_traversal(client: TestClient) -> None:
    p = _make_project(client)
    r = client.get(
        f"/api/projects/{p['id']}/thumb?bucket=download&name=../etc/passwd"
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


def test_cancel_pending_job_endpoint(client: TestClient) -> None:
    p = _make_project(client)
    job = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "x", "count": 1},
    ).json()
    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 200
    again = client.get(f"/api/jobs/{job['id']}").json()
    assert again["status"] == "canceled"


def test_cancel_terminal_job_400(client: TestClient) -> None:
    p = _make_project(client)
    job = client.post(
        f"/api/projects/{p['id']}/download",
        json={"tag": "x", "count": 1},
    ).json()
    with db.connection_for() as conn:
        project_jobs.mark_done(conn, job["id"])
    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 400
