"""PP1 — /api/projects + /api/projects/{pid}/versions HTTP 端到端。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, projects, server, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    tdir = tmp_path / "_trash" / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(projects, "TRASH_DIR", tdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    return {"db": dbfile}


@pytest.fixture
def client(isolated) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# projects CRUD
# ---------------------------------------------------------------------------


def test_create_then_list(client: TestClient) -> None:
    assert client.get("/api/projects").json()["items"] == []
    resp = client.post(
        "/api/projects", json={"title": "Cosmic Kaguya", "note": "first"}
    )
    assert resp.status_code == 200, resp.text
    p = resp.json()
    assert p["slug"] == "cosmic-kaguya"
    assert len(p["versions"]) == 1
    assert p["versions"][0]["label"] == "v1"

    items = client.get("/api/projects").json()["items"]
    assert len(items) == 1
    assert items[0]["slug"] == "cosmic-kaguya"


def test_create_with_slug_override_and_no_initial_version(
    client: TestClient,
) -> None:
    resp = client.post(
        "/api/projects",
        json={
            "title": "Anything",
            "slug": "custom-slug",
            "initial_version_label": None,
        },
    )
    assert resp.status_code == 200
    p = resp.json()
    assert p["slug"] == "custom-slug"
    assert p["versions"] == []
    assert p["active_version_id"] is None


def test_create_rejects_empty_title(client: TestClient) -> None:
    resp = client.post("/api/projects", json={"title": "   "})
    assert resp.status_code == 400


def test_get_404(client: TestClient) -> None:
    assert client.get("/api/projects/9999").status_code == 404


def test_patch_updates_note_and_stage(client: TestClient) -> None:
    p = client.post("/api/projects", json={"title": "X"}).json()
    resp = client.patch(
        f"/api/projects/{p['id']}", json={"note": "edited", "stage": "curating"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["note"] == "edited"
    assert body["stage"] == "curating"


def test_delete_moves_to_trash_and_empty_trash(client: TestClient) -> None:
    p = client.post("/api/projects", json={"title": "ToDel"}).json()
    pdir = projects.project_dir(p["id"], p["slug"])
    assert pdir.exists()
    assert client.delete(f"/api/projects/{p['id']}").status_code == 200
    assert client.get(f"/api/projects/{p['id']}").status_code == 404
    assert not pdir.exists()
    r = client.post("/api/projects/_trash/empty").json()
    assert r["removed"] == 1


# ---------------------------------------------------------------------------
# versions CRUD
# ---------------------------------------------------------------------------


def test_version_create_list_get(client: TestClient) -> None:
    p = client.post("/api/projects", json={"title": "P"}).json()
    pid = p["id"]
    resp = client.post(
        f"/api/projects/{pid}/versions", json={"label": "high-lr"}
    )
    assert resp.status_code == 200, resp.text
    v = resp.json()
    assert v["label"] == "high-lr"
    items = client.get(f"/api/projects/{pid}/versions").json()["items"]
    assert {x["label"] for x in items} == {"v1", "high-lr"}
    got = client.get(f"/api/projects/{pid}/versions/{v['id']}")
    assert got.status_code == 200
    assert "stats" in got.json()


def test_version_label_must_be_unique_in_project(client: TestClient) -> None:
    p = client.post("/api/projects", json={"title": "P"}).json()
    resp = client.post(
        f"/api/projects/{p['id']}/versions", json={"label": "v1"}
    )
    assert resp.status_code == 400


def test_version_activate_updates_project(client: TestClient) -> None:
    p = client.post("/api/projects", json={"title": "P"}).json()
    v2 = client.post(
        f"/api/projects/{p['id']}/versions", json={"label": "v2"}
    ).json()
    resp = client.post(
        f"/api/projects/{p['id']}/versions/{v2['id']}/activate"
    )
    assert resp.status_code == 200
    assert resp.json()["active_version_id"] == v2["id"]


def test_version_delete_endpoint(client: TestClient) -> None:
    p = client.post("/api/projects", json={"title": "P"}).json()
    v = client.post(
        f"/api/projects/{p['id']}/versions", json={"label": "extra"}
    ).json()
    assert (
        client.delete(f"/api/projects/{p['id']}/versions/{v['id']}").status_code
        == 200
    )
    items = client.get(f"/api/projects/{p['id']}/versions").json()["items"]
    assert {x["label"] for x in items} == {"v1"}


def test_alien_version_404(client: TestClient) -> None:
    a = client.post("/api/projects", json={"title": "A"}).json()
    b = client.post("/api/projects", json={"title": "B"}).json()
    av = a["versions"][0]["id"]
    # 在 b 路径下访问 a 的 version → 404
    assert (
        client.get(f"/api/projects/{b['id']}/versions/{av}").status_code == 404
    )
