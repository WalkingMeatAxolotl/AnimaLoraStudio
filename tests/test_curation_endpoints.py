"""PP3 — /api/projects/{pid}/versions/{vid}/curation HTTP。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, projects, server, versions


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(projects, "TRASH_DIR", tmp_path / "_trash")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    return {"db": dbfile}


@pytest.fixture
def client(env) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


def _make(client: TestClient) -> tuple[int, int]:
    p = client.post("/api/projects", json={"title": "P"}).json()
    return p["id"], p["versions"][0]["id"]


def _drop(client, pid: int, name: str = "1.png") -> Path:
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
    pdir = projects.project_dir(proj["id"], proj["slug"]) / "download"
    pdir.mkdir(parents=True, exist_ok=True)
    f = pdir / name
    f.write_bytes(b"\x89PNG fake")
    return f


# ---------------------------------------------------------------------------
# basic flow
# ---------------------------------------------------------------------------


def _names(entries: list[dict]) -> list[str]:
    return [e["name"] for e in entries]


def test_curation_view_initial_empty(client: TestClient) -> None:
    """新 version 默认有一个 1_data 训练文件夹，里面是空的。"""
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/curation").json()
    assert r == {
        "left": [],
        "right": {"1_data": []},
        "download_total": 0,
        "train_total": 0,
        "folders": ["1_data"],
    }


def test_copy_then_view(client: TestClient) -> None:
    pid, vid = _make(client)
    _drop(client, pid, "1.png")
    _drop(client, pid, "2.png")
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/copy",
        json={"files": ["1.png"], "dest_folder": "5_concept"},
    ).json()
    assert r["copied"] == ["1.png"]
    view = client.get(f"/api/projects/{pid}/versions/{vid}/curation").json()
    assert _names(view["left"]) == ["2.png"]
    # mtime 字段附带；前端按需排序
    assert all("mtime" in e for e in view["left"])
    assert _names(view["right"]["5_concept"]) == ["1.png"]
    assert view["right"]["1_data"] == []
    assert set(view["folders"]) == {"1_data", "5_concept"}


def test_copy_advances_stage(client: TestClient) -> None:
    pid, vid = _make(client)
    _drop(client, pid, "1.png")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/copy",
        json={"files": ["1.png"], "dest_folder": "5_x"},
    )
    proj = client.get(f"/api/projects/{pid}").json()
    assert proj["stage"] == "tagging"
    v = next(v for v in proj["versions"] if v["id"] == vid)
    assert v["stage"] == "tagging"


def test_remove_only_deletes_train(client: TestClient) -> None:
    pid, vid = _make(client)
    _drop(client, pid, "1.png")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/copy",
        json={"files": ["1.png"], "dest_folder": "5_x"},
    )
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/remove",
        json={"folder": "5_x", "files": ["1.png"]},
    ).json()
    assert r["removed"] == ["1.png"]
    # download/ 应保留
    view = client.get(f"/api/projects/{pid}/versions/{vid}/curation").json()
    assert _names(view["left"]) == ["1.png"]
    assert view["right"]["5_x"] == []
    assert view["right"]["1_data"] == []


# ---------------------------------------------------------------------------
# folder ops
# ---------------------------------------------------------------------------


def test_folder_create_rename_delete(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/folder",
        json={"op": "create", "name": "10_a"},
    )
    assert r.status_code == 200
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/folder",
        json={"op": "rename", "name": "10_a", "new_name": "5_b"},
    )
    assert r.status_code == 200
    view = client.get(f"/api/projects/{pid}/versions/{vid}/curation").json()
    assert "5_b" in view["folders"]
    assert "10_a" not in view["folders"]
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/folder",
        json={"op": "delete", "name": "5_b"},
    )
    assert r.status_code == 200
    view = client.get(f"/api/projects/{pid}/versions/{vid}/curation").json()
    # 默认 1_data 仍在
    assert view["folders"] == ["1_data"]


def test_folder_create_bad_name_400(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/folder",
        json={"op": "create", "name": "../etc"},
    )
    assert r.status_code == 400


def test_folder_rename_requires_new_name(client: TestClient) -> None:
    pid, vid = _make(client)
    client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/folder",
        json={"op": "create", "name": "x"},
    )
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/folder",
        json={"op": "rename", "name": "x"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# thumb
# ---------------------------------------------------------------------------


def test_version_thumb_serves_train_image(client: TestClient) -> None:
    pid, vid = _make(client)
    _drop(client, pid, "1.png")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/curation/copy",
        json={"files": ["1.png"], "dest_folder": "5_x"},
    )
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/thumb"
        "?bucket=train&folder=5_x&name=1.png"
    )
    assert r.status_code == 200
    assert r.content == b"\x89PNG fake"


def test_version_thumb_rejects_traversal(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/thumb"
        "?bucket=train&folder=../etc&name=passwd"
    )
    assert r.status_code == 400
