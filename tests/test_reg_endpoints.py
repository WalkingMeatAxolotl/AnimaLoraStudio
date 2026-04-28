"""PP5 — /api/projects/.../reg/* HTTP。"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio import db, project_jobs, projects, secrets, server, versions
from studio.services import reg_builder


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(projects, "TRASH_DIR", tmp_path / "_trash")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


@pytest.fixture
def client(env) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


def _make(client: TestClient) -> tuple[int, int]:
    p = client.post("/api/projects", json={"title": "P"}).json()
    return p["id"], p["versions"][0]["id"]


def _seed_train(
    client: TestClient, pid: int, vid: int, folder: str, files: dict[str, list[str]]
) -> Path:
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    train = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "train"
    d = train / folder
    d.mkdir(parents=True, exist_ok=True)
    for name, tags in files.items():
        # 真写一张 PNG（preview-tags 走 read_tags 但需要 IMAGE_EXTS 判定）
        img = Image.new("RGB", (32, 32), (255, 0, 0))
        img.save(d / name, "PNG")
        (d / name).with_suffix(".txt").write_text(", ".join(tags), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# preview-tags
# ---------------------------------------------------------------------------


def test_preview_tags_returns_top_freq(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {
        "a.png": ["1girl", "solo", "blue_hair"],
        "b.png": ["1girl", "long_hair"],
        "c.png": ["1girl", "outdoor"],
    })
    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg/preview-tags?top=3")
    assert r.status_code == 200
    items = r.json()["items"]
    # 1girl 最高（3 次）
    assert items[0]["tag"] == "1girl"
    assert items[0]["count"] == 3
    assert len(items) == 3


def test_preview_tags_empty_train(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg/preview-tags")
    assert r.status_code == 200
    assert r.json()["items"] == []


# ---------------------------------------------------------------------------
# GET /reg
# ---------------------------------------------------------------------------


def test_get_reg_when_not_exists(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg")
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is False
    assert body["meta"] is None
    assert body["image_count"] == 0


def test_get_reg_when_exists_returns_meta_and_files(
    client: TestClient, tmp_path: Path
) -> None:
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    rdir = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg" / "1_general"
    rdir.mkdir(parents=True)
    img = Image.new("RGB", (16, 16), (0, 0, 255))
    img.save(rdir / "100.png", "PNG")
    img.save(rdir / "200.png", "PNG")
    meta = reg_builder.RegMeta(
        generated_at=1.0, based_on_version="baseline", api_source="gelbooru",
        target_count=5, actual_count=2, source_tags=["1girl"],
        excluded_tags=[], blacklist_tags=[], failed_tags=[],
        train_tag_distribution={"1girl": 5}, auto_tagged=True,
    )
    reg_builder.write_meta(rdir, meta)

    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg")
    body = r.json()
    assert body["exists"] is True
    assert body["image_count"] == 2
    assert sorted(body["files"]) == ["100.png", "200.png"]
    assert body["meta"]["actual_count"] == 2
    assert body["meta"]["auto_tagged"] is True


# ---------------------------------------------------------------------------
# POST /reg/build
# ---------------------------------------------------------------------------


def test_start_reg_build_requires_train_data(client: TestClient) -> None:
    pid, vid = _make(client)
    # train 空 → 400
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build", json={}
    )
    assert r.status_code == 400


def test_start_reg_build_creates_job(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {
        "1.png": ["1girl"],
    })
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"target_count": 10, "excluded_tags": ["character_x"], "auto_tag": False},
    )
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["kind"] == "reg_build"
    assert job["status"] == "pending"
    p_dict = json.loads(job["params"])
    assert p_dict["target_count"] == 10
    assert p_dict["excluded_tags"] == ["character_x"]
    assert p_dict["auto_tag"] is False
    # version stage 推到 regularizing
    v = client.get(f"/api/projects/{pid}/versions/{vid}").json()
    assert v["stage"] == "regularizing"


def test_start_reg_build_rejects_bad_api_source(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"api_source": "pixiv"},
    )
    assert r.status_code == 400


def test_start_reg_build_rejects_negative_target(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"target_count": -5},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /reg
# ---------------------------------------------------------------------------


def test_delete_reg_removes_dir(client: TestClient) -> None:
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    rdir = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg" / "1_general"
    rdir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(rdir / "1.png", "PNG")
    assert rdir.exists()

    r = client.delete(f"/api/projects/{pid}/versions/{vid}/reg")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert not rdir.exists()


def test_delete_reg_when_not_exists(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.delete(f"/api/projects/{pid}/versions/{vid}/reg")
    assert r.status_code == 200
    assert r.json()["deleted"] is False
