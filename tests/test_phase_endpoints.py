"""ADR-0007 §11.5-A: POST /api/projects/{pid}/versions/{vid}/advance-phase + skip-phase 测试。"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, secrets, server
from studio.services.projects import jobs as project_jobs, projects, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


@pytest.fixture
def client(isolated) -> TestClient:
    return TestClient(server.app)


def _make_pv(client: TestClient) -> tuple[dict, dict]:
    """创建项目 + 默认 version 并返回 (project, version)。"""
    p = client.post("/api/projects", json={"title": "P", "initial_version_label": "v1"}).json()
    v = p["versions"][0]
    return p, v


def _put_image(folder: Path, name: str, with_caption: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.png").write_bytes(b"fake")
    if with_caption:
        (folder / f"{name}.txt").write_text("tag", encoding="utf-8")


# ---------------------------------------------------------------------------
# advance-phase
# ---------------------------------------------------------------------------


def test_advance_404_on_wrong_project(client: TestClient) -> None:
    p, v = _make_pv(client)
    resp = client.post(f"/api/projects/99999/versions/{v['id']}/advance-phase")
    assert resp.status_code == 404


def test_advance_404_on_missing_version(client: TestClient) -> None:
    p, _ = _make_pv(client)
    resp = client.post(f"/api/projects/{p['id']}/versions/99999/advance-phase")
    assert resp.status_code == 404


def test_advance_fails_with_empty_train(client: TestClient) -> None:
    """curating + train 空 → advance 失败 + reason 包含训练集为空。"""
    p, v = _make_pv(client)
    resp = client.post(f"/api/projects/{p['id']}/versions/{v['id']}/advance-phase")
    assert resp.status_code == 200
    body = resp.json()
    assert body["advanced"] is False
    assert body["ok"] is False
    assert "训练集" in body["reason"]
    assert body["new_phase"] is None


def test_advance_curating_to_tagging_with_image(client: TestClient) -> None:
    p, v = _make_pv(client)
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    _put_image(vdir / "train" / "5_concept", "001", with_caption=False)

    resp = client.post(f"/api/projects/{p['id']}/versions/{v['id']}/advance-phase")
    assert resp.status_code == 200
    body = resp.json()
    assert body["advanced"] is True
    assert body["new_phase"] == "tagging"
    assert body["version"]["phase"] == "tagging"


def test_advance_tagging_fails_with_missing_caption(client: TestClient) -> None:
    p, v = _make_pv(client)
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    _put_image(vdir / "train" / "5_concept", "001", with_caption=False)
    _put_image(vdir / "train" / "5_concept", "002", with_caption=True)
    # cursor 推到 tagging
    with db.connection_for() as conn:
        versions.update_version(conn, v["id"], phase="tagging")

    resp = client.post(f"/api/projects/{p['id']}/versions/{v['id']}/advance-phase")
    body = resp.json()
    assert body["advanced"] is False
    assert "1 张" in body["reason"]


# ---------------------------------------------------------------------------
# skip-phase
# ---------------------------------------------------------------------------


def test_skip_404_on_wrong_project(client: TestClient) -> None:
    p, v = _make_pv(client)
    resp = client.post(f"/api/projects/99999/versions/{v['id']}/skip-phase")
    assert resp.status_code == 404


def test_skip_fails_when_phase_not_skippable(client: TestClient) -> None:
    """curating 不可跳过。"""
    p, v = _make_pv(client)
    resp = client.post(f"/api/projects/{p['id']}/versions/{v['id']}/skip-phase")
    body = resp.json()
    assert body["advanced"] is False
    assert "不可跳过" in body["reason"]


def test_skip_regularizing_jumps_to_ready(client: TestClient) -> None:
    p, v = _make_pv(client)
    with db.connection_for() as conn:
        versions.update_version(conn, v["id"], phase="regularizing")
    resp = client.post(f"/api/projects/{p['id']}/versions/{v['id']}/skip-phase")
    body = resp.json()
    assert body["advanced"] is True
    assert body["new_phase"] == "ready"
    assert body["version"]["phase"] == "ready"


def test_skip_regularizing_blocked_by_running_job(client: TestClient) -> None:
    p, v = _make_pv(client)
    with db.connection_for() as conn:
        versions.update_version(conn, v["id"], phase="regularizing")
        conn.execute(
            "INSERT INTO project_jobs(project_id, version_id, kind, params, status) "
            "VALUES (?, ?, 'reg_build', '{}', 'running')",
            (p["id"], v["id"]),
        )
        conn.commit()
    resp = client.post(f"/api/projects/{p['id']}/versions/{v['id']}/skip-phase")
    body = resp.json()
    assert body["advanced"] is False
    assert "正则" in body["reason"]
