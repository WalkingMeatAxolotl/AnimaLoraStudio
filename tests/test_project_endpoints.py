"""PP1 — /api/projects + /api/projects/{pid}/versions HTTP 端到端。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.services.projects import projects, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
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
        f"/api/projects/{p['id']}", json={"note": "edited"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["note"] == "edited"


def test_delete_removes_dir(client: TestClient) -> None:
    p = client.post("/api/projects", json={"title": "ToDel"}).json()
    pdir = projects.project_dir(p["id"], p["slug"])
    assert pdir.exists()
    assert client.delete(f"/api/projects/{p['id']}").status_code == 200
    assert client.get(f"/api/projects/{p['id']}").status_code == 404
    assert not pdir.exists()


# ---------------------------------------------------------------------------
# archive（v12 软隐藏）
# ---------------------------------------------------------------------------


def test_archive_roundtrip(client: TestClient) -> None:
    p = client.post("/api/projects", json={"title": "Arch"}).json()
    pid = p["id"]
    pdir = projects.project_dir(pid, p["slug"])

    resp = client.post(f"/api/projects/{pid}/archive")
    assert resp.status_code == 200, resp.text
    assert resp.json()["archived_at"] is not None
    # 归档只是软隐藏：目录原样、list 仍返回该行（带 archived_at 给前端切分）
    assert pdir.exists()
    items = client.get("/api/projects").json()["items"]
    assert [r["id"] for r in items if r["archived_at"]] == [pid]

    resp = client.post(f"/api/projects/{pid}/unarchive")
    assert resp.status_code == 200
    assert resp.json()["archived_at"] is None


def test_archive_404(client: TestClient) -> None:
    assert client.post("/api/projects/9999/archive").status_code == 404


def test_archive_keeps_updated_at(client: TestClient) -> None:
    """归档/恢复不动 updated_at —— 恢复后按活跃时间排序的位置不变。"""
    p = client.post("/api/projects", json={"title": "Keep"}).json()
    client.post(f"/api/projects/{p['id']}/archive")
    after = client.post(f"/api/projects/{p['id']}/unarchive").json()
    assert after["updated_at"] == p["updated_at"]


def test_list_enriches_active_version_phase(client: TestClient) -> None:
    """preparing 项目的 list 行带 phase cursor（badge"准备中 · 打标"用）。"""
    client.post("/api/projects", json={"title": "PhaseTest"})
    row = client.get("/api/projects").json()["items"][0]
    assert row["active_version_status"] == "preparing"
    assert row["active_version_phase"] == "curating"


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


# ---------------------------------------------------------------------------
# PP7 — train.zip export / import
# ---------------------------------------------------------------------------


def test_train_zip_export_then_import(client: TestClient) -> None:
    """端到端：创建项目 → 放打标后的 train/ → 导出 zip → 上传 → 新项目应有同样的 train/。"""
    p = client.post("/api/projects", json={"title": "Round Trip"}).json()
    pid = p["id"]
    vid = p["versions"][0]["id"]
    train = versions.version_dir(pid, p["slug"], "v1") / "train" / "1_data"
    train.mkdir(parents=True, exist_ok=True)
    (train / "a.png").write_bytes(b"png")
    (train / "a.txt").write_text("tag1, tag2", encoding="utf-8")
    (train / "b.png").write_bytes(b"png2")

    # export
    resp = client.get(f"/api/projects/{pid}/versions/{vid}/train.zip")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/zip"
    assert "round-trip-v1.train.zip" in resp.headers.get("content-disposition", "")
    zip_bytes = resp.content
    assert len(zip_bytes) > 0

    # import
    resp = client.post(
        "/api/projects/import-train",
        files=[("file", ("round-trip-v1.train.zip", zip_bytes, "application/zip"))],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["project"]["id"] != pid
    # ADR-0007 PR-5: import-train 不再推 stage
    assert body["stats"]["image_count"] == 2
    assert body["stats"]["tagged_count"] == 1

    new_train = versions.version_dir(
        body["project"]["id"], body["project"]["slug"], "v1"
    ) / "train" / "1_data"
    assert (new_train / "a.png").exists()
    assert (new_train / "a.txt").read_text(encoding="utf-8") == "tag1, tag2"
    assert (new_train / "b.png").exists()


def test_train_zip_export_empty_returns_400(client: TestClient) -> None:
    p = client.post("/api/projects", json={"title": "Empty"}).json()
    vid = p["versions"][0]["id"]
    resp = client.get(f"/api/projects/{p['id']}/versions/{vid}/train.zip")
    assert resp.status_code == 400


def test_train_zip_export_alien_version_returns_404(client: TestClient) -> None:
    a = client.post("/api/projects", json={"title": "A"}).json()
    b = client.post("/api/projects", json={"title": "B"}).json()
    av = a["versions"][0]["id"]
    resp = client.get(f"/api/projects/{b['id']}/versions/{av}/train.zip")
    assert resp.status_code == 404


def test_import_train_rejects_zip_slip(client: TestClient) -> None:
    import io
    import json
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "exported_at": 0,
                    "source": {"title": "Evil", "version_label": "v1", "slug": "evil"},
                    "stats": {},
                }
            ),
        )
        zf.writestr("train/../escape.png", b"x")

    resp = client.post(
        "/api/projects/import-train",
        files=[("file", ("evil.zip", buf.getvalue(), "application/zip"))],
    )
    assert resp.status_code == 400


def test_import_train_rejects_corrupt_zip(client: TestClient) -> None:
    resp = client.post(
        "/api/projects/import-train",
        files=[("file", ("bad.zip", b"not a zip", "application/zip"))],
    )
    assert resp.status_code == 400
