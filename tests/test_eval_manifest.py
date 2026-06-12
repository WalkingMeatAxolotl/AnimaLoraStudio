"""LoRA eval manifest service and HTTP endpoints."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.services import eval_manifest
from studio.services.projects import projects, versions


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


def _new_project(isolated) -> tuple[dict, dict, Path]:
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Eval Project")
        version = versions.create_version(
            conn, project_id=project["id"], label="baseline"
        )
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def _seed_train(vdir: Path) -> None:
    train = vdir / "train"
    (train / "1_data").mkdir(parents=True, exist_ok=True)
    (train / "5_face").mkdir(parents=True, exist_ok=True)
    (train / "1_data" / "a.png").write_bytes(b"png-a")
    (train / "1_data" / "a.txt").write_text("solo, red hair", encoding="utf-8")
    (train / "5_face" / "b.webp").write_bytes(b"webp-b")
    (train / "5_face" / "b.json").write_text(
        '{"tags": {"quality": ["masterpiece"], "character": "mika"}}',
        encoding="utf-8",
    )
    (train / "5_face" / "ignore.txt").write_text("not an image", encoding="utf-8")


def test_create_default_manifest_scans_train_and_keeps_relative_paths(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_train(vdir)

    manifest = eval_manifest.create_default_manifest(
        project, version, vdir, now=1000.0
    )

    assert manifest["schema_version"] == 1
    assert manifest["project_id"] == project["id"]
    assert manifest["version_id"] == version["id"]
    assert manifest["metadata"]["train_image_count"] == 2
    assert [item["image"] for item in manifest["heldout"]] == [
        "1_data/a.png",
        "5_face/b.webp",
    ]
    assert manifest["heldout"][0]["caption"] == "1_data/a.txt"
    assert manifest["heldout"][0]["prompt"] == "solo, red hair"
    assert manifest["heldout"][1]["caption"] == "5_face/b.json"
    assert "masterpiece" in manifest["heldout"][1]["prompt"]
    assert not (vdir / "eval" / "manifest.json").exists()


def test_save_manifest_normalizes_and_writes_json(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_train(vdir)

    saved = eval_manifest.save_manifest(
        project,
        version,
        vdir,
        {
            "source": "manual",
            "heldout": [
                {
                    "image": "1_data/a.png",
                    "caption": "1_data/a.txt",
                    "prompt": "solo",
                    "seed": -1,
                }
            ],
            "seeds": [-5, 42, 42],
            "generation": {"width": 16, "height": 99999, "steps": 0},
        },
        now=2000.0,
    )

    assert saved["heldout"][0]["seed"] == 0
    assert saved["seeds"] == [0, 42]
    assert saved["generation"]["width"] == 64
    assert saved["generation"]["height"] == 4096
    assert saved["generation"]["steps"] == 1
    path = vdir / "eval" / "manifest.json"
    assert path.exists()
    assert eval_manifest.load_manifest(vdir)["updated_at"] == 2000.0


def test_save_manifest_rejects_missing_or_escaping_paths(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_train(vdir)

    for image in ("../escape.png", "C:/escape.png"):
        with pytest.raises(eval_manifest.EvalManifestError, match="相对路径"):
            eval_manifest.save_manifest(
                project,
                version,
                vdir,
                {"heldout": [{"image": image}]},
            )

    with pytest.raises(eval_manifest.EvalManifestError, match="不存在"):
        eval_manifest.save_manifest(
            project,
            version,
            vdir,
            {"heldout": [{"image": "1_data/missing.png"}]},
        )


def _make(client: TestClient) -> tuple[int, int]:
    project = client.post("/api/projects", json={"title": "Eval HTTP"}).json()
    return project["id"], project["versions"][0]["id"]


def _vdir_for(pid: int, vid: int) -> Path:
    with db.connection_for() as conn:
        project = projects.get_project(conn, pid)
        version = versions.get_version(conn, vid)
    assert project and version
    return versions.version_dir(project["id"], project["slug"], version["label"])


def test_get_manifest_returns_default_without_writing(client: TestClient) -> None:
    pid, vid = _make(client)
    vdir = _vdir_for(pid, vid)
    _seed_train(vdir)

    resp = client.get(f"/api/projects/{pid}/versions/{vid}/eval/manifest")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["has_manifest"] is False
    assert body["manifest"] is None
    assert body["default_manifest"]["heldout"][0]["image"] == "1_data/a.png"
    assert body["path"] == "eval/manifest.json"
    assert not (vdir / "eval" / "manifest.json").exists()


def test_post_default_manifest_writes_file(client: TestClient) -> None:
    pid, vid = _make(client)
    vdir = _vdir_for(pid, vid)
    _seed_train(vdir)

    resp = client.post(f"/api/projects/{pid}/versions/{vid}/eval/manifest/default")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["has_manifest"] is True
    assert body["manifest"]["heldout"][0]["image"] == "1_data/a.png"
    assert (vdir / "eval" / "manifest.json").exists()

    got = client.get(f"/api/projects/{pid}/versions/{vid}/eval/manifest").json()
    assert got["has_manifest"] is True
    assert got["default_manifest"] is None
    assert got["manifest"]["heldout"][1]["image"] == "5_face/b.webp"


def test_put_manifest_validates_and_404s_wrong_project(client: TestClient) -> None:
    pid, vid = _make(client)
    other_pid, _ = _make(client)
    vdir = _vdir_for(pid, vid)
    _seed_train(vdir)

    bad = client.put(
        f"/api/projects/{pid}/versions/{vid}/eval/manifest",
        json={"manifest": {"heldout": [{"image": "../escape.png"}]}},
    )
    assert bad.status_code == 400

    good = client.put(
        f"/api/projects/{pid}/versions/{vid}/eval/manifest",
        json={
            "manifest": {
                "source": "manual",
                "heldout": [{"image": "1_data/a.png", "prompt": "solo"}],
            }
        },
    )
    assert good.status_code == 200, good.text
    assert good.json()["manifest"]["heldout"][0]["caption"] is None

    wrong = client.get(f"/api/projects/{other_pid}/versions/{vid}/eval/manifest")
    assert wrong.status_code == 404
