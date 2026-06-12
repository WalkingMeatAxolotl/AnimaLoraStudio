"""LoRA eval sample run service and endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.services import eval_manifest, eval_samples
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


def _new_project(isolated) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Eval Samples")
        version = versions.create_version(
            conn, project_id=project["id"], label="baseline"
        )
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def _seed_train_and_ckpt(vdir: Path) -> Path:
    train = vdir / "train" / "1_data"
    train.mkdir(parents=True, exist_ok=True)
    (train / "a.png").write_bytes(b"png-a")
    (train / "a.txt").write_text("solo, red hair", encoding="utf-8")
    (train / "b.png").write_bytes(b"png-b")
    (train / "b.txt").write_text("smile, blue eyes", encoding="utf-8")
    output = vdir / "output"
    output.mkdir(parents=True, exist_ok=True)
    ckpt = output / "model_step100.safetensors"
    ckpt.write_bytes(b"fake-lora")
    return ckpt


def _fake_generator(
    run: dict[str, Any], version_dir: Path, progress
) -> None:
    for idx, item in enumerate(run["items"]):
        progress(f"fake {idx}")
        run = eval_samples.mark_item_running(version_dir, run, idx)
        path = eval_samples.sample_image_path(
            version_dir, run["run_id"], item["filename"]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG")
        run = eval_samples.mark_item_done(version_dir, run, idx)


def test_create_and_run_eval_samples_with_fake_generator(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_train_and_ckpt(vdir)
    eval_manifest.save_default_manifest(project, version, vdir, now=1000.0)

    run = eval_samples.create_run(
        project,
        version,
        vdir,
        checkpoint_path="model_step100.safetensors",
        max_items=1,
        now=2000.0,
    )
    assert run["status"] == "pending"
    assert run["checkpoint"]["path"] == "output/model_step100.safetensors"
    assert run["summary"] == {
        "total": 1,
        "pending": 1,
        "running": 0,
        "done": 0,
        "failed": 0,
    }

    finished = eval_samples.run_sample_job(
        project,
        version,
        vdir,
        run["run_id"],
        generator=_fake_generator,
    )

    assert finished["status"] == "done"
    assert finished["summary"]["done"] == 1
    item = finished["items"][0]
    assert item["status"] == "done"
    assert item["prompt"] == "solo, red hair"
    assert (vdir / item["path"]).read_bytes() == b"PNG"


def test_start_job_creates_project_job_and_run(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_train_and_ckpt(vdir)

    with db.connection_for(isolated["db"]) as conn:
        job, run = eval_samples.start_job(
            conn,
            project,
            version,
            vdir,
            checkpoint_path="model_step100.safetensors",
            max_items=2,
        )

    assert job["kind"] == "eval_samples"
    assert job["version_id"] == version["id"]
    assert job["params_decoded"]["run_id"] == run["run_id"]
    assert len(run["items"]) == 2
    assert (vdir / "eval" / "manifest.json").exists()
    assert (vdir / "eval" / "samples" / run["run_id"] / "run.json").exists()


def _make(client: TestClient) -> tuple[int, int]:
    project = client.post("/api/projects", json={"title": "Eval Sample HTTP"}).json()
    return project["id"], project["versions"][0]["id"]


def _vdir_for(pid: int, vid: int) -> Path:
    with db.connection_for() as conn:
        project = projects.get_project(conn, pid)
        version = versions.get_version(conn, vid)
    assert project and version
    return versions.version_dir(project["id"], project["slug"], version["label"])


def test_eval_samples_http_start_list_get_and_image(client: TestClient) -> None:
    pid, vid = _make(client)
    vdir = _vdir_for(pid, vid)
    _seed_train_and_ckpt(vdir)

    created = client.post(
        f"/api/projects/{pid}/versions/{vid}/eval/samples",
        json={"checkpoint_path": "model_step100.safetensors", "max_items": 1},
    )
    assert created.status_code == 200, created.text
    body = created.json()
    run_id = body["run"]["run_id"]
    assert body["job"]["kind"] == "eval_samples"
    assert body["run"]["summary"]["total"] == 1

    listed = client.get(f"/api/projects/{pid}/versions/{vid}/eval/samples")
    assert listed.status_code == 200, listed.text
    assert listed.json()["runs"][0]["run_id"] == run_id
    assert listed.json()["latest_job"]["kind"] == "eval_samples"

    got = client.get(f"/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}")
    assert got.status_code == 200, got.text
    assert got.json()["run"]["items"][0]["filename"].endswith(".png")

    item = got.json()["run"]["items"][0]
    image_path = vdir / item["path"]
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"PNG")
    image = client.get(
        f"/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}"
        f"/images/{item['filename']}"
    )
    assert image.status_code == 200, image.text
    assert image.content == b"PNG"


def test_eval_samples_rejects_checkpoint_outside_output(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_train_and_ckpt(vdir)

    with pytest.raises(eval_samples.EvalSamplesError, match="output"):
        eval_samples.create_run(
            project,
            version,
            vdir,
            checkpoint_path="../escape.safetensors",
        )
