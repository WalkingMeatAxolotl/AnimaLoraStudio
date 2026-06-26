"""LoRA eval sample run service and endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.infrastructure import paths as infra_paths
from studio.services import eval_samples
from studio.services.projects import projects, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(infra_paths, "TASKS_DIR", tmp_path / "tasks")
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


def _seed_validation_and_ckpt(vdir: Path) -> Path:
    # held-out validation set = eval reference + prompts (replaces old manifest)
    val = vdir / "validation" / "1_data"
    val.mkdir(parents=True, exist_ok=True)
    (val / "a.png").write_bytes(b"png-a")
    (val / "a.txt").write_text("solo, red hair", encoding="utf-8")
    (val / "b.png").write_bytes(b"png-b")
    (val / "b.txt").write_text("smile, blue eyes", encoding="utf-8")
    output = vdir / "output"
    output.mkdir(parents=True, exist_ok=True)
    ckpt = output / "model_step100.safetensors"
    ckpt.write_bytes(b"fake-lora")
    return ckpt


def _fake_generator(
    run: dict[str, Any], version_dir: Path, progress
) -> None:
    eval_root = (
        Path(str(run["eval_root"]))
        if run.get("storage_scope") == "task" and run.get("eval_root")
        else None
    )
    for idx, item in enumerate(run["items"]):
        progress(f"fake {idx}")
        run = eval_samples.mark_item_running(version_dir, run, idx, eval_root)
        path = eval_samples.sample_image_path(
            version_dir, run["run_id"], item["filename"], eval_root
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG")
        run = eval_samples.mark_item_done(version_dir, run, idx, eval_root)


def test_create_and_run_eval_samples_with_fake_generator(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)

    run = eval_samples.create_run(
        project,
        version,
        vdir,
        checkpoint_path="model_step100.safetensors",
        now=2000.0,
    )
    assert run["status"] == "pending"
    assert run["checkpoint"]["path"] == "output/model_step100.safetensors"
    # validation set has 2 images → whole set evaluated, no max_items cap
    assert run["summary"] == {
        "total": 2,
        "pending": 2,
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
    assert finished["summary"]["done"] == 2
    item = finished["items"][0]
    assert item["status"] == "done"
    assert item["prompt"] == "solo, red hair"
    assert (vdir / item["path"]).read_bytes() == b"PNG"


def test_default_generator_path_has_no_dead_schema_import(isolated) -> None:
    """Regression: every other test injects a fake generator, so the real
    ``generator=None`` path (``_default_generator``) was never exercised and a
    stale ``from studio.schema import migrate_legacy_attention`` shipped — the
    symbol was removed in the 0.11.0 schema split, so the default auto-eval /
    worker path raised ImportError before generating anything.

    No version config is written here, so the real generator must fail with a
    *config* error, never an ImportError/NameError on that removed symbol.
    """
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)

    run = eval_samples.create_run(
        project,
        version,
        vdir,
        checkpoint_path="model_step100.safetensors",
        now=2000.0,
    )

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - asserting on message
        eval_samples.run_sample_job(project, version, vdir, run["run_id"])
    assert "migrate_legacy_attention" not in str(excinfo.value)

    reloaded = eval_samples.load_run(vdir, run["run_id"])
    assert reloaded["status"] == "failed"
    assert "migrate_legacy_attention" not in str(reloaded.get("error") or "")


def test_start_job_creates_project_job_and_run(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)

    with db.connection_for(isolated["db"]) as conn:
        job, run = eval_samples.start_job(
            conn,
            project,
            version,
            vdir,
            checkpoint_path="model_step100.safetensors",
        )

    assert job["kind"] == "eval_samples"
    assert job["version_id"] == version["id"]
    assert job["params_decoded"]["run_id"] == run["run_id"]
    assert len(run["items"]) == 2
    # items come from the validation set, each carrying its reference image
    assert run["items"][0]["reference_image"] == "validation/1_data/a.png"
    assert (vdir / "eval" / "samples" / run["run_id"] / "run.json").exists()


def test_eval_samples_can_store_runs_under_task_eval_root(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    eval_root = infra_paths.task_eval_dir(42)

    run = eval_samples.create_run(
        project,
        version,
        vdir,
        checkpoint_path="model_step100.safetensors",
        eval_root=eval_root,
        now=2000.0,
    )
    finished = eval_samples.run_sample_job(
        project,
        version,
        vdir,
        run["run_id"],
        generator=_fake_generator,
        eval_root=eval_root,
    )

    assert run["storage_scope"] == "task"
    assert run["eval_root"] == str(eval_root.resolve())
    assert finished["status"] == "done"
    assert (eval_root / "samples" / run["run_id"] / "run.json").exists()
    assert not (vdir / "eval" / "samples" / run["run_id"] / "run.json").exists()
    assert eval_samples.load_run(vdir, run["run_id"], eval_root)["status"] == "done"


def _make(client: TestClient) -> tuple[int, int]:
    project = client.post("/api/projects", json={"title": "Eval Sample HTTP"}).json()
    return project["id"], project["versions"][0]["id"]


def _vdir_for(pid: int, vid: int) -> Path:
    with db.connection_for() as conn:
        project = projects.get_project(conn, pid)
        version = versions.get_version(conn, vid)
    assert project and version
    return versions.version_dir(project["id"], project["slug"], version["label"])


def test_eval_samples_http_run_list_get_and_image(client: TestClient, isolated) -> None:
    pid, vid = _make(client)
    vdir = _vdir_for(pid, vid)
    ckpt = _seed_validation_and_ckpt(vdir)
    with db.connection_for() as conn:
        project = projects.get_project(conn, pid)
        version = versions.get_version(conn, vid)
    tid = _bound_task({"db": db.STUDIO_DB}, project, version)

    created = client.post(
        f"/api/projects/{pid}/versions/{vid}/eval/run",
        json={"task_id": tid, "checkpoints": [str(ckpt)]},
    )
    assert created.status_code == 200, created.text
    body = created.json()
    run_id = body["runs"][0]["run_id"]
    assert body["queued"] == 1
    assert body["jobs"][0]["kind"] == "eval_samples"

    q = f"?task_id={tid}"
    listed = client.get(f"/api/projects/{pid}/versions/{vid}/eval/samples{q}")
    assert listed.status_code == 200, listed.text
    assert listed.json()["runs"][0]["run_id"] == run_id

    got = client.get(f"/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}{q}")
    assert got.status_code == 200, got.text
    item = got.json()["run"]["items"][0]
    assert item["filename"].endswith(".png")

    eval_root = infra_paths.task_eval_dir(tid)
    image_path = eval_samples.sample_image_path(vdir, run_id, item["filename"], eval_root)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"PNG")
    image = client.get(
        f"/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}"
        f"/images/{item['filename']}{q}"
    )
    assert image.status_code == 200, image.text
    assert image.content == b"PNG"


def test_eval_samples_rejects_checkpoint_outside_output(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)

    with pytest.raises(eval_samples.EvalSamplesError, match="output"):
        eval_samples.create_run(
            project,
            version,
            vdir,
            checkpoint_path="../escape.safetensors",
        )


def _bound_task(isolated, project: dict[str, Any], version: dict[str, Any]) -> int:
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="train", config_name="fake")
        db.update_task(conn, tid, project_id=project["id"], version_id=version["id"])
    return tid


def test_run_task_eval_endpoint_queues_task_scoped(client, isolated) -> None:
    project, version, vdir = _new_project(isolated)
    ckpt = _seed_validation_and_ckpt(vdir)
    pid, vid = project["id"], version["id"]
    tid = _bound_task(isolated, project, version)

    resp = client.post(
        f"/api/projects/{pid}/versions/{vid}/eval/run",
        json={"task_id": tid, "checkpoints": [str(ckpt)]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["queued"] == 1
    assert body["runs"][0]["storage_scope"] == "task"
    assert body["jobs"][0]["params_decoded"]["task_id"] == tid


def test_run_task_eval_endpoint_validates_task_ownership(client, isolated) -> None:
    project, version, vdir = _new_project(isolated)
    ckpt = _seed_validation_and_ckpt(vdir)
    pid, vid = project["id"], version["id"]

    # 不存在的 task → 404
    missing = client.post(
        f"/api/projects/{pid}/versions/{vid}/eval/run",
        json={"task_id": 99999, "checkpoints": [str(ckpt)]},
    )
    assert missing.status_code == 404, missing.text

    # task 未绑定到该 project/version → 400
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="other", config_name="fake")
    other = client.post(
        f"/api/projects/{pid}/versions/{vid}/eval/run",
        json={"task_id": tid, "checkpoints": [str(ckpt)]},
    )
    assert other.status_code == 400, other.text

    # 空 checkpoints → 400
    tid_ok = _bound_task(isolated, project, version)
    empty = client.post(
        f"/api/projects/{pid}/versions/{vid}/eval/run",
        json={"task_id": tid_ok, "checkpoints": []},
    )
    assert empty.status_code == 400, empty.text
