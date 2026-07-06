"""DINO-I eval metric runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studio import db, secrets, server
from studio.services import eval_dino, eval_metrics, eval_samples
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.supervisor import GPU_BOUND_JOB_KINDS


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
    server.app.state.supervisor = None
    return TestClient(server.app)


def _new_project(isolated) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Eval DINO")
        version = versions.create_version(
            conn, project_id=project["id"], label="baseline"
        )
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def _seed_validation_and_ckpt(vdir: Path) -> None:
    val = vdir / "validation" / "1_data"
    val.mkdir(parents=True, exist_ok=True)
    (val / "a.png").write_bytes(b"png-a")
    (val / "a.txt").write_text("solo, red hair", encoding="utf-8")
    output = vdir / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_step100.safetensors").write_bytes(b"fake-lora")


def _fake_generator(run: dict[str, Any], version_dir: Path, progress) -> None:
    for idx, item in enumerate(run["items"]):
        progress(f"fake {idx}")
        run = eval_samples.mark_item_running(version_dir, run, idx)
        path = eval_samples.sample_image_path(
            version_dir, run["run_id"], item["filename"]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG")
        run = eval_samples.mark_item_done(version_dir, run, idx)


def _sample_run(
    project: dict[str, Any], version: dict[str, Any], vdir: Path
) -> dict[str, Any]:
    _seed_validation_and_ckpt(vdir)
    run = eval_samples.create_run(
        project,
        version,
        vdir,
        checkpoint_path="model_step100.safetensors",
        now=2000.0,
    )
    return eval_samples.run_sample_job(
        project,
        version,
        vdir,
        run["run_id"],
        generator=_fake_generator,
    )


def _make(client: TestClient) -> tuple[int, int]:
    project = client.post("/api/projects", json={"title": "Eval DINO HTTP"}).json()
    return project["id"], project["versions"][0]["id"]


def _vdir_for(pid: int, vid: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for() as conn:
        project = projects.get_project(conn, pid)
        version = versions.get_version(conn, vid)
    assert project and version
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def test_start_eval_dino_job_marks_dino_metric_pending(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)

    with db.connection_for(isolated["db"]) as conn:
        job, result = eval_dino.start_job(
            conn,
            project,
            version,
            vdir,
            run["run_id"],
            model_name="mock/dino",
        )

    assert job["kind"] == "eval_dino"
    assert job["version_id"] == version["id"]
    assert job["params_decoded"]["run_id"] == run["run_id"]
    assert job["params_decoded"]["model_name"] == "mock/dino"
    assert result["status"] == "running"
    assert result["metric_states"]["dino_i"]["status"] == "pending"
    assert result["metric_states"]["clip_t"]["status"] == "not_run"
    assert result["metrics"] == {}


def test_run_dino_job_with_fake_scorer_preserves_existing_metrics(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metrics": {"clip_t": 0.72}},
        now=3000.0,
    )

    def scorer(_run, version_dir: Path, model_name: str, _progress):
        cache = eval_metrics.ensure_embeddings_cache_dir(version_dir) / "dino"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "fake.npy").write_bytes(b"cache")
        return {
            "model_name": model_name,
            "dino_i": 0.81,
            "dino_i_count": 1,
        }

    result = eval_dino.run_dino_job(
        project,
        version,
        vdir,
        run["run_id"],
        scorer=scorer,
        model_name="mock/dino",
    )

    assert result["status"] == "partial"
    assert result["metrics"]["clip_t"] == 0.72
    assert result["metrics"]["dino_i"] == 0.81
    assert result["metric_states"]["dino_i"]["status"] == "done"
    assert result["metric_states"]["dino_i"]["count"] == 1
    assert result["metric_states"]["dino_i"]["model_name"] == "mock/dino"
    assert result["cache"]["entries"] == [{
        "key": "dino",
        "path": "eval/cache/embeddings/dino",
        "file_count": 1,
        "size_bytes": 5,
    }]


def test_run_dino_job_clears_stale_dino_value_when_unavailable(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metrics": {"dino_i": 0.5}},
        now=3000.0,
    )

    result = eval_dino.run_dino_job(
        project,
        version,
        vdir,
        run["run_id"],
        scorer=lambda _run, _vdir, _model, _progress: {"dino_i": None},
    )

    assert "dino_i" not in result["metrics"]
    assert result["metric_states"]["dino_i"]["status"] == "unavailable"
    assert result["metric_states"]["dino_i"]["value"] is None


def test_feature_tensor_uses_cls_token_when_pooler_missing() -> None:
    class FakeHidden:
        def __init__(self):
            self.key = "hidden"

        def __getitem__(self, item):
            return ("slice", item)

    class FakeOutput:
        def __init__(self):
            self.last_hidden_state = FakeHidden()

    got = eval_dino._feature_tensor(FakeOutput())

    assert got == ("slice", (slice(None, None, None), 0))


def test_eval_dino_http_start_queues_job(client: TestClient) -> None:
    pid, vid = _make(client)
    project, version, vdir = _vdir_for(pid, vid)
    run = _sample_run(project, version, vdir)

    started = client.post(
        f"/api/projects/{pid}/versions/{vid}/eval/samples/{run['run_id']}/metrics/dino",
        json={"model_name": "mock/dino"},
    )

    assert started.status_code == 200, started.text
    body = started.json()
    assert body["job"]["kind"] == "eval_dino"
    assert body["result"]["metric_states"]["dino_i"]["status"] == "pending"


def test_eval_dino_http_start_uses_saved_default_model(client: TestClient) -> None:
    pid, vid = _make(client)
    project, version, vdir = _vdir_for(pid, vid)
    run = _sample_run(project, version, vdir)
    secrets.update({"eval_metrics": {"dino_model_name": "/models/local-dino"}})

    started = client.post(
        f"/api/projects/{pid}/versions/{vid}/eval/samples/{run['run_id']}/metrics/dino",
        json={},
    )

    assert started.status_code == 200, started.text
    body = started.json()
    assert body["job"]["params_decoded"]["model_name"] == "/models/local-dino"
    assert body["result"]["metric_states"]["dino_i"]["model_name"] == "/models/local-dino"


def test_eval_dino_job_kind_is_schedulable_and_gpu_bound(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Eval DINO Job")
        job = project_jobs.create_job(
            conn,
            project_id=project["id"],
            kind="eval_dino",
            params={"run_id": "run-1"},
        )

    assert job["kind"] == "eval_dino"
    assert "eval_dino" in GPU_BOUND_JOB_KINDS
