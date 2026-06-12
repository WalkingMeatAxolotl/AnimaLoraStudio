"""CLIP-T / CLIP-I eval metric runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studio import db, secrets, server
from studio.services import eval_clip, eval_manifest, eval_metrics, eval_samples
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
        project = projects.create_project(conn, title="Eval CLIP")
        version = versions.create_version(
            conn, project_id=project["id"], label="baseline"
        )
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def _seed_train_and_ckpt(vdir: Path) -> None:
    train = vdir / "train" / "1_data"
    train.mkdir(parents=True, exist_ok=True)
    (train / "a.png").write_bytes(b"png-a")
    (train / "a.txt").write_text("solo, red hair", encoding="utf-8")
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
    return eval_samples.run_sample_job(
        project,
        version,
        vdir,
        run["run_id"],
        generator=_fake_generator,
    )


def _make(client: TestClient) -> tuple[int, int]:
    project = client.post("/api/projects", json={"title": "Eval CLIP HTTP"}).json()
    return project["id"], project["versions"][0]["id"]


def _vdir_for(pid: int, vid: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for() as conn:
        project = projects.get_project(conn, pid)
        version = versions.get_version(conn, vid)
    assert project and version
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def test_start_eval_clip_job_marks_clip_metrics_pending(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)

    with db.connection_for(isolated["db"]) as conn:
        job, result = eval_clip.start_job(
            conn,
            project,
            version,
            vdir,
            run["run_id"],
            model_name="mock/clip",
        )

    assert job["kind"] == "eval_clip"
    assert job["version_id"] == version["id"]
    assert job["params_decoded"]["run_id"] == run["run_id"]
    assert job["params_decoded"]["model_name"] == "mock/clip"
    assert result["status"] == "running"
    assert result["metric_states"]["clip_t"]["status"] == "pending"
    assert result["metric_states"]["clip_i"]["status"] == "pending"
    assert result["metric_states"]["dino_i"]["status"] == "not_run"
    assert result["metrics"] == {}


def test_run_clip_job_with_fake_scorer_preserves_existing_metrics(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metrics": {"paired_cmmd2": 0.42}},
        now=3000.0,
    )

    def scorer(_run, version_dir: Path, model_name: str, _progress):
        cache = eval_metrics.ensure_embeddings_cache_dir(version_dir) / "clip"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "fake.npy").write_bytes(b"cache")
        return {
            "model_name": model_name,
            "clip_t": 0.75,
            "clip_i": 0.62,
            "clip_t_count": 1,
            "clip_i_count": 1,
        }

    result = eval_clip.run_clip_job(
        project,
        version,
        vdir,
        run["run_id"],
        scorer=scorer,
        model_name="mock/clip",
    )

    assert result["status"] == "partial"
    assert result["metrics"]["clip_t"] == 0.75
    assert result["metrics"]["clip_i"] == 0.62
    assert result["metrics"]["paired_cmmd2"] == 0.42
    assert result["metric_states"]["clip_t"]["status"] == "done"
    assert result["metric_states"]["clip_t"]["count"] == 1
    assert result["metric_states"]["clip_i"]["model_name"] == "mock/clip"
    assert result["cache"]["entries"] == [{
        "key": "clip",
        "path": "eval/cache/embeddings/clip",
        "file_count": 1,
        "size_bytes": 5,
    }]


def test_run_clip_job_clears_stale_clip_value_when_unavailable(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metrics": {"clip_i": 0.5}},
        now=3000.0,
    )

    result = eval_clip.run_clip_job(
        project,
        version,
        vdir,
        run["run_id"],
        scorer=lambda _run, _vdir, _model, _progress: {"clip_t": 0.7},
    )

    assert result["metrics"]["clip_t"] == 0.7
    assert "clip_i" not in result["metrics"]
    assert result["metric_states"]["clip_i"]["status"] == "unavailable"
    assert result["metric_states"]["clip_i"]["value"] is None


def test_feature_tensor_accepts_model_output_pooler() -> None:
    class FakeTensor:
        projected = False
        shape = (1, 768)

        def float(self):
            return self

    class FakeOutput:
        def __init__(self):
            self.pooler_output = FakeTensor()

    output = FakeOutput()

    def projection(tensor):
        tensor.projected = True
        return tensor

    got = eval_clip._feature_tensor(output, projection=projection)

    assert got is output.pooler_output
    assert got.projected is True


def test_feature_tensor_skips_mismatched_projection() -> None:
    class FakeTensor:
        projected = False
        shape = (1, 512)

        def float(self):
            return self

    class FakeOutput:
        def __init__(self):
            self.pooler_output = FakeTensor()

    class FakeProjection:
        in_features = 768

        def __call__(self, tensor):
            tensor.projected = True
            return tensor

    output = FakeOutput()
    got = eval_clip._feature_tensor(output, projection=FakeProjection())

    assert got is output.pooler_output
    assert got.projected is False


def test_eval_clip_http_start_queues_job(client: TestClient) -> None:
    pid, vid = _make(client)
    project, version, vdir = _vdir_for(pid, vid)
    run = _sample_run(project, version, vdir)

    started = client.post(
        f"/api/projects/{pid}/versions/{vid}/eval/samples/{run['run_id']}/metrics/clip",
        json={"model_name": "mock/clip"},
    )

    assert started.status_code == 200, started.text
    body = started.json()
    assert body["job"]["kind"] == "eval_clip"
    assert body["result"]["metric_states"]["clip_t"]["status"] == "pending"
    assert body["result"]["metric_states"]["clip_i"]["status"] == "pending"


def test_eval_clip_http_start_uses_saved_default_model(client: TestClient) -> None:
    pid, vid = _make(client)
    project, version, vdir = _vdir_for(pid, vid)
    run = _sample_run(project, version, vdir)
    secrets.update({"eval_metrics": {"clip_model_name": "/models/local-clip"}})

    started = client.post(
        f"/api/projects/{pid}/versions/{vid}/eval/samples/{run['run_id']}/metrics/clip",
        json={},
    )

    assert started.status_code == 200, started.text
    body = started.json()
    assert body["job"]["params_decoded"]["model_name"] == "/models/local-clip"
    assert (
        body["result"]["metric_states"]["clip_t"]["model_name"]
        == "/models/local-clip"
    )


def test_eval_clip_job_kind_is_schedulable_and_gpu_bound(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Eval CLIP Job")
        job = project_jobs.create_job(
            conn,
            project_id=project["id"],
            kind="eval_clip",
            params={"run_id": "run-1"},
        )

    assert job["kind"] == "eval_clip"
    assert "eval_clip" in GPU_BOUND_JOB_KINDS
