"""LoRA eval metric result contract and endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.services import eval_manifest, eval_metrics, eval_samples
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
        project = projects.create_project(conn, title="Eval Metrics")
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


def _sample_run(
    project: dict[str, Any], version: dict[str, Any], vdir: Path
) -> dict[str, Any]:
    _seed_train_and_ckpt(vdir)
    eval_manifest.save_default_manifest(project, version, vdir, now=1000.0)
    return eval_samples.create_run(
        project,
        version,
        vdir,
        checkpoint_path="model_step100.safetensors",
        max_items=1,
        now=2000.0,
    )


def _make(client: TestClient) -> tuple[int, int]:
    project = client.post("/api/projects", json={"title": "Eval Metrics HTTP"}).json()
    return project["id"], project["versions"][0]["id"]


def _vdir_for(pid: int, vid: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for() as conn:
        project = projects.get_project(conn, pid)
        version = versions.get_version(conn, vid)
    assert project and version
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def test_empty_metric_result_describes_not_run_states(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)

    result = eval_metrics.load_result(vdir, run["run_id"])

    assert result is not None
    assert result["has_metrics"] is False
    assert result["status"] == "empty"
    assert result["metrics"] == {}
    assert result["summary"]["not_run"] == 6
    assert result["metric_states"]["clip_t"]["status"] == "not_run"
    assert result["metric_states"]["clip_t"]["question"]
    assert result["metric_states"]["sscd_nn"]["higher_is_better"] is False
    assert result["cache"]["embeddings_dir"] == "eval/cache/embeddings"


def test_save_metric_result_normalizes_states_and_preserves_created_at(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)

    first = eval_metrics.save_result(
        vdir,
        run["run_id"],
        {
            "metrics": {
                "clip_t": {"value": 0.31},
                "paired_cmmd2": 0.42,
            }
        },
        now=3000.0,
    )
    second = eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metric_states": {"clip_t": {"status": "failed", "error": "missing model"}}},
        now=4000.0,
    )

    assert first["has_metrics"] is True
    assert first["status"] == "partial"
    assert first["metric_states"]["clip_t"]["status"] == "done"
    assert first["metric_states"]["clip_t"]["value"] == 0.31
    assert first["metric_states"]["paired_cmmd2"]["status"] == "done"
    assert second["created_at"] == 3000.0
    assert second["updated_at"] == 4000.0
    assert second["status"] == "failed"
    assert second["metrics"]["clip_t"]["value"] == 0.31
    assert second["metrics"]["paired_cmmd2"] == 0.42
    assert second["metric_states"]["clip_t"]["error"] == "missing model"


def test_save_metric_result_can_clear_stale_values(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metrics": {"clip_i": 0.5}},
        now=3000.0,
    )

    result = eval_metrics.save_result(
        vdir,
        run["run_id"],
        {
            "metrics": {"clip_i": None},
            "metric_states": {
                "clip_i": {
                    "status": "unavailable",
                    "value": None,
                    "reason": "no paired references",
                }
            },
        },
        now=4000.0,
    )

    assert "clip_i" not in result["metrics"]
    assert result["status"] == "partial"
    assert result["metric_states"]["clip_i"]["status"] == "unavailable"
    assert result["metric_states"]["clip_i"]["value"] is None


def test_eval_metrics_http_empty_list_and_single_run(client: TestClient) -> None:
    pid, vid = _make(client)
    project, version, vdir = _vdir_for(pid, vid)
    run = _sample_run(project, version, vdir)

    listed = client.get(f"/api/projects/{pid}/versions/{vid}/eval/metrics")
    assert listed.status_code == 200, listed.text
    listed_body = listed.json()
    assert [spec["key"] for spec in listed_body["metric_specs"]] == [
        "clip_t",
        "clip_i",
        "dino_i",
        "diversity",
        "sscd_nn",
        "paired_cmmd2",
    ]
    assert listed_body["results"][0]["run_id"] == run["run_id"]
    assert listed_body["results"][0]["status"] == "empty"
    assert listed_body["cache"]["embeddings_dir"] == "eval/cache/embeddings"

    got = client.get(
        f"/api/projects/{pid}/versions/{vid}/eval/samples/{run['run_id']}/metrics"
    )
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["result"]["has_metrics"] is False
    assert body["result"]["sample_run"]["summary"]["total"] == 1


def test_eval_metrics_http_reads_saved_result(client: TestClient) -> None:
    pid, vid = _make(client)
    project, version, vdir = _vdir_for(pid, vid)
    run = _sample_run(project, version, vdir)
    eval_metrics.ensure_embeddings_cache_dir(vdir)
    (vdir / "eval" / "cache" / "embeddings" / "clip").mkdir(parents=True)
    (vdir / "eval" / "cache" / "embeddings" / "clip" / "real.npy").write_bytes(
        b"cache"
    )
    eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metrics": {"clip_t": 0.75}},
        now=3000.0,
    )

    got = client.get(
        f"/api/projects/{pid}/versions/{vid}/eval/samples/{run['run_id']}/metrics"
    )

    assert got.status_code == 200, got.text
    result = got.json()["result"]
    assert result["has_metrics"] is True
    assert result["status"] == "partial"
    assert result["metric_states"]["clip_t"]["value"] == 0.75
    assert result["cache"]["entries"] == [{
        "key": "clip",
        "path": "eval/cache/embeddings/clip",
        "file_count": 1,
        "size_bytes": 5,
    }]
