"""Automatic training-checkpoint eval POC glue."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studio import db, secrets
from studio.services import eval_auto, eval_samples
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.supervisor import Supervisor
from studio.supervisor.slot import _Slot


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


def _project_version(isolated) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Auto Eval")
        version = versions.create_version(
            conn, project_id=project["id"], label="baseline"
        )
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    train = vdir / "train" / "1_data"
    train.mkdir(parents=True, exist_ok=True)
    (train / "a.png").write_bytes(b"png")
    (train / "a.txt").write_text("solo", encoding="utf-8")
    output = vdir / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_epoch2.safetensors").write_bytes(b"lora")
    return project, version, vdir


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


def test_queue_checkpoint_eval_respects_switch(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    task = {"id": 7, "project_id": project["id"], "version_id": version["id"]}
    payload = {
        "checkpoint_path": str(vdir / "output" / "model_epoch2.safetensors"),
        "epoch": 2,
        "step": 20,
        "trigger": "epoch",
    }

    with db.connection_for(isolated["db"]) as conn:
        assert eval_auto.queue_checkpoint_eval(conn, task, payload) is None

    secrets.update({
        "eval_metrics": {
            "auto_eval_on_checkpoint": True,
            "auto_eval_trigger": "checkpoint",
            "auto_eval_max_items": 1,
        }
    })
    with db.connection_for(isolated["db"]) as conn:
        queued = eval_auto.queue_checkpoint_eval(conn, task, payload)

    assert queued is not None
    job, run = queued
    assert job["kind"] == "eval_samples"
    assert job["params_decoded"]["auto_metrics"] is True
    assert job["params_decoded"]["auto_source"] == {
        "task_id": 7,
        "epoch": 2,
        "step": 20,
        "trigger": "epoch",
    }
    assert job["params_decoded"]["checkpoint_path"] == "output/model_epoch2.safetensors"
    assert run["checkpoint"]["path"] == "output/model_epoch2.safetensors"
    assert run["auto_metrics"] is True
    assert run["auto_source"] == {
        "task_id": 7,
        "epoch": 2,
        "step": 20,
        "trigger": "epoch",
    }
    assert run["summary"]["total"] == 1


def test_queue_metric_jobs_for_sample_uses_saved_defaults(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    secrets.update({
        "eval_metrics": {
            "clip_model_name": "/models/clip",
            "dino_model_name": "/models/dino",
        }
    })
    with db.connection_for(isolated["db"]) as conn:
        sample_job, run = eval_auto.queue_checkpoint_eval(conn, {
            "id": 7,
            "project_id": project["id"],
            "version_id": version["id"],
        }, {
            "checkpoint_path": str(vdir / "output" / "model_epoch2.safetensors"),
        }) or (None, None)
    assert sample_job is None

    secrets.update({
        "eval_metrics": {
            "auto_eval_on_checkpoint": True,
            "auto_eval_trigger": "checkpoint",
            "clip_model_name": "/models/clip",
            "dino_model_name": "/models/dino",
        }
    })
    with db.connection_for(isolated["db"]) as conn:
        _sample_job, run = eval_auto.queue_checkpoint_eval(conn, {
            "id": 7,
            "project_id": project["id"],
            "version_id": version["id"],
        }, {
            "checkpoint_path": str(vdir / "output" / "model_epoch2.safetensors"),
        })
        run = eval_samples.run_sample_job(
            project,
            version,
            vdir,
            run["run_id"],
            generator=_fake_generator,
        )
        jobs = eval_auto.queue_metric_jobs_for_sample(
            conn, project, version, vdir, run["run_id"]
        )

    assert [j["kind"] for j in jobs] == ["eval_clip", "eval_dino"]
    assert jobs[0]["params_decoded"]["model_name"] == "/models/clip"
    assert jobs[1]["params_decoded"]["model_name"] == "/models/dino"


def test_queue_metric_jobs_for_sample_reuses_active_jobs(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    secrets.update({
        "eval_metrics": {
            "auto_eval_on_checkpoint": True,
            "auto_eval_trigger": "checkpoint",
            "clip_model_name": "/models/clip",
            "dino_model_name": "/models/dino",
        }
    })
    with db.connection_for(isolated["db"]) as conn:
        _sample_job, run = eval_auto.queue_checkpoint_eval(conn, {
            "id": 7,
            "project_id": project["id"],
            "version_id": version["id"],
        }, {
            "checkpoint_path": str(vdir / "output" / "model_epoch2.safetensors"),
        })
        run = eval_samples.run_sample_job(
            project,
            version,
            vdir,
            run["run_id"],
            generator=_fake_generator,
        )
        first = eval_auto.queue_metric_jobs_for_sample(
            conn, project, version, vdir, run["run_id"]
        )
        second = eval_auto.queue_metric_jobs_for_sample(
            conn, project, version, vdir, run["run_id"]
        )
        clip_jobs = project_jobs.list_jobs(conn, kind="eval_clip")
        dino_jobs = project_jobs.list_jobs(conn, kind="eval_dino")

    assert [job["id"] for job in second] == [job["id"] for job in first]
    assert len(clip_jobs) == 1
    assert len(dino_jobs) == 1
    assert clip_jobs[0]["params_decoded"]["model_name"] == "/models/clip"
    assert dino_jobs[0]["params_decoded"]["model_name"] == "/models/dino"


def test_after_training_eval_queues_all_checkpoints_by_default(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    (vdir / "output" / "model_epoch4.safetensors").write_bytes(b"lora")
    task = {"id": 7, "project_id": project["id"], "version_id": version["id"]}
    payload = {
        "checkpoint_path": str(vdir / "output" / "model_epoch2.safetensors"),
        "epoch": 4,
        "step": 40,
    }
    secrets.update({
        "eval_metrics": {
            "auto_eval_on_checkpoint": True,
            "auto_eval_max_items": 1,
        }
    })

    with db.connection_for(isolated["db"]) as conn:
        assert eval_auto.queue_checkpoint_eval(conn, task, payload) is None
        queued = eval_auto.queue_training_finished_eval(conn, task, payload)
        jobs = project_jobs.list_jobs(conn, kind="eval_samples", status="pending")

    assert len(queued) == 2
    assert len(jobs) == 2
    paths = {job["params_decoded"]["checkpoint_path"] for job in jobs}
    assert paths == {
        "output/model_epoch2.safetensors",
        "output/model_epoch4.safetensors",
    }
    assert all(job["params_decoded"]["auto_metrics"] is True for job in jobs)
    assert all(
        job["params_decoded"]["auto_source"]["trigger"] == "after_training"
        for job in jobs
    )


def test_run_checkpoint_eval_for_task_runs_inline_without_queue_jobs(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    secrets.update({
        "eval_metrics": {
            "auto_eval_on_checkpoint": True,
            "auto_eval_trigger": "checkpoint",
            "auto_eval_max_items": 1,
            "clip_model_name": "/models/clip",
            "dino_model_name": "/models/dino",
        }
    })
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="train", config_name="fake")
        db.update_task(
            conn,
            tid,
            project_id=project["id"],
            version_id=version["id"],
        )

    result = eval_auto.run_checkpoint_eval_for_task(
        tid,
        {
            "checkpoint_path": str(vdir / "output" / "model_epoch2.safetensors"),
            "epoch": 2,
            "step": 20,
            "trigger": "epoch",
        },
        sample_generator=_fake_generator,
        clip_scorer=lambda _run, _vdir, model, _progress: {
            "clip_t": 0.11,
            "clip_i": 0.22,
            "clip_t_count": 1,
            "clip_i_count": 1,
            "model_name": model,
        },
        dino_scorer=lambda _run, _vdir, model, _progress: {
            "dino_i": 0.33,
            "dino_i_count": 1,
            "model_name": model,
        },
    )

    assert result is not None
    run = result["run"]
    assert run["status"] == "done"
    assert run["auto_source"]["inline"] is True
    assert result["metrics"]["clip"]["metric_states"]["clip_t"]["model_name"] == "/models/clip"
    assert result["metrics"]["clip"]["metric_states"]["clip_i"]["value"] == 0.22
    assert result["metrics"]["dino"]["metric_states"]["dino_i"]["model_name"] == "/models/dino"
    assert result["metrics"]["dino"]["metric_states"]["dino_i"]["value"] == 0.33
    with db.connection_for(isolated["db"]) as conn:
        assert project_jobs.list_jobs(conn) == []


def test_supervisor_eval_checkpoint_event_queues_sample_job(isolated) -> None:
    import json

    project, version, vdir = _project_version(isolated)
    secrets.update({
        "eval_metrics": {
            "auto_eval_on_checkpoint": True,
            "auto_eval_trigger": "checkpoint",
            "auto_eval_max_items": 1,
        }
    })
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="train", config_name="fake")
        db.update_task(
            conn,
            tid,
            project_id=project["id"],
            version_id=version["id"],
        )

    events: list[dict[str, Any]] = []
    sup = Supervisor(on_event=events.append, db_path=isolated["db"])
    callback = sup._make_task_log_callback(_Slot(name="train", kind="task", id=tid), tid)
    callback(
        "__EVENT__:eval_checkpoint_saved:"
        + json.dumps({
            "checkpoint_path": str(vdir / "output" / "model_epoch2.safetensors"),
            "epoch": 2,
            "step": 20,
            "trigger": "epoch",
        })
    )

    queued = [e for e in events if e["type"] == "eval_auto_sample_queued"]
    assert queued
    assert queued[0]["task_id"] == tid
    assert queued[0]["run_id"]
    with db.connection_for(isolated["db"]) as conn:
        jobs = project_jobs.list_jobs(conn, kind="eval_samples", status="pending")
    assert len(jobs) == 1
    assert jobs[0]["params_decoded"]["auto_metrics"] is True


def test_supervisor_eval_training_finished_queues_after_task_done(isolated) -> None:
    import json

    project, version, _vdir = _project_version(isolated)
    secrets.update({
        "eval_metrics": {
            "auto_eval_on_checkpoint": True,
            "auto_eval_max_items": 1,
        }
    })
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="train", config_name="fake")
        db.update_task(
            conn,
            tid,
            project_id=project["id"],
            version_id=version["id"],
            status="running",
        )

    events: list[dict[str, Any]] = []
    sup = Supervisor(on_event=events.append, db_path=isolated["db"])
    slot = _Slot(name="train", kind="task", id=tid)
    callback = sup._make_task_log_callback(slot, tid)
    callback(
        "__EVENT__:eval_training_finished:"
        + json.dumps({
            "epoch": 2,
            "step": 20,
        })
    )

    with db.connection_for(isolated["db"]) as conn:
        assert project_jobs.list_jobs(conn, kind="eval_samples") == []

    sup._finish_slot(slot, 0)

    with db.connection_for(isolated["db"]) as conn:
        jobs = project_jobs.list_jobs(conn, kind="eval_samples", status="pending")
    assert len(jobs) == 1
    assert jobs[0]["params_decoded"]["auto_source"]["trigger"] == "after_training"
    queued = [e for e in events if e["type"] == "eval_auto_after_training_queued"]
    assert queued and queued[0]["count"] == 1
