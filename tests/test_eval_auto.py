"""Automatic training-checkpoint eval POC glue."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studio import db, secrets
from studio.infrastructure import paths as infra_paths
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
    monkeypatch.setattr(infra_paths, "TASKS_DIR", tmp_path / "tasks")
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


def _enable_validation(project: dict[str, Any], version: dict[str, Any]) -> None:
    """Opt this version into post-training validation metrics (training config)."""
    from studio.services import version_config
    version_config.write_version_config(
        project, version, {"eval_validation_enabled": True}
    )


def _fake_generator(run: dict[str, Any], version_dir: Path, progress) -> None:
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


def _make_sample_run(conn, project, version, vdir, task_id: int, ckpt: str):
    """建一个 task-scoped 评估 run（替代已删的 queue_checkpoint_eval setup）。"""
    return eval_samples.start_job(
        conn, project, version, vdir,
        checkpoint_path=ckpt,
        auto_metrics=True,
        auto_source={"task_id": task_id},
        eval_root=infra_paths.task_eval_dir(task_id),
    )


def test_queue_metric_jobs_for_sample_uses_saved_defaults(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    _enable_validation(project, version)
    secrets.update({
        "eval_metrics": {
            "clip_model_name": "/models/clip",
            "dino_model_name": "/models/dino",
        }
    })
    with db.connection_for(isolated["db"]) as conn:
        _sample_job, run = _make_sample_run(
            conn, project, version, vdir, 7,
            str(vdir / "output" / "model_epoch2.safetensors"),
        )
        eval_root = infra_paths.task_eval_dir(7)
        run = eval_samples.run_sample_job(
            project,
            version,
            vdir,
            run["run_id"],
            generator=_fake_generator,
            eval_root=eval_root,
        )
        jobs = eval_auto.queue_metric_jobs_for_sample(
            conn,
            project,
            version,
            vdir,
            run["run_id"],
            eval_root=eval_root,
            task_id=7,
        )

    assert [j["kind"] for j in jobs] == ["eval_clip", "eval_dino"]
    assert jobs[0]["params_decoded"]["model_name"] == "/models/clip"
    assert jobs[1]["params_decoded"]["model_name"] == "/models/dino"
    assert jobs[0]["params_decoded"]["task_id"] == 7
    assert jobs[1]["params_decoded"]["task_id"] == 7


def test_queue_metric_jobs_for_sample_respects_enabled_metrics(isolated) -> None:
    """Settings 关掉 dino_i → 只排 eval_clip，不排 eval_dino（registry 门控）。"""
    project, version, vdir = _project_version(isolated)
    _enable_validation(project, version)
    secrets.update({
        "eval_metrics": {
            "enabled_metrics": ["clip_t", "clip_i"],
        }
    })
    with db.connection_for(isolated["db"]) as conn:
        _sample_job, run = _make_sample_run(
            conn, project, version, vdir, 8,
            str(vdir / "output" / "model_epoch2.safetensors"),
        )
        eval_root = infra_paths.task_eval_dir(8)
        run = eval_samples.run_sample_job(
            project, version, vdir, run["run_id"],
            generator=_fake_generator, eval_root=eval_root,
        )
        jobs = eval_auto.queue_metric_jobs_for_sample(
            conn, project, version, vdir, run["run_id"],
            eval_root=eval_root, task_id=8,
        )
    assert [j["kind"] for j in jobs] == ["eval_clip"]


def test_queue_metric_jobs_for_sample_reuses_active_jobs(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    _enable_validation(project, version)
    secrets.update({
        "eval_metrics": {
            "clip_model_name": "/models/clip",
            "dino_model_name": "/models/dino",
        }
    })
    with db.connection_for(isolated["db"]) as conn:
        _sample_job, run = _make_sample_run(
            conn, project, version, vdir, 7,
            str(vdir / "output" / "model_epoch2.safetensors"),
        )
        eval_root = infra_paths.task_eval_dir(7)
        run = eval_samples.run_sample_job(
            project,
            version,
            vdir,
            run["run_id"],
            generator=_fake_generator,
            eval_root=eval_root,
        )
        first = eval_auto.queue_metric_jobs_for_sample(
            conn,
            project,
            version,
            vdir,
            run["run_id"],
            eval_root=eval_root,
            task_id=7,
        )
        second = eval_auto.queue_metric_jobs_for_sample(
            conn,
            project,
            version,
            vdir,
            run["run_id"],
            eval_root=eval_root,
            task_id=7,
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
    _enable_validation(project, version)
    task = {"id": 7, "project_id": project["id"], "version_id": version["id"]}
    payload = {
        "checkpoint_path": str(vdir / "output" / "model_epoch2.safetensors"),
        "epoch": 4,
        "step": 40,
    }

    with db.connection_for(isolated["db"]) as conn:
        queued = eval_auto.queue_training_finished_eval(conn, task, payload)
        jobs = project_jobs.list_jobs(conn, kind="eval_samples", status="pending")

    # 2 个 checkpoint + 1 个 baseline（纯底模对照）= 3
    assert len(queued) == 3
    assert len(jobs) == 3
    assert sum(1 for _job, r in queued if r.get("baseline")) == 1
    paths = {job["params_decoded"]["checkpoint_path"] for job in jobs}
    assert paths == {
        "output/model_epoch2.safetensors",
        "output/model_epoch4.safetensors",
    }
    assert all(job["params_decoded"]["auto_metrics"] is True for job in jobs)
    assert all(job["params_decoded"]["task_id"] == 7 for job in jobs)
    assert all(
        job["params_decoded"]["auto_source"]["trigger"] == "after_training"
        for job in jobs
    )


def test_after_training_eval_skips_baseline_when_disabled(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    (vdir / "output" / "model_epoch4.safetensors").write_bytes(b"lora")
    _enable_validation(project, version)
    secrets.update({"eval_metrics": {"eval_baseline_enabled": False}})
    task = {"id": 9, "project_id": project["id"], "version_id": version["id"]}
    payload = {"checkpoint_path": str(vdir / "output" / "model_epoch2.safetensors")}
    with db.connection_for(isolated["db"]) as conn:
        queued = eval_auto.queue_training_finished_eval(conn, task, payload)
    assert queued and not any(r.get("baseline") for _job, r in queued)


def test_queue_manual_task_eval_bypasses_switch_and_scopes_to_task(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    (vdir / "output" / "model_epoch4.safetensors").write_bytes(b"lora")
    task = {"id": 7, "project_id": project["id"], "version_id": version["id"]}
    ck2 = str(vdir / "output" / "model_epoch2.safetensors")
    ck4 = str(vdir / "output" / "model_epoch4.safetensors")

    # 手动入口无视 per-version 开关照样排队，且写 task-scoped。
    with db.connection_for(isolated["db"]) as conn:
        queued = eval_auto.queue_manual_task_eval(conn, task, [ck2, ck4])
        jobs = project_jobs.list_jobs(conn, kind="eval_samples", status="pending")

    # 2 个 checkpoint + 1 个 baseline（手动入口现在也排 baseline，给 Δ）
    assert len(queued) == 3
    assert len(jobs) == 3
    assert sum(1 for _job, r in queued if r.get("baseline")) == 1
    paths = {job["params_decoded"]["checkpoint_path"] for job in jobs}
    assert paths == {
        "output/model_epoch2.safetensors",
        "output/model_epoch4.safetensors",
    }
    assert all(job["params_decoded"]["task_id"] == 7 for job in jobs)
    assert all(job["params_decoded"]["auto_metrics"] is True for job in jobs)
    assert all(
        job["params_decoded"]["auto_source"]["trigger"] == "manual" for job in jobs
    )
    assert all(run["storage_scope"] == "task" for _job, run in queued)


def test_queue_manual_task_eval_dedupes_and_skips_invalid(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    task = {"id": 9, "project_id": project["id"], "version_id": version["id"]}
    ck2 = str(vdir / "output" / "model_epoch2.safetensors")

    with db.connection_for(isolated["db"]) as conn:
        # 同一 ckpt 传两次去重；output/ 外的路径被丢弃。
        queued = eval_auto.queue_manual_task_eval(
            conn, task, [ck2, ck2, "/etc/passwd"]
        )
        jobs = project_jobs.list_jobs(conn, kind="eval_samples", status="pending")

    # 1 个 checkpoint + 1 个 baseline
    assert len(jobs) == 2
    assert sum(1 for _job, r in queued if r.get("baseline")) == 1
    real = [r for _job, r in queued if not r.get("baseline")]
    assert len(real) == 1
    assert real[0]["checkpoint"]["path"] == "output/model_epoch2.safetensors"


def test_supervisor_eval_training_finished_queues_after_task_done(isolated) -> None:
    import json

    project, version, _vdir = _project_version(isolated)
    _enable_validation(project, version)
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
    # 1 个 checkpoint + 1 个 baseline
    assert len(jobs) == 2
    assert all(j["params_decoded"]["auto_source"]["trigger"] == "after_training" for j in jobs)
    assert all(j["params_decoded"]["task_id"] == tid for j in jobs)
    assert sum(1 for j in jobs if j["params_decoded"]["auto_source"].get("baseline")) == 1
    queued = [e for e in events if e["type"] == "eval_auto_after_training_queued"]
    assert queued and queued[0]["count"] == 2


def test_eval_runs_full_validation_set_no_cap(isolated) -> None:
    """评估覆盖整个 validation set（无 max_items 截断）：每张图一个 item、各带 reference。"""
    project, version, vdir = _project_version(isolated)
    val = vdir / "validation" / "1_data"
    val.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (val / f"v{i}.png").write_bytes(b"png")
        (val / f"v{i}.txt").write_text("solo", encoding="utf-8")

    run = eval_samples.create_run(
        project,
        version,
        vdir,
        checkpoint_path=str(vdir / "output" / "model_epoch2.safetensors"),
    )

    assert run["summary"]["total"] == 3
    assert all(item["reference_image"] for item in run["items"])
