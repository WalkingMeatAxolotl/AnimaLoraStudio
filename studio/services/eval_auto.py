"""POC glue for automatically evaluating saved training checkpoints."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from studio import db
from studio import secrets
from studio.services import eval_clip, eval_dino, eval_samples
from studio.services.projects import jobs as project_jobs, projects, versions

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


def queue_checkpoint_eval(
    conn,
    task: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Queue eval_samples for a saved LoRA checkpoint when the POC switch is on."""
    cfg = secrets.load().eval_metrics
    if not cfg.auto_eval_on_checkpoint:
        return None
    if cfg.auto_eval_trigger != "checkpoint":
        return None

    project_id = int(task.get("project_id") or 0)
    version_id = int(task.get("version_id") or 0)
    if not project_id or not version_id:
        return None

    project = projects.get_project(conn, project_id)
    version = versions.get_version(conn, version_id)
    if not version or int(version["project_id"]) != project_id:
        return None
    if not project:
        return None

    checkpoint = _checkpoint_relative_to_output(
        project_id,
        str(project.get("slug") or ""),
        version,
        str(payload.get("checkpoint_path") or ""),
    )
    if not checkpoint:
        return None

    vdir = versions.version_dir(project_id, project["slug"], str(version["label"]))
    try:
        job, run = eval_samples.start_job(
            conn,
            project,
            version,
            vdir,
            checkpoint_path=checkpoint,
            max_items=cfg.auto_eval_max_items,
            auto_metrics=True,
            auto_source={
                "task_id": int(task.get("id") or 0),
                "epoch": payload.get("epoch"),
                "step": payload.get("step"),
                "trigger": payload.get("trigger"),
            },
        )
    except Exception:
        logger.exception(
            "auto eval sample enqueue failed for task=%s checkpoint=%s",
            task.get("id"),
            payload.get("checkpoint_path"),
        )
        return None

    logger.info(
        "queued auto eval_samples job=%s run=%s checkpoint=%s",
        job.get("id"),
        run.get("run_id"),
        checkpoint,
    )
    return job, run


def run_checkpoint_eval_for_task(
    task_id: int,
    payload: dict[str, Any],
    *,
    sample_generator: eval_samples.SampleGenerator | None = None,
    clip_scorer: eval_clip.ClipScorer | None = None,
    dino_scorer: eval_dino.DinoScorer | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any] | None:
    """Run checkpoint eval immediately from inside the training process.

    This path is used for auto eval during training. It intentionally does not
    create project_jobs, because the training task itself owns the GPU and must
    wait for the eval result before continuing to the next epoch/step.
    """
    cfg = secrets.load().eval_metrics
    if not cfg.auto_eval_on_checkpoint:
        return None
    if cfg.auto_eval_trigger != "checkpoint":
        return None

    progress = on_progress or (lambda _line: None)
    with db.connection_for() as conn:
        task = db.get_task(conn, int(task_id))
        if not task:
            return None
        resolved = _resolve_task_checkpoint(conn, task, payload)
        if resolved is None:
            return None
        project, version, vdir, checkpoint = resolved

    auto_source = {
        "task_id": int(task_id),
        "epoch": payload.get("epoch"),
        "step": payload.get("step"),
        "trigger": payload.get("trigger"),
        "inline": True,
    }
    progress(
        f"[eval-auto] start checkpoint={checkpoint} "
        f"epoch={payload.get('epoch')} step={payload.get('step')}"
    )
    run = eval_samples.create_run(
        project,
        version,
        vdir,
        checkpoint_path=checkpoint,
        max_items=cfg.auto_eval_max_items,
        auto_metrics=True,
        auto_source=auto_source,
    )

    sample_result = eval_samples.run_sample_job(
        project,
        version,
        vdir,
        str(run["run_id"]),
        generator=sample_generator,
        on_progress=progress,
    )
    results: dict[str, Any] = {"run": sample_result, "metrics": {}}
    if sample_result.get("status") != "done":
        progress(f"[eval-auto] sample status={sample_result.get('status')}")
        return results

    for key, runner, model_name, scorer in (
        ("clip", eval_clip.run_clip_job, cfg.clip_model_name, clip_scorer),
        ("dino", eval_dino.run_dino_job, cfg.dino_model_name, dino_scorer),
    ):
        try:
            results["metrics"][key] = runner(
                project,
                version,
                vdir,
                str(run["run_id"]),
                model_name=model_name,
                scorer=scorer,
                on_progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("inline auto eval %s failed for run=%s", key, run["run_id"])
            progress(f"[eval-auto] {key} failed: {exc}")
    return results


def queue_training_finished_eval(
    conn,
    task: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Queue eval sample jobs for all saved LoRA checkpoints after training."""
    cfg = secrets.load().eval_metrics
    if not cfg.auto_eval_on_checkpoint:
        return []
    if cfg.auto_eval_trigger != "after_training":
        return []

    project_id = int(task.get("project_id") or 0)
    version_id = int(task.get("version_id") or 0)
    if not project_id or not version_id:
        return []

    project = projects.get_project(conn, project_id)
    version = versions.get_version(conn, version_id)
    if not project or not version or int(version["project_id"]) != project_id:
        return []

    vdir = versions.version_dir(project_id, str(project["slug"]), str(version["label"]))
    queued: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for checkpoint in versions.list_lora_ckpts(vdir):
        rel = _checkpoint_relative_to_output(
            project_id,
            str(project.get("slug") or ""),
            version,
            str(checkpoint.get("path") or ""),
        )
        if not rel:
            continue
        try:
            job, run = eval_samples.start_job(
                conn,
                project,
                version,
                vdir,
                checkpoint_path=rel,
                max_items=cfg.auto_eval_max_items,
                auto_metrics=True,
                auto_source={
                    "task_id": int(task.get("id") or 0),
                    "epoch": (payload or {}).get("epoch"),
                    "step": (payload or {}).get("step"),
                    "trigger": "after_training",
                },
            )
        except Exception:
            logger.exception(
                "auto eval after-training enqueue failed for task=%s checkpoint=%s",
                task.get("id"),
                checkpoint.get("path"),
            )
            continue
        queued.append((job, run))

    logger.info(
        "queued after-training eval sample jobs for task=%s count=%s",
        task.get("id"),
        len(queued),
    )
    return queued


def queue_metric_jobs_for_sample(
    conn,
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    """Queue CLIP and DINO metrics after an automatically queued sample run."""
    cfg = secrets.load().eval_metrics
    jobs: list[dict[str, Any]] = []
    clip_job = _active_metric_job(
        conn,
        project=project,
        version=version,
        kind=eval_clip.JOB_KIND,
        run_id=run_id,
    )
    if clip_job is None:
        clip_job, _ = eval_clip.start_job(
            conn,
            project,
            version,
            version_dir,
            run_id,
            model_name=cfg.clip_model_name,
        )
    jobs.append(clip_job)
    dino_job = _active_metric_job(
        conn,
        project=project,
        version=version,
        kind=eval_dino.JOB_KIND,
        run_id=run_id,
    )
    if dino_job is None:
        dino_job, _ = eval_dino.start_job(
            conn,
            project,
            version,
            version_dir,
            run_id,
            model_name=cfg.dino_model_name,
        )
    jobs.append(dino_job)
    logger.info(
        "queued auto eval metric jobs for run=%s clip_job=%s dino_job=%s",
        run_id,
        clip_job.get("id"),
        dino_job.get("id"),
    )
    return jobs


def _resolve_task_checkpoint(
    conn,
    task: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, str] | None:
    project_id = int(task.get("project_id") or 0)
    version_id = int(task.get("version_id") or 0)
    if not project_id or not version_id:
        return None

    project = projects.get_project(conn, project_id)
    version = versions.get_version(conn, version_id)
    if not project or not version or int(version["project_id"]) != project_id:
        return None

    checkpoint = _checkpoint_relative_to_output(
        project_id,
        str(project.get("slug") or ""),
        version,
        str(payload.get("checkpoint_path") or ""),
    )
    if not checkpoint:
        return None

    vdir = versions.version_dir(project_id, str(project["slug"]), str(version["label"]))
    return project, version, vdir, checkpoint


def _active_metric_job(
    conn,
    *,
    project: dict[str, Any],
    version: dict[str, Any],
    kind: str,
    run_id: str,
) -> dict[str, Any] | None:
    for status in ("pending", "running"):
        for job in project_jobs.list_jobs(
            conn,
            project_id=int(project["id"]),
            version_id=int(version["id"]),
            kind=kind,
            status=status,
        ):
            params = job.get("params_decoded")
            if not isinstance(params, dict):
                continue
            if str(params.get("run_id") or "") == str(run_id):
                return job
    return None


def _checkpoint_relative_to_output(
    project_id: int,
    project_slug: str,
    version: dict[str, Any],
    raw_path: str,
) -> str | None:
    if not raw_path or not project_slug:
        return None
    vdir = versions.version_dir(project_id, project_slug, str(version["label"]))
    output_dir = (vdir / "output").resolve()
    path = Path(raw_path)
    if not path.is_absolute():
        path = output_dir / raw_path.replace("\\", "/")
    try:
        rel = path.resolve().relative_to(output_dir)
    except ValueError:
        logger.warning("auto eval checkpoint outside output dir: %s", raw_path)
        return None
    return rel.as_posix()
