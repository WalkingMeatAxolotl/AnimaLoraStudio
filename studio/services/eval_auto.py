"""POC glue for automatically evaluating saved training checkpoints."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from studio import secrets
from studio.services import eval_clip, eval_dino, eval_samples
from studio.services.projects import projects, versions

logger = logging.getLogger(__name__)


def queue_checkpoint_eval(
    conn,
    task: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Queue eval_samples for a saved LoRA checkpoint when the POC switch is on."""
    cfg = secrets.load().eval_metrics
    if not cfg.auto_eval_on_checkpoint:
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
    clip_job, _ = eval_clip.start_job(
        conn,
        project,
        version,
        version_dir,
        run_id,
        model_name=cfg.clip_model_name,
    )
    jobs.append(clip_job)
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
