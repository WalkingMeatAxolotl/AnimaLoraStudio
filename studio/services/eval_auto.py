"""POC glue for automatically evaluating saved training checkpoints."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from studio import db
from studio import secrets
from studio.infrastructure.paths import task_eval_dir
from studio.services import eval_clip, eval_dino, eval_registry, eval_samples
from studio.services.projects import jobs as project_jobs, projects, versions

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


def _version_eval_enabled(project: dict[str, Any], version: dict[str, Any]) -> bool:
    """Per-version opt-in for post-training validation metrics (training config)."""
    from studio.services import version_config
    try:
        cfg = version_config.read_version_config(project, version)
    except Exception:
        return False
    return bool(cfg.get("eval_validation_enabled"))


def queue_checkpoint_eval(
    conn,
    task: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Queue eval_samples for a saved LoRA checkpoint (checkpoint-trigger path).

    Gated on the version's per-version opt-in (training config
    ``eval_validation_enabled``) and the global ``auto_eval_trigger`` being
    ``checkpoint``. This is the supervisor-side fallback for when inline eval
    cannot run inside the training process (see ``run_checkpoint_eval_for_task``).
    """
    cfg = secrets.load().eval_metrics
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
    if not _version_eval_enabled(project, version):
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
    task_id = int(task.get("id") or 0)
    eval_root = task_eval_dir(task_id) if task_id else None
    try:
        job, run = eval_samples.start_job(
            conn,
            project,
            version,
            vdir,
            checkpoint_path=checkpoint,
            auto_metrics=True,
            auto_source={
                "task_id": int(task.get("id") or 0),
                "epoch": payload.get("epoch"),
                "step": payload.get("step"),
                "trigger": payload.get("trigger"),
            },
            eval_root=eval_root,
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

    Gated on the version's per-version opt-in (training config
    ``eval_validation_enabled``) and the global ``auto_eval_trigger`` being
    ``checkpoint``. Returns ``None`` when not applicable so the caller may fall
    back to the supervisor event path.
    """
    cfg = secrets.load().eval_metrics
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
        if not _version_eval_enabled(project, version):
            return None

    auto_source = {
        "task_id": int(task_id),
        "epoch": payload.get("epoch"),
        "step": payload.get("step"),
        "trigger": payload.get("trigger"),
        "inline": True,
    }
    eval_root = task_eval_dir(int(task_id))
    progress(
        f"[eval-auto] start checkpoint={checkpoint} "
        f"epoch={payload.get('epoch')} step={payload.get('step')}"
    )
    run = eval_samples.create_run(
        project,
        version,
        vdir,
        checkpoint_path=checkpoint,
        auto_metrics=True,
        auto_source=auto_source,
        eval_root=eval_root,
    )

    sample_result = eval_samples.run_sample_job(
        project,
        version,
        vdir,
        str(run["run_id"]),
        generator=sample_generator,
        on_progress=progress,
        eval_root=eval_root,
    )
    results: dict[str, Any] = {"run": sample_result, "metrics": {}}
    if sample_result.get("status") != "done":
        progress(f"[eval-auto] sample status={sample_result.get('status')}")
        return results

    # 只跑「Settings 勾选的指标」对应的 runner（registry 决定）；clip/dino 先接进来，
    # 全启用时与原行为一致。ccip/tag runner 后续接入这张表。
    inline_runners = {
        "clip": (eval_clip.run_clip_job, cfg.clip_model_name, clip_scorer),
        "dino": (eval_dino.run_dino_job, cfg.dino_model_name, dino_scorer),
    }
    for runner_key in eval_registry.enabled_runners(cfg.enabled_metrics):
        spec = inline_runners.get(runner_key)
        if spec is None:
            continue
        run_fn, model_name, scorer = spec
        try:
            results["metrics"][runner_key] = run_fn(
                project,
                version,
                vdir,
                str(run["run_id"]),
                model_name=model_name,
                scorer=scorer,
                on_progress=progress,
                eval_root=eval_root,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("inline auto eval %s failed for run=%s", runner_key, run["run_id"])
            progress(f"[eval-auto] {runner_key} failed: {exc}")
    return results


def queue_training_finished_eval(
    conn,
    task: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Queue eval sample jobs for all saved LoRA checkpoints after training.

    Gated on the version's per-version opt-in (training config
    ``eval_validation_enabled``) and the global ``auto_eval_trigger`` being
    ``after_training``; the held-out validation set is the reference.
    """
    cfg = secrets.load().eval_metrics
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

    if not _version_eval_enabled(project, version):
        return []

    vdir = versions.version_dir(project_id, str(project["slug"]), str(version["label"]))
    task_id = int(task.get("id") or 0)
    eval_root = task_eval_dir(task_id) if task_id else None
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
                auto_metrics=True,
                auto_source={
                    "task_id": int(task.get("id") or 0),
                    "epoch": (payload or {}).get("epoch"),
                    "step": (payload or {}).get("step"),
                    "trigger": "after_training",
                },
                eval_root=eval_root,
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


def queue_manual_task_eval(
    conn,
    task: dict[str, Any],
    checkpoints: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Queue task-scoped eval sample+metric jobs for an explicit checkpoint set.

    Unlike :func:`queue_training_finished_eval` / :func:`queue_checkpoint_eval`,
    this is the *manual* entry point (a user clicking "运行评估" on a finished
    task), so it does NOT gate on the per-version opt-in or ``auto_eval_trigger``.
    It evaluates the full validation set and reuses the Settings metric models,
    writing under ``tasks/<id>/eval/`` so results show up in that task's eval page.
    """
    project_id = int(task.get("project_id") or 0)
    version_id = int(task.get("version_id") or 0)
    task_id = int(task.get("id") or 0)
    if not project_id or not version_id or not task_id:
        return []

    project = projects.get_project(conn, project_id)
    version = versions.get_version(conn, version_id)
    if not project or not version or int(version["project_id"]) != project_id:
        return []

    vdir = versions.version_dir(project_id, str(project["slug"]), str(version["label"]))
    eval_root = task_eval_dir(task_id)
    queued: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for raw in checkpoints:
        rel = _checkpoint_relative_to_output(
            project_id, str(project.get("slug") or ""), version, str(raw or "")
        )
        if not rel or rel in seen:
            continue
        seen.add(rel)
        try:
            job, run = eval_samples.start_job(
                conn,
                project,
                version,
                vdir,
                checkpoint_path=rel,
                auto_metrics=True,
                auto_source={"task_id": task_id, "trigger": "manual"},
                eval_root=eval_root,
            )
        except Exception:
            logger.exception(
                "manual eval enqueue failed for task=%s checkpoint=%s", task_id, raw
            )
            continue
        queued.append((job, run))

    logger.info(
        "queued manual eval sample jobs for task=%s count=%s", task_id, len(queued)
    )
    return queued


def queue_metric_jobs_for_sample(
    conn,
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
    *,
    eval_root: Path | None = None,
    task_id: int | None = None,
) -> list[dict[str, Any]]:
    """Queue metric jobs（按 Settings 勾选的指标对应 runner）after a sample run。

    只排「启用集合」对应的 runner；已有活跃 job 的复用、不重排。clip/dino 先接进
    这张表，全启用时与原行为一致；ccip/tag runner 后续加入 `queue_runners`。
    """
    cfg = secrets.load().eval_metrics
    # runner key → (JOB_KIND, start_job, model_name)
    queue_runners = {
        "clip": (eval_clip.JOB_KIND, eval_clip.start_job, cfg.clip_model_name),
        "dino": (eval_dino.JOB_KIND, eval_dino.start_job, cfg.dino_model_name),
    }
    jobs: list[dict[str, Any]] = []
    for runner_key in eval_registry.enabled_runners(cfg.enabled_metrics):
        spec = queue_runners.get(runner_key)
        if spec is None:
            continue
        kind, start_job, model_name = spec
        job = _active_metric_job(
            conn,
            project=project,
            version=version,
            kind=kind,
            run_id=run_id,
            task_id=task_id,
        )
        if job is None:
            job, _ = start_job(
                conn,
                project,
                version,
                version_dir,
                run_id,
                model_name=model_name,
                eval_root=eval_root,
                task_id=task_id,
            )
        jobs.append(job)
    logger.info(
        "queued auto eval metric jobs for run=%s: %s",
        run_id,
        [j.get("id") for j in jobs],
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
    task_id: int | None = None,
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
                if task_id is not None and int(params.get("task_id") or 0) != int(task_id):
                    continue
                if task_id is None and int(params.get("task_id") or 0):
                    continue
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
