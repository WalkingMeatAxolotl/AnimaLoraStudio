"""Eval CLIP worker.

Runs CLIP-T / CLIP-I for one completed eval sample run and writes metrics.json.
"""
from __future__ import annotations

import logging
from typing import Any

from studio import db
from studio.services import eval_clip
from studio.services.projects import jobs as project_jobs, projects, versions

logger = logging.getLogger(__name__)


def run(job_id: int) -> int:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != eval_clip.JOB_KIND:
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params: dict[str, Any] = job.get("params_decoded") or {}

    def progress(line: str) -> None:
        print(line, flush=True)

    try:
        version_id = int(params.get("version_id") or job.get("version_id") or 0)
        run_id = str(params.get("run_id") or "")
        model_name = str(params.get("model_name") or eval_clip.DEFAULT_MODEL_NAME)
        if not version_id or not run_id:
            progress("[error] missing version_id or run_id")
            return 1

        with db.connection_for() as conn:
            version = versions.get_version(conn, version_id)
            if not version or int(version["project_id"]) != int(job["project_id"]):
                progress(f"[error] version {version_id} not in project {job['project_id']}")
                return 1
            project = projects.get_project(conn, int(job["project_id"]))
        if not project:
            progress(f"[error] project {job['project_id']} missing")
            return 1

        vdir = versions.version_dir(
            int(project["id"]), str(project["slug"]), str(version["label"])
        )
        progress(f"[start] eval_clip run={run_id} version={version['label']}")
        result = eval_clip.run_clip_job(
            project,
            version,
            vdir,
            run_id,
            model_name=model_name,
            on_progress=progress,
        )
        states = result.get("metric_states") or {}
        clip_t = states.get("clip_t") or {}
        clip_i = states.get("clip_i") or {}
        progress(
            "[done] "
            f"clip_t={clip_t.get('status')} value={clip_t.get('value')} "
            f"clip_i={clip_i.get('status')} value={clip_i.get('value')}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("eval clip worker crashed (job_id=%s)", job_id)
        progress(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
