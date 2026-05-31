"""Eval sample worker.

Runs one manifest-backed eval sample run and writes persistent PNGs plus
run.json metadata under versions/{label}/eval/samples/{run_id}/.
"""
from __future__ import annotations

import logging
from typing import Any

from studio import db
from studio.services import eval_samples
from studio.services.projects import jobs as project_jobs, projects, versions

logger = logging.getLogger(__name__)


def run(job_id: int) -> int:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != eval_samples.JOB_KIND:
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params: dict[str, Any] = job.get("params_decoded") or {}

    def progress(line: str) -> None:
        print(line, flush=True)

    try:
        version_id = int(params.get("version_id") or job.get("version_id") or 0)
        run_id = str(params.get("run_id") or "")
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
        progress(f"[start] eval_samples run={run_id} version={version['label']}")
        result = eval_samples.run_sample_job(
            project,
            version,
            vdir,
            run_id,
            on_progress=progress,
        )
        progress(
            f"[done] status={result['status']} "
            f"done={result['summary']['done']}/{result['summary']['total']}"
        )
        return 0 if result["status"] == "done" else 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("eval samples worker crashed (job_id=%s)", job_id)
        progress(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
