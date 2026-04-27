"""下载 worker 子进程入口（pp2）。

由 supervisor 启动：`python -m studio.workers.download_worker --job-id N`。
读 `project_jobs` 行 + `secrets.gelbooru` → 调
`studio.services.downloader.download()` → 写日志 → 退出码反映成败。
状态字段（running / done / failed）由 supervisor 在子进程结束时统一回写。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Any

from studio import db, project_jobs, projects, secrets
from studio.services import downloader


def _open_log(log_path: str) -> Any:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "a", encoding="utf-8", buffering=1)


def run(job_id: int) -> int:
    """主体：返回退出码（0 成功 / 1 失败）。"""
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != "download":
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params = job.get("params_decoded") or {}
    log_path = job.get("log_path") or str(
        project_jobs.log_path_for(job_id)
    )

    with _open_log(log_path) as log_fp:
        def progress(line: str) -> None:
            log_fp.write(line + "\n")
            print(line, flush=True)  # supervisor 也捕获 stdout

        try:
            with db.connection_for() as conn:
                project = projects.get_project(conn, job["project_id"])
            if not project:
                progress(f"[error] project {job['project_id']} missing")
                return 1
            dest = projects.project_dir(project["id"], project["slug"]) / "download"
            sec = secrets.load()
            api_source = params.get("api_source", "gelbooru")
            if api_source == "danbooru":
                user_id = ""
                username = sec.danbooru.username
                api_key = sec.danbooru.api_key
            else:
                user_id = sec.gelbooru.user_id
                username = ""
                api_key = sec.gelbooru.api_key
            opts = downloader.DownloadOptions(
                tag=params.get("tag", ""),
                count=int(params.get("count", 0)),
                api_source=api_source,
                save_tags=sec.gelbooru.save_tags,
                convert_to_png=sec.gelbooru.convert_to_png,
                remove_alpha_channel=sec.gelbooru.remove_alpha_channel,
                user_id=user_id,
                username=username,
                api_key=api_key,
                exclude_tags=list(sec.download.exclude_tags),
            )
            progress(
                f"[start] tag={opts.tag!r} count={opts.count} "
                f"source={opts.api_source} "
                f"exclude={','.join(opts.exclude_tags) or '(none)'}"
            )
            saved = downloader.download(
                opts,
                dest,
                on_progress=progress,
                cancel_event=threading.Event(),  # supervisor 走 SIGTERM
            )
            progress(f"[done] saved={saved}")
            return 0
        except Exception as exc:
            progress(f"[error] {exc}")
            import traceback
            log_fp.write(traceback.format_exc())
            return 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", type=int, required=True)
    args = p.parse_args()
    sys.exit(run(args.job_id))


if __name__ == "__main__":
    main()
