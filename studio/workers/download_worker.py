"""下载 worker 子进程入口（pp2）。

由 supervisor 启动：`python -m studio.workers.download_worker --job-id N`。
读 `project_jobs` 行 + `secrets.gelbooru` → 调
`studio.services.downloader.download()` → 写日志 → 退出码反映成败。
状态字段（running / done / failed）由 supervisor 在子进程结束时统一回写。

日志只走 stdout：supervisor 在 `subprocess.Popen(stdout=log_fp,
stderr=STDOUT)` 把整个子进程输出重定向到 task log 文件，worker 自己**不能**
再 open 同一个 log 直接 write —— 否则同一行会落盘两次，LogTailer 读两次，
前端就看到每条日志重复一次。
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import traceback

from studio import db, project_jobs, projects, secrets
from studio.services import downloader


_cancel_event = threading.Event()


def _on_signal(_signum, _frame) -> None:  # pragma: no cover - signal path
    _cancel_event.set()


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGBREAK"):  # Windows CTRL_BREAK_EVENT
        signal.signal(signal.SIGBREAK, _on_signal)  # type: ignore[attr-defined]


def run(job_id: int) -> int:
    """主体：返回退出码（0 成功 / 1 失败）。"""
    _cancel_event.clear()
    _install_signal_handlers()

    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != "download":
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params = job.get("params_decoded") or {}

    def progress(line: str) -> None:
        print(line, flush=True)

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
            cancel_event=_cancel_event,
        )
        progress(f"[done] saved={saved}")
        return 0
    except Exception as exc:
        progress(f"[error] {exc}")
        print(traceback.format_exc(), flush=True)
        return 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", type=int, required=True)
    args = p.parse_args()
    sys.exit(run(args.job_id))


if __name__ == "__main__":
    main()
