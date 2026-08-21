"""下载 worker 子进程入口（pp2）。

由 supervisor 启动：`python -m studio.workers.download_worker --job-id N`。
读 `project_jobs` 行 + `secrets.gelbooru` → 调
`studio.services.downloader.download()` → 写日志 → 退出码反映成败。
状态字段（running / done / failed）由 supervisor 在子进程结束时统一回写。

日志只走 logger（stderr，Human 行契约见 docs/design/logging-target-state.md
§3.2）：supervisor 在 `subprocess.Popen(stdout=log_fp, stderr=STDOUT)` 把整个
子进程输出重定向到 task log 文件，worker 自己**不能**再 open 同一个 log 直接
write —— 否则同一行会落盘两次，LogTailer 读两次，前端就看到每条日志重复一次。
裸 print 只留给 stdout 协议行（`__EVENT__:`，见 preprocess_worker）。
"""
from __future__ import annotations

import logging
import threading

from studio.infrastructure.log_messages import msg
from studio.infrastructure.task_log import TaskLog
from studio import db, secrets

# 固定名：worker 经 `python -m studio.workers.download_worker` 拉起时 __name__ 是 __main__，
# 行契约里的来源列会失真、也不在 OWN_LOGGER_NAMESPACES 里。
logger = logging.getLogger("studio.workers.download_worker")
from studio.services.projects import jobs as project_jobs, projects
from studio.services.booru import downloader


def run(job_id: int) -> int:
    """主体：返回退出码（0 成功 / 1 失败）。"""
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        logger.error("Download job %s not found in the database; nothing to run", job_id)
        return 1
    if job["kind"] != "download":
        logger.error(
            "Internal error: job %s has kind=%s, not a download job; aborting",
            job_id, job["kind"],
        )
        return 1

    params = job.get("params_decoded") or {}

    progress = TaskLog(logger)

    try:
        with db.connection_for() as conn:
            project = projects.get_project(conn, job["project_id"])
        if not project:
            progress.error(
                "Project %s no longer exists; download aborted", job["project_id"]
            )
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
            save_tags=sec.download.save_tags,
            convert_to_png=sec.download.convert_to_png,
            remove_alpha_channel=sec.download.remove_alpha_channel,
            user_id=user_id,
            username=username,
            api_key=api_key,
            exclude_tags=list(sec.download.exclude_tags),
        )
        progress.info(msg(
            "worker.download.start",
            tag=opts.tag,
            count=opts.count,
            source=opts.api_source,
            exclude=",".join(opts.exclude_tags) or "(none)",
        ))
        saved = downloader.download(
            opts,
            dest,
            on_progress=progress,
            cancel_event=threading.Event(),  # supervisor 走 SIGTERM
        )
        progress.info(msg("worker.download.done", saved=saved))
        return 0
    except Exception:
        # PR-1 C7: logger.exception 带 trace_id 进 stderr；异常摘要由 traceback
        # 提供，不再另发一条 progress（C6：{e} 与 traceback 二选一）。
        logger.exception("Download worker crashed: job=%s", job_id)
        return 1


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
