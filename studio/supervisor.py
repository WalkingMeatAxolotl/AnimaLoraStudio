"""任务调度守护线程：从 SQLite 拉 pending 任务，spawn 子进程。

设计要点：
    - 单进程串行（一次最多一个 worker，避开多任务抢 GPU 的复杂度）
    - 调度优先级：project_jobs (download/tag/reg_build) > training tasks
      —— 让数据准备类工作不被训练堵住
    - 每个任务一份独立日志：
        * task: studio_data/logs/{task_id}.log
        * job:  studio_data/jobs/{job_id}.log
      job 跑的时候开 LogTailer 把日志增量 publish 成 job_log_appended SSE
    - 取消用 SIGTERM (Unix) / CTRL_BREAK_EVENT (Windows)，30 秒超时再 kill
    - 启动恢复：重启时把 status='running' 的孤儿 task / job 标 failed
    - 测试可注入 cmd_builder 替代真实 worker 调用
"""
from __future__ import annotations

import itertools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import db, project_jobs
from .log_tail import LogTailer
from .paths import LOGS_DIR, REPO_ROOT, STUDIO_DB, USER_PRESETS_DIR

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]
CmdBuilder = Callable[[dict[str, Any], Path], list[str]]
JobCmdBuilder = Callable[[dict[str, Any]], list[str]]


def _default_cmd_builder(task: dict[str, Any], config_path: Path) -> list[str]:
    """默认调用 anima_train.py --config <path> --no-monitor。"""
    return [
        sys.executable,
        str(REPO_ROOT / "anima_train.py"),
        "--config",
        str(config_path),
        "--no-monitor",
    ]


def _default_job_cmd_builder(job: dict[str, Any]) -> list[str]:
    """默认按 kind 选 worker 模块。"""
    kind = job["kind"]
    return [
        sys.executable,
        "-m",
        f"studio.workers.{kind}_worker",
        "--job-id",
        str(job["id"]),
    ]


class Supervisor:
    POLL_INTERVAL = 1.0
    TERMINATE_GRACE = 30.0

    def __init__(
        self,
        *,
        on_event: Optional[EventCallback] = None,
        cmd_builder: Optional[CmdBuilder] = None,
        job_cmd_builder: Optional[JobCmdBuilder] = None,
        db_path: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        configs_dir: Optional[Path] = None,
        poll_interval: Optional[float] = None,
        terminate_grace: Optional[float] = None,
    ) -> None:
        self._on_event: EventCallback = on_event or (lambda _evt: None)
        self._cmd_builder: CmdBuilder = cmd_builder or _default_cmd_builder
        self._job_cmd_builder: JobCmdBuilder = (
            job_cmd_builder or _default_job_cmd_builder
        )
        self._db_path = db_path or STUDIO_DB
        self._logs_dir = logs_dir or LOGS_DIR
        self._configs_dir = configs_dir or USER_PRESETS_DIR
        self._poll = poll_interval if poll_interval is not None else self.POLL_INTERVAL
        self._grace = terminate_grace if terminate_grace is not None else self.TERMINATE_GRACE

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # 当前活跃单元（task 或 job 二选一）
        self._current_proc: Optional[subprocess.Popen] = None
        self._current_kind: Optional[str] = None  # "task" | "job"
        self._current_id: Optional[int] = None
        self._current_log_fp: Optional[Any] = None
        self._current_tailer: Optional[LogTailer] = None
        self._cancel_pending = False
        self._log_seq = itertools.count()

    # ------------------------------------------------------------------ 控制
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="studio-supervisor", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._current_proc:
            self._terminate_current()
        if self._thread:
            self._thread.join(timeout=timeout)

    def cancel(self, task_id: int) -> bool:
        """取消 task：pending → status=canceled；running → SIGTERM。"""
        with db.connection_for(self._db_path) as conn:
            task = db.get_task(conn, task_id)
            if not task:
                return False
            if task["status"] == "pending":
                db.update_task(
                    conn, task_id, status="canceled", finished_at=time.time()
                )
                self._on_event(
                    {"type": "task_state_changed", "task_id": task_id, "status": "canceled"}
                )
                return True
        if (
            task["status"] == "running"
            and self._current_kind == "task"
            and self._current_id == task_id
        ):
            self._terminate_current()
            return True
        return False

    def cancel_job(self, job_id: int) -> bool:
        """取消 project_job：pending → canceled；running → SIGTERM。"""
        with db.connection_for(self._db_path) as conn:
            job = project_jobs.get_job(conn, job_id)
            if not job:
                return False
            if job["status"] == "pending":
                project_jobs.mark_canceled(conn, job_id)
                self._on_event(
                    {
                        "type": "job_state_changed",
                        "job_id": job_id,
                        "project_id": job["project_id"],
                        "version_id": job.get("version_id"),
                        "kind": job["kind"],
                        "status": "canceled",
                    }
                )
                return True
        if (
            job["status"] == "running"
            and self._current_kind == "job"
            and self._current_id == job_id
        ):
            self._terminate_current()
            return True
        return False

    @property
    def current_task_id(self) -> Optional[int]:
        return self._current_id if self._current_kind == "task" else None

    @property
    def current_job_id(self) -> Optional[int]:
        return self._current_id if self._current_kind == "job" else None

    # -------------------------------------------------------------- 主循环
    def _loop(self) -> None:
        try:
            self._reconcile_orphans()
        except Exception:
            logger.exception("reconcile failed")
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("supervisor tick failed")
            self._stop.wait(self._poll)

    def _reconcile_orphans(self) -> None:
        with db.connection_for(self._db_path) as conn:
            for t in db.list_tasks(conn, status="running"):
                logger.info("orphan running task %d → failed", t["id"])
                db.update_task(
                    conn,
                    t["id"],
                    status="failed",
                    finished_at=time.time(),
                    pid=None,
                    error_msg="supervisor restart while task was running",
                )
                self._on_event(
                    {
                        "type": "task_state_changed",
                        "task_id": t["id"],
                        "status": "failed",
                    }
                )
            n = project_jobs.cleanup_orphan_running(conn)
            if n:
                logger.info("orphan running jobs → failed: %d", n)

    def _tick(self) -> None:
        if self._current_proc:
            rc = self._current_proc.poll()
            if rc is None:
                return
            self._finish_current(rc)
            return

        # 优先调度 project_jobs（数据准备）
        with db.connection_for(self._db_path) as conn:
            job = project_jobs.next_pending(conn)
        if job:
            self._spawn_job(job)
            return

        with db.connection_for(self._db_path) as conn:
            task = db.next_pending(conn)
        if task:
            self._spawn_task(task)

    # -------------------------------------------------------------- 子进程
    def _spawn_task(self, task: dict[str, Any]) -> None:
        cfg_path = self._configs_dir / f"{task['config_name']}.yaml"
        if not cfg_path.exists():
            with db.connection_for(self._db_path) as conn:
                now = time.time()
                db.update_task(
                    conn,
                    task["id"],
                    status="failed",
                    started_at=now,
                    finished_at=now,
                    error_msg=f"preset not found: {task['config_name']}",
                )
            self._on_event(
                {
                    "type": "task_state_changed",
                    "task_id": task["id"],
                    "status": "failed",
                }
            )
            return

        self._logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._logs_dir / f"{task['id']}.log"
        log_fp = open(log_path, "wb")

        cmd = self._cmd_builder(task, cfg_path)
        proc = self._popen(cmd, log_fp)

        self._current_proc = proc
        self._current_kind = "task"
        self._current_id = task["id"]
        self._current_log_fp = log_fp
        self._cancel_pending = False

        with db.connection_for(self._db_path) as conn:
            db.update_task(
                conn,
                task["id"],
                status="running",
                started_at=time.time(),
                pid=proc.pid,
            )
        self._on_event(
            {
                "type": "task_state_changed",
                "task_id": task["id"],
                "status": "running",
            }
        )
        logger.info("started task %d (pid=%d)", task["id"], proc.pid)

    def _spawn_job(self, job: dict[str, Any]) -> None:
        log_path = Path(job.get("log_path") or project_jobs.log_path_for(job["id"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # worker 自己 append 模式开 log，supervisor 这里只挂个 stdout 转发到同一文件
        log_fp = open(log_path, "ab")

        cmd = self._job_cmd_builder(job)
        proc = self._popen(cmd, log_fp)

        with db.connection_for(self._db_path) as conn:
            project_jobs.mark_running(conn, job["id"], pid=proc.pid)

        self._current_proc = proc
        self._current_kind = "job"
        self._current_id = job["id"]
        self._current_log_fp = log_fp
        self._cancel_pending = False

        # tail 增量 → SSE
        jid = job["id"]
        pid_ = job["project_id"]
        vid = job.get("version_id")
        kind = job["kind"]

        def _on_line(line: str) -> None:
            self._on_event({
                "type": "job_log_appended",
                "job_id": jid,
                "project_id": pid_,
                "version_id": vid,
                "kind": kind,
                "text": line,
                "seq": next(self._log_seq),
            })

        self._current_tailer = LogTailer(log_path, _on_line)
        self._current_tailer.start()

        self._on_event({
            "type": "job_state_changed",
            "job_id": jid,
            "project_id": pid_,
            "version_id": vid,
            "kind": kind,
            "status": "running",
        })
        logger.info("started job %d (kind=%s, pid=%d)", jid, kind, proc.pid)

    def _popen(self, cmd: list[str], log_fp: Any) -> subprocess.Popen:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        return subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            creationflags=creationflags,
        )

    def _finish_current(self, rc: int) -> None:
        kind = self._current_kind
        cid = self._current_id
        assert cid is not None and kind is not None
        if self._current_log_fp:
            try:
                self._current_log_fp.close()
            except Exception:
                pass
        if self._current_tailer:
            try:
                self._current_tailer.stop()
            except Exception:
                pass

        if self._cancel_pending:
            status = "canceled"
        elif rc == 0:
            status = "done"
        else:
            status = "failed"

        if kind == "task":
            with db.connection_for(self._db_path) as conn:
                fields: dict[str, Any] = {
                    "status": status,
                    "exit_code": rc,
                    "finished_at": time.time(),
                    "pid": None,
                }
                if status == "failed":
                    fields["error_msg"] = f"exit code {rc}"
                db.update_task(conn, cid, **fields)
            self._on_event(
                {"type": "task_state_changed", "task_id": cid, "status": status}
            )
            logger.info("task %d finished: %s (rc=%d)", cid, status, rc)
        else:  # job
            with db.connection_for(self._db_path) as conn:
                if status == "done":
                    project_jobs.mark_done(conn, cid)
                elif status == "canceled":
                    project_jobs.mark_canceled(conn, cid)
                else:
                    project_jobs.mark_failed(conn, cid, f"exit code {rc}")
                job = project_jobs.get_job(conn, cid)
            self._on_event({
                "type": "job_state_changed",
                "job_id": cid,
                "project_id": job["project_id"] if job else None,
                "version_id": job.get("version_id") if job else None,
                "kind": job["kind"] if job else None,
                "status": status,
            })
            logger.info("job %d finished: %s (rc=%d)", cid, status, rc)

        self._current_proc = None
        self._current_kind = None
        self._current_id = None
        self._current_log_fp = None
        self._current_tailer = None
        self._cancel_pending = False

    def _terminate_current(self) -> None:
        if not self._current_proc:
            return
        self._cancel_pending = True
        try:
            if os.name == "nt":
                self._current_proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._current_proc.terminate()
        except Exception:
            logger.exception("send terminate signal failed")
        try:
            self._current_proc.wait(timeout=self._grace)
        except subprocess.TimeoutExpired:
            logger.warning(
                "%s %s did not exit in %.0fs, killing",
                self._current_kind,
                self._current_id,
                self._grace,
            )
            self._current_proc.kill()
            try:
                self._current_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
