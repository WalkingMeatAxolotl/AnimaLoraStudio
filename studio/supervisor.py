"""任务调度守护线程：从 SQLite 拉 pending 任务，spawn anima_train.py 子进程。

设计要点：
    - 单进程串行（一次最多一个 worker，避开多任务抢 GPU 的复杂度）
    - 每个任务一份独立日志：studio_data/logs/{task_id}.log
    - 取消用 SIGTERM (Unix) / CTRL_BREAK_EVENT (Windows)，30 秒超时再 kill
    - 启动恢复：重启时把 status='running' 的孤儿任务标 failed
    - 测试可注入 cmd_builder 替代 anima_train.py 真实调用
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import db
from .paths import LOGS_DIR, REPO_ROOT, STUDIO_DB, USER_CONFIGS_DIR

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]
CmdBuilder = Callable[[dict[str, Any], Path], list[str]]


def _default_cmd_builder(task: dict[str, Any], config_path: Path) -> list[str]:
    """默认调用 anima_train.py --config <path> --no-monitor。"""
    return [
        sys.executable,
        str(REPO_ROOT / "anima_train.py"),
        "--config",
        str(config_path),
        "--no-monitor",
    ]


class Supervisor:
    POLL_INTERVAL = 1.0
    TERMINATE_GRACE = 30.0

    def __init__(
        self,
        *,
        on_event: Optional[EventCallback] = None,
        cmd_builder: Optional[CmdBuilder] = None,
        db_path: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        configs_dir: Optional[Path] = None,
        poll_interval: Optional[float] = None,
        terminate_grace: Optional[float] = None,
    ) -> None:
        self._on_event: EventCallback = on_event or (lambda _evt: None)
        self._cmd_builder: CmdBuilder = cmd_builder or _default_cmd_builder
        self._db_path = db_path or STUDIO_DB
        self._logs_dir = logs_dir or LOGS_DIR
        self._configs_dir = configs_dir or USER_CONFIGS_DIR
        self._poll = poll_interval if poll_interval is not None else self.POLL_INTERVAL
        self._grace = terminate_grace if terminate_grace is not None else self.TERMINATE_GRACE

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # 当前任务（仅持有，不并发）
        self._current_proc: Optional[subprocess.Popen] = None
        self._current_id: Optional[int] = None
        self._current_log_fp: Optional[Any] = None
        self._cancel_pending = False

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
        """取消任务：pending → status=canceled；running → SIGTERM。"""
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
        if task["status"] == "running" and self._current_id == task_id:
            self._terminate_current()
            return True
        return False

    @property
    def current_task_id(self) -> Optional[int]:
        return self._current_id

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

    def _tick(self) -> None:
        if self._current_proc:
            rc = self._current_proc.poll()
            if rc is None:
                return
            self._finish_current(rc)
            return

        with db.connection_for(self._db_path) as conn:
            task = db.next_pending(conn)
        if task:
            self._spawn(task)

    # -------------------------------------------------------------- 子进程
    def _spawn(self, task: dict[str, Any]) -> None:
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
                    error_msg=f"config not found: {task['config_name']}",
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
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            creationflags=creationflags,
        )
        self._current_proc = proc
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

    def _finish_current(self, rc: int) -> None:
        task_id = self._current_id
        assert task_id is not None
        if self._current_log_fp:
            try:
                self._current_log_fp.close()
            except Exception:
                pass

        if self._cancel_pending:
            status = "canceled"
        elif rc == 0:
            status = "done"
        else:
            status = "failed"

        with db.connection_for(self._db_path) as conn:
            fields: dict[str, Any] = {
                "status": status,
                "exit_code": rc,
                "finished_at": time.time(),
                "pid": None,
            }
            if status == "failed":
                fields["error_msg"] = f"exit code {rc}"
            db.update_task(conn, task_id, **fields)

        self._on_event(
            {"type": "task_state_changed", "task_id": task_id, "status": status}
        )
        logger.info("task %d finished: %s (rc=%d)", task_id, status, rc)
        self._current_proc = None
        self._current_id = None
        self._current_log_fp = None
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
                "task %s did not exit in %.0fs, killing", self._current_id, self._grace
            )
            self._current_proc.kill()
            try:
                self._current_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
