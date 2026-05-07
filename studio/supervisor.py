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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import db, project_jobs, secrets as _secrets
from .log_tail import LogTailer, MonitorStatePoller
from .paths import (
    GENERATE_JOBS_DIR, LOGS_DIR, REPO_ROOT, STUDIO_DATA, STUDIO_DB, USER_PRESETS_DIR,
)

logger = logging.getLogger(__name__)

# PP10.2.b：哪些 job kind 吃 GPU。这些 kind 在训练运行中默认会被推迟，
# 除非 secrets.queue.allow_gpu_during_train=True 显式允许并行。
GPU_BOUND_JOB_KINDS: frozenset[str] = frozenset({"tag", "reg_build"})

# 槽位名常量
SLOT_TRAIN = "train"
SLOT_DATA = "data"


@dataclass
class _Slot:
    """Supervisor 内的一个执行槽位。每个槽位最多跑 1 个子进程。

    PP10.2.a 起从「单 _current_* 字段」改成「list[_Slot]」；10.2.b 拆成
    两个槽位：
      - TRAIN 槽：只跑 training tasks（db.tasks 表）
      - DATA  槽：只跑 project_jobs（download / tag / reg_build）
    download 永远跟训练并行；tag / reg_build 看 settings 开关。
    """
    name: str = "main"
    proc: Optional[subprocess.Popen] = None
    kind: Optional[str] = None  # "task" | "job"
    id: Optional[int] = None
    log_fp: Optional[Any] = None
    tailer: Optional[LogTailer] = None
    state_poller: Optional[MonitorStatePoller] = None
    cancel_pending: bool = False

    @property
    def busy(self) -> bool:
        return self.proc is not None

    def reset(self) -> None:
        self.proc = None
        self.kind = None
        self.id = None
        self.log_fp = None
        self.tailer = None
        self.state_poller = None
        self.cancel_pending = False

EventCallback = Callable[[dict[str, Any]], None]
CmdBuilder = Callable[[dict[str, Any], Path], list[str]]
JobCmdBuilder = Callable[[dict[str, Any]], list[str]]


def _default_cmd_builder(task: dict[str, Any], config_path: Path) -> list[str]:
    """根据 task_type 路由到对应脚本。"""
    task_type = task.get("task_type", "train")
    if task_type == "generate":
        script = "anima_generate.py"
    elif task_type == "reg_ai":
        script = "anima_reg_ai.py"
    else:
        script = "anima_train.py"
    cmd = [
        sys.executable,
        str(REPO_ROOT / script),
        "--config",
        str(config_path),
    ]
    msp = task.get("monitor_state_path")
    if msp:
        cmd.extend(["--monitor-state-file", str(msp)])
    return cmd


def _maybe_finalize_version(conn: Any, task_id: int) -> None:
    """PP6.3：task 成功完成 → 找 version → 回填 output_lora_path + stage=done。

    output_lora_path 推断：`versions/{label}/output/{output_name}_final.safetensors`
    （anima_train 标准命名）。文件不存在不报错 — 有可能用户用别的命名规则。
    project.stage 也推到 'done' 让 Stepper 反映。
    """
    from . import projects as _projects, versions as _versions
    task_row = db.get_task(conn, task_id)
    if not task_row:
        return
    vid = task_row.get("version_id")
    pid = task_row.get("project_id")
    if not (vid and pid):
        return
    v = _versions.get_version(conn, int(vid))
    p = _projects.get_project(conn, int(pid))
    if not v or not p:
        return
    # 推断 output_lora_path（与 anima_train 默认 `{output_name}_final.safetensors` 一致）
    output_name = f"{p['slug']}_{v['label']}"
    vdir = _versions.version_dir(int(pid), p["slug"], v["label"])
    candidate = vdir / "output" / f"{output_name}_final.safetensors"
    fields: dict[str, Any] = {"stage": "done"}
    if candidate.exists():
        fields["output_lora_path"] = str(candidate)
    _versions.update_version(conn, int(vid), **fields)
    # 项目也推到 done（用户视角整条链跑完了）
    if p.get("stage") in ("training", "configured"):
        _projects.advance_stage(conn, int(pid), "done")


def _resolve_monitor_state_path(task: dict[str, Any]) -> Path:
    """PP6.1 — 决定 task 的 monitor_state.json 落盘路径。

    generate 任务：`studio_data/generate_jobs/{id}/monitor_state.json`。
    训练任务有 version_id：`versions/{label}/monitor_state.json`。
    旧任务兜底：`studio_data/monitors/task_{id}/state.json`。
    """
    if task.get("task_type") in ("generate", "reg_ai"):
        return GENERATE_JOBS_DIR / str(task["id"]) / "monitor_state.json"

    vid = task.get("version_id")
    pid = task.get("project_id")
    if vid and pid:
        # 不在这里 import projects/versions（避免循环）；直接通过 db 查
        with db.connection_for() as conn:
            row = conn.execute(
                "SELECT projects.slug AS slug, versions.label AS label "
                "FROM versions JOIN projects ON versions.project_id = projects.id "
                "WHERE versions.id = ?",
                (vid,),
            ).fetchone()
        if row:
            return (
                STUDIO_DATA / "projects" / f"{pid}-{row['slug']}"
                / "versions" / row["label"] / "monitor_state.json"
            )
    return STUDIO_DATA / "monitors" / f"task_{task['id']}" / "state.json"


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
        # PP10.2.b：双槽位。TRAIN 槽只跑 tasks，DATA 槽只跑 project_jobs。
        # download 永远跟训练并行；tag / reg_build 默认在训练时推迟。
        self._slots: list[_Slot] = [
            _Slot(name=SLOT_TRAIN),
            _Slot(name=SLOT_DATA),
        ]
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
        for slot in self._slots:
            if slot.busy:
                self._terminate_slot(slot)
        if self._thread:
            self._thread.join(timeout=timeout)

    def _find_slot(self, *, kind: str, id: int) -> Optional[_Slot]:
        for slot in self._slots:
            if slot.kind == kind and slot.id == id:
                return slot
        return None

    def cancel(self, task_id: int) -> bool:
        """取消 task：pending → status=canceled；running → 异步发信号立即返回。

        异步路径关键：**不阻塞 web 请求线程**。supervisor 主循环会自然 poll
        proc.poll() 拿到退出码并走 `_finish_slot` 流程，把 status 写为
        canceled。后台 grace timer 在 30s 后还没退就强杀整棵进程树。
        """
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
        if task["status"] == "running":
            slot = self._find_slot(kind="task", id=task_id)
            if slot is not None:
                self._signal_terminate_async(slot)
                return True
        return False

    def cancel_job(self, job_id: int) -> bool:
        """取消 project_job：pending → canceled；running → 异步发信号立即返回。"""
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
        if job["status"] == "running":
            slot = self._find_slot(kind="job", id=job_id)
            if slot is not None:
                self._signal_terminate_async(slot)
                return True
        return False

    @property
    def current_task_id(self) -> Optional[int]:
        for slot in self._slots:
            if slot.kind == "task":
                return slot.id
        return None

    @property
    def current_job_id(self) -> Optional[int]:
        for slot in self._slots:
            if slot.kind == "job":
                return slot.id
        return None

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
        # 1) 先收尸：所有 busy 槽位 poll 一遍，退出的走 _finish_slot
        for slot in self._slots:
            if not slot.busy:
                continue
            assert slot.proc is not None
            rc = slot.proc.poll()
            if rc is not None:
                self._finish_slot(slot, rc)

        # 2) 给空闲槽位派活（按槽位职责分工）
        for slot in self._slots:
            if slot.busy:
                continue
            if slot.name == SLOT_TRAIN:
                self._dispatch_train(slot)
            elif slot.name == SLOT_DATA:
                self._dispatch_data(slot)

    def _dispatch_train(self, slot: _Slot) -> None:
        """TRAIN 槽：只跑 db.tasks 表里的训练 task。"""
        with db.connection_for(self._db_path) as conn:
            task = db.next_pending(conn)
        if task:
            self._spawn_task(slot, task)

    def _dispatch_data(self, slot: _Slot) -> None:
        """DATA 槽：跑 project_jobs（download / tag / reg_build）。

        - download 永远 OK（IO-only，不抢 GPU）
        - tag / reg_build 是 GPU-bound：训练正在跑且未开
          `secrets.queue.allow_gpu_during_train` → 跳过这条 job，等训练结束
          再拉。下一条非 GPU job 仍可派。
        """
        train_busy = self._train_busy()
        allow_gpu = self._allow_gpu_during_train()
        # 简单实现：取 next_pending 那一条；若 GPU-bound 且训练在跑，留着不动
        # （让它仍然 pending），其他 IO-only job 会在下一次 tick 通过 next_pending
        # 重新被取到（next_pending 按 id ASC，下一次同样会指向这条；所以需要
        # 跳过）。这里用 list_jobs 选第一条**可跑**的。
        with db.connection_for(self._db_path) as conn:
            pending = project_jobs.list_jobs(conn, status="pending")
        # list_jobs 默认 ORDER BY id DESC；按入队顺序应该 ASC
        pending.sort(key=lambda j: j["id"])
        for job in pending:
            kind = job["kind"]
            if kind in GPU_BOUND_JOB_KINDS and train_busy and not allow_gpu:
                continue  # 训练中暂缓 GPU job
            self._spawn_job(slot, job)
            return

    def _train_busy(self) -> bool:
        for slot in self._slots:
            if slot.name == SLOT_TRAIN and slot.busy:
                return True
        return False

    def _allow_gpu_during_train(self) -> bool:
        try:
            return bool(_secrets.load().queue.allow_gpu_during_train)
        except Exception:
            return False

    # -------------------------------------------------------------- 子进程
    def _spawn_task(self, slot: _Slot, task: dict[str, Any]) -> None:
        # PP6.3：优先用 task.config_path（version 私有 config 绝对路径）；
        # 没有时降级到老路径 _configs_dir / {config_name}.yaml。
        explicit_cfg = task.get("config_path")
        if explicit_cfg:
            cfg_path = Path(explicit_cfg)
        else:
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
                    error_msg=(
                        f"config not found: {cfg_path}"
                        if explicit_cfg
                        else f"preset not found: {task['config_name']}"
                    ),
                )
            self._on_event(
                {
                    "type": "task_state_changed",
                    "task_id": task["id"],
                    "status": "failed",
                }
            )
            return

        # PP6.1 — 计算 per-task monitor 状态文件路径
        # 有 version_id：versions/{label}/monitor_state.json
        # 没有：studio_data/monitors/task_{id}/state.json（兜底）
        monitor_state_path = _resolve_monitor_state_path(task)
        # 提前注入到 task dict 供 cmd_builder 用，以及落库
        task = dict(task)
        task["monitor_state_path"] = str(monitor_state_path)

        self._logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._logs_dir / f"{task['id']}.log"
        log_fp = open(log_path, "wb")

        cmd = self._cmd_builder(task, cfg_path)
        proc = self._popen(cmd, log_fp)

        slot.proc = proc
        slot.kind = "task"
        slot.id = task["id"]
        slot.log_fp = log_fp
        slot.cancel_pending = False

        # PP6.4 — log tail → SSE（取代前端 2s 轮询 /api/logs/{id}）
        tid = task["id"]

        def _on_task_log(line: str) -> None:
            self._on_event({
                "type": "task_log_appended",
                "task_id": tid,
                "text": line,
                "seq": next(self._log_seq),
            })

        slot.tailer = LogTailer(log_path, _on_task_log)
        slot.tailer.start()

        # PP6.4 — monitor_state.json 变化 → SSE（取代前端 1Hz 轮询 /api/state）
        def _on_state(state: dict[str, Any]) -> None:
            self._on_event({
                "type": "monitor_state_updated",
                "task_id": tid,
                "state": state,
            })

        slot.state_poller = MonitorStatePoller(monitor_state_path, _on_state)
        slot.state_poller.start()

        with db.connection_for(self._db_path) as conn:
            db.update_task(
                conn,
                task["id"],
                status="running",
                started_at=time.time(),
                pid=proc.pid,
                monitor_state_path=str(monitor_state_path),
            )
        self._on_event(
            {
                "type": "task_state_changed",
                "task_id": task["id"],
                "status": "running",
            }
        )
        logger.info(
            "started task %d on slot=%s (pid=%d)", task["id"], slot.name, proc.pid
        )

    def _spawn_job(self, slot: _Slot, job: dict[str, Any]) -> None:
        log_path = Path(job.get("log_path") or project_jobs.log_path_for(job["id"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # worker 自己 append 模式开 log，supervisor 这里只挂个 stdout 转发到同一文件
        log_fp = open(log_path, "ab")

        cmd = self._job_cmd_builder(job)
        proc = self._popen(cmd, log_fp)

        with db.connection_for(self._db_path) as conn:
            project_jobs.mark_running(conn, job["id"], pid=proc.pid)

        slot.proc = proc
        slot.kind = "job"
        slot.id = job["id"]
        slot.log_fp = log_fp
        slot.cancel_pending = False

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

        slot.tailer = LogTailer(log_path, _on_line)
        slot.tailer.start()

        self._on_event({
            "type": "job_state_changed",
            "job_id": jid,
            "project_id": pid_,
            "version_id": vid,
            "kind": kind,
            "status": "running",
        })
        logger.info(
            "started job %d on slot=%s (kind=%s, pid=%d)",
            jid, slot.name, kind, proc.pid,
        )

    def _popen(self, cmd: list[str], log_fp: Any) -> subprocess.Popen:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        # Windows 默认 stdout 用 cp936；任何 worker 写中文 / emoji 会触发
        # UnicodeEncodeError，logging 默认 backslashreplace 转成 \uXXXX，让
        # task log 里全是乱码。这里给所有子进程兜底 UTF-8 + 不缓冲。
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        # 减少底层库的加载进度条（safetensors / transformers / accelerate 等
        # 在 stdout=pipe 时会逐行打几百行 `Loading weights: NN%|...`，淹没用户
        # 自己的训练日志）。仅静音「加载进度」，不影响 logger.error / 训练步进。
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TRANSFORMERS_VERBOSITY", "error")
        env.setdefault("DIFFUSERS_VERBOSITY", "error")
        env.setdefault("ACCELERATE_DISABLE_RICH", "1")
        return subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            creationflags=creationflags,
            env=env,
        )

    def _finish_slot(self, slot: _Slot, rc: int) -> None:
        kind = slot.kind
        cid = slot.id
        assert cid is not None and kind is not None
        if slot.log_fp:
            try:
                slot.log_fp.close()
            except Exception:
                pass
        if slot.tailer:
            try:
                slot.tailer.stop()
            except Exception:
                pass
        if slot.state_poller:
            try:
                slot.state_poller.stop()
            except Exception:
                pass

        if slot.cancel_pending:
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
                # PP6.3：训练成功时回填 version.output_lora_path + 推 stage=done
                if status == "done":
                    _maybe_finalize_version(conn, cid)
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

        slot.reset()

    def _terminate_slot(self, slot: _Slot) -> None:
        """同步终止指定槽位的子进程（仅 supervisor.stop() 用）。

        web 请求路径下的 cancel 请用 `_signal_terminate_async`，避免阻塞
        请求线程 30 秒。
        """
        if not slot.proc:
            return
        slot.cancel_pending = True
        proc = slot.proc
        self._send_terminate_signal(proc)
        try:
            proc.wait(timeout=self._grace)
        except subprocess.TimeoutExpired:
            logger.warning(
                "%s %s on slot=%s did not exit in %.0fs, killing process tree",
                slot.kind, slot.id, slot.name, self._grace,
            )
            _kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _signal_terminate_async(self, slot: _Slot) -> None:
        """非阻塞：发软终止信号，启动后台 grace timer 强杀进程树。

        web 请求线程立刻返回，让 reload() 不被取消请求阻塞 30 秒。supervisor
        主循环每 POLL_INTERVAL 秒 poll proc.poll()，进程一旦退出就走
        `_finish_slot` 把 status 改成 canceled 并 publish 事件。
        """
        if not slot.proc:
            return
        slot.cancel_pending = True
        proc = slot.proc
        self._send_terminate_signal(proc)

        grace = self._grace

        def _grace_then_kill_tree() -> None:
            # 不能用 proc.wait() — 会跟 supervisor 主循环的 poll 抢；改成轮询
            deadline = time.time() + grace
            while time.time() < deadline:
                if proc.poll() is not None:
                    return
                time.sleep(0.5)
            if proc.poll() is None:
                logger.warning(
                    "proc %d did not exit in %.0fs, killing process tree",
                    proc.pid, grace,
                )
                _kill_process_tree(proc.pid)

        threading.Thread(
            target=_grace_then_kill_tree,
            name=f"cancel-grace-{proc.pid}",
            daemon=True,
        ).start()

    @staticmethod
    def _send_terminate_signal(proc: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        except Exception:
            logger.exception("send terminate signal failed")


def _kill_process_tree(pid: int) -> None:
    """杀掉以 pid 为根的整棵进程树。

    Windows 上 `proc.kill()` 只杀 immediate child，DataLoader workers /
    accelerate 的 sub-subprocess 会留下来占着 GPU；用 `taskkill /T /F` 能
    递归到整个进程树。POSIX 用 killpg。
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                check=False, capture_output=True, timeout=10,
            )
        except Exception:
            logger.exception("taskkill /T /F failed for pid %d", pid)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            logger.exception("killpg failed for pid %d", pid)
