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
from .paths import LOGS_DIR, REPO_ROOT, STUDIO_DATA, STUDIO_DB, USER_PRESETS_DIR
from .services.inference_daemon import (
    InferenceDaemon,
    STATE_STOPPED as _DAEMON_STOPPED,
    get_daemon,
)

logger = logging.getLogger(__name__)

# PP10.2.b：哪些 job kind 吃 GPU。这些 kind 在训练运行中默认会被推迟，
# 除非 secrets.queue.allow_gpu_during_train=True 显式允许并行。
# preprocess 走 spandrel super-resolution，加载权重到 GPU 推理。
GPU_BOUND_JOB_KINDS: frozenset[str] = frozenset({"preprocess", "tag", "reg_build"})

# Worker → supervisor 的结构化事件标记。worker 写
#   __EVENT__:my_event_type:{"foo":1,"bar":"x"}
# 到 stdout，supervisor 在 _on_line 里识别并 publish 成 typed SSE 事件
# （job_id / project_id 自动注入），不会进 job_log。比专门搭 IPC 轻。
_EVENT_MARKER = "__EVENT__:"

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
    # ADR 0006 PR-2 pause/resume backend ----------------------------------
    pause_pending: bool = False
    # `__EVENT__:pause_state` payload — handle_interrupt 写好 .pt + snapshot 后
    # 子进程通过 stdout emit；_on_line 抓到后填这三个字段，_finish_slot 据此
    # 把 task 标 paused。pause_pending=True 但缺这几个字段 → 子进程退出前
    # 没来得及 emit → 视为 cancel 兜底（ADR §4.3 modal "强制取消保存进度"）。
    pause_state_path: Optional[str] = None
    pause_config_path: Optional[str] = None
    pause_step: Optional[int] = None
    # ADR §8.1 is_pausable 信号 — resume phase emit `train_loop_started` 后才
    # 允许暂停。UI 端 SSE 用这个解锁暂停按钮，API 端 defense-in-depth 拒绝
    # 过早 pause 请求。
    train_loop_started: bool = False

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
        self.pause_pending = False
        self.pause_state_path = None
        self.pause_config_path = None
        self.pause_step = None
        self.train_loop_started = False

EventCallback = Callable[[dict[str, Any]], None]
CmdBuilder = Callable[[dict[str, Any], Path], list[str]]
JobCmdBuilder = Callable[[dict[str, Any]], list[str]]


def _default_cmd_builder(task: dict[str, Any], config_path: Path) -> list[str]:
    """根据 task_type 路由到对应脚本。

    train (默认 / 老 task): runtime/anima_train.py
    reg_ai: runtime/anima_reg_ai.py（先验生成）
    generate: 走 inference_daemon，**不**经这个 cmd_builder，supervisor
        在 _dispatch_generate 里直接派给 daemon。这里 fallback 到 anima_generate.py
        只是为了某天测试可能注入 cmd_builder 时不爆 KeyError —— 实际跑
        不到这条 path（_next_pending_task_in 在 dispatch_train 里只挑
        train/reg_ai）。
    """
    task_type = task.get("task_type") or "train"
    if task_type == "reg_ai":
        script = REPO_ROOT / "runtime" / "anima_reg_ai.py"
    elif task_type == "generate":
        script = REPO_ROOT / "runtime" / "anima_generate.py"  # 兜底，正常路径不来这
    else:
        script = REPO_ROOT / "runtime" / "anima_train.py"
    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(config_path),
    ]
    msp = task.get("monitor_state_path")
    if msp:
        cmd.extend(["--monitor-state-file", str(msp)])
    # ADR 0006 PR-3: paused task 复活 → 注入 --resume-state，让 anima_train
    # 的 resume_phase 加载 state；旁边的 .config.json snapshot 由 bootstrap_phase
    # 自动检测并 freeze args（ADR §5.7）。
    paused_state = task.get("paused_state_path")
    if paused_state:
        cmd.extend(["--resume-state", str(paused_state)])
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

    有 version_id：`versions/{label}/monitor_state.json`，与 train/output/samples
    放一起；用户切 version 监控自然独立。
    没有 version_id（PP1 之前的旧任务）：兜底到
    `studio_data/monitors/task_{id}/state.json`，避免老任务无处可写。
    """
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

        # commit 9：generate task 走 daemon，不占任何 _Slot；用单独字段跟踪。
        # daemon 一次只跑一个 task；模型 lazy load + 跨 task 复用。
        self._daemon_lock = threading.Lock()
        self._daemon_active_task_id: Optional[int] = None
        self._daemon_state_poller: Optional[MonitorStatePoller] = None
        self._daemon_cancel_pending: bool = False
        self._daemon_listener_registered = False

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
        # 关 inference daemon（如果起着）。失败不影响 supervisor 本身退出。
        try:
            get_daemon().stop(timeout=timeout)
        except Exception:
            logger.exception("inference daemon stop failed")
        if self._thread:
            self._thread.join(timeout=timeout)

    def _find_slot(self, *, kind: str, id: int) -> Optional[_Slot]:
        for slot in self._slots:
            if slot.kind == kind and slot.id == id:
                return slot
        return None

    def cancel(self, task_id: int) -> bool:
        """取消 task：pending → status=canceled；running → 异步发信号立即返回。

        ADR 0006 PR-2：paused task 也可被取消（"彻底放弃这个 task"），状态从
        paused 直接改 canceled，并清理对应的 pause 文件对（已不需要了）。

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
        if task["status"] == "paused":
            # 进程已退出，无需发信号 — 复用 _clear_pause_artifacts 删文件 + 清字段，
            # 再单独写 status=canceled + finished_at（ADR §5.5）。
            # 故意走 with 块外：_clear_pause_artifacts 内部开自己 conn，避免嵌套。
            self._clear_pause_artifacts(task_id)
            with db.connection_for(self._db_path) as conn:
                db.update_task(
                    conn, task_id,
                    status="canceled",
                    finished_at=time.time(),
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
            # daemon 上跑的 generate task：kill daemon 进程让它带着丢 cache 退出，
            # 下次 generate task 再 lazy spawn。粗暴但可接受（用户主动取消低频）。
            with self._daemon_lock:
                if self._daemon_active_task_id == task_id:
                    self._daemon_cancel_pending = True
            try:
                get_daemon().stop(timeout=3.0)
            except Exception:
                logger.exception("failed to stop daemon for cancel")
            return True
        return False

    def is_task_pausable(self, task_id: int) -> bool:
        """ADR §8.1: UI is_pausable 信号。

        条件：task 在 slot 上 running、`train_loop_started` 事件已收到、
        没有 pause / cancel pending。任一不满足 → UI 应隐藏暂停按钮。
        """
        slot = self._find_slot(kind="task", id=task_id)
        if slot is None:
            return False
        return (
            slot.proc is not None
            and slot.train_loop_started
            and not slot.pause_pending
            and not slot.cancel_pending
        )

    def pause(self, task_id: int) -> tuple[bool, str]:
        """暂停 running task：发软信号让 handle_interrupt 保 state 后退出。

        返回 (success, reason_if_failed)。

        ADR §8.1 defense-in-depth：API 端调本方法时，UI 应已用 SSE
        `is_pausable` 字段隐藏暂停按钮；本方法服务端再校验 train_loop_started
        信号，未就绪 / 状态非 running / task 不存在 → 拒绝。

        非阻塞：调 `_signal_pause_async` 立刻返回。子进程 emit 事件 →
        `_on_task_log` 更新 slot → 子进程退出 → `_finish_slot` 标 paused。
        UI 端 modal 订阅 SSE 看进度（ADR §4.3）。
        """
        with db.connection_for(self._db_path) as conn:
            task = db.get_task(conn, task_id)
        if not task:
            return False, "task not found"
        if task["status"] != "running":
            return False, f"task status is {task['status']!r}, not running"
        slot = self._find_slot(kind="task", id=task_id)
        if slot is None:
            return False, "task not on a slot (generate-on-daemon not supported)"
        if not slot.train_loop_started:
            return False, "train loop not started yet, retry after a few seconds"
        if slot.pause_pending:
            return False, "pause already pending"
        if slot.cancel_pending:
            return False, "task is being canceled"
        self._signal_pause_async(slot)
        return True, ""

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
        # ADR 0006 PR-2 兼容性 note：此处 list_tasks(status="running") 精确按
        # status 过滤，paused task（status='paused'）天然不进 this loop —
        # 跨 supervisor 重启的 paused task 保持状态不变（ADR §8.4）。
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

        # 3) 派 generate task 给 daemon（独立资源，不占 _Slot）
        self._dispatch_generate()

    # ---- pending task 选择 ----------------------------------------------------
    def _next_pending_task_in(self, types: tuple[str, ...]) -> Optional[dict[str, Any]]:
        """从 pending 队列里找第一条匹配 task_type 的任务。"""
        with db.connection_for(self._db_path) as conn:
            pending = db.list_tasks(conn, status="pending")
        for t in pending:
            tt = t.get("task_type") or "train"
            if tt in types:
                return t
        return None

    def _dispatch_train(self, slot: _Slot) -> None:
        """TRAIN 槽：跑 train / reg_ai task。generate 走 daemon，不在这。

        commit 12：派活前先要求 daemon 让位（unload 释放 VRAM），除非
        secrets.queue.allow_gpu_during_train=true。daemon 在跑 generate
        时不强中断，等下次 tick 它跑完再卸载。

        ADR 0006 PR-2：queue_held=True 时跳过本次派发（ADR §3.2）。已 running
        的 task 不受影响（继续跑到自然结束/暂停/取消）。
        """
        if self._queue_held():
            return
        task = self._next_pending_task_in(("train", "reg_ai"))
        if task is None:
            return
        if self._maybe_yield_daemon():
            return  # daemon 还占 GPU，等下次 tick 派
        self._spawn_task(slot, task)

    def _dispatch_generate(self) -> None:
        """commit 9：把 generate pending task 提交给 daemon，daemon idle 时执行。

        ADR 0006 PR-2：queue_held=True 时跳过（hold 语义全队列覆盖，不区分
        task type）。
        """
        if self._queue_held():
            return
        with self._daemon_lock:
            if self._daemon_active_task_id is not None:
                return
        task = self._next_pending_task_in(("generate",))
        if task is None:
            return
        self._submit_to_daemon(task)

    def _queue_held(self) -> bool:
        """ADR §3.2 queue hold 开关，跨 supervisor 重启保留（db kv）。"""
        try:
            with db.connection_for(self._db_path) as conn:
                return db.get_queue_held(conn)
        except Exception:
            logger.exception("failed to read queue_held")
            return False  # 读失败默认放行，安全降级

    def _maybe_yield_daemon(self) -> bool:
        """commit 12：daemon 占着 GPU 且不许并行 → 触发 unload，调用方应跳过这次派发。

        返回值：
          - True：daemon 还占着 VRAM（在跑 generate 或刚发了 unload 请求），
                  调用方不应该派 GPU 任务，等下次 tick 重检
          - False：daemon 没占 GPU（未起 / 已 unloaded / 用户允许并行）
                  调用方可立刻派
        """
        daemon = get_daemon()
        if not daemon.is_model_loaded:
            return False
        if self._allow_gpu_during_train():
            return False
        if daemon.is_busy:
            # 用户主动触发的 generate 不强中断；等它跑完
            return True
        try:
            daemon.request_unload()
            logger.info("requested daemon unload to yield GPU")
        except Exception:
            logger.exception("daemon unload request failed")
        return True

    def _dispatch_data(self, slot: _Slot) -> None:
        """DATA 槽：跑 project_jobs（download / tag / reg_build）。

        - download 永远 OK（IO-only，不抢 GPU）
        - tag / reg_build 是 GPU-bound：
            * 训练正在跑且未开 `allow_gpu_during_train` → 跳过
            * daemon 占着 VRAM 且未开 `allow_gpu_during_train` → 触发 daemon
              让位（_maybe_yield_daemon），跳过等下次 tick

        ADR 0006 PR-2：queue_held=True 时跳过本次派发，包含 download。语义上
        hold 是"全队列暂停新派活"，不区分 GPU vs IO。
        """
        if self._queue_held():
            return
        train_busy = self._train_busy()
        allow_gpu = self._allow_gpu_during_train()
        with db.connection_for(self._db_path) as conn:
            pending = project_jobs.list_jobs(conn, status="pending")
        pending.sort(key=lambda j: j["id"])
        for job in pending:
            kind = job["kind"]
            if kind in GPU_BOUND_JOB_KINDS:
                if train_busy and not allow_gpu:
                    continue
                if self._maybe_yield_daemon():
                    continue  # daemon 还占 GPU，等
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
        # ADR 0006 PR-1：LORA_TASK_ID 注入让训练子进程把 state 文件写到
        # output_dir/state/task_<TID>/ 子目录，避免同 version 多 task 互覆盖。
        proc = self._popen(cmd, log_fp, extra_env={"LORA_TASK_ID": str(task["id"])})

        slot.proc = proc
        slot.kind = "task"
        slot.id = task["id"]
        slot.log_fp = log_fp
        slot.cancel_pending = False

        # PP6.4 — log tail → SSE（取代前端 2s 轮询 /api/logs/{id}）
        tid = task["id"]

        def _on_task_log(line: str) -> None:
            # ADR 0006 PR-2：训练 worker 通过 __EVENT__: 协议跟 supervisor 通信
            # （pause_state / train_loop_started / resume_state_loaded）。跟
            # jobs 的 _on_line 路径对齐 — 之前 task 这条 path 不识别 event marker，
            # 现在补上。
            if line.startswith(_EVENT_MARKER):
                try:
                    rest = line[len(_EVENT_MARKER):]
                    evt_type, payload_str = rest.split(":", 1)
                    import json as _json
                    payload = _json.loads(payload_str) if payload_str else {}
                except Exception:
                    logger.exception("malformed event marker: %r", line[:200])
                    return  # 不当 log 推
                # 状态机镜像（ADR §8.1 / §`_on_line`）
                if evt_type == "pause_state":
                    slot.pause_state_path = str(payload.get("state_path") or "")
                    slot.pause_config_path = str(payload.get("config_path") or "")
                    slot.pause_step = payload.get("step")
                elif evt_type == "train_loop_started":
                    slot.train_loop_started = True
                elif evt_type == "resume_state_loaded":
                    # ADR §5.5 / PR-3：训练子进程 load_training_state 成功 →
                    # 旧 pause 文件对已被消费完，删盘 + 清 db 字段，避免下次
                    # pause 时跟旧文件命名撞 / 让 ResumeFieldPicker 显示 stale 项。
                    self._clear_pause_artifacts(tid)
                self._on_event({
                    "type": evt_type,
                    "task_id": tid,
                    **payload,
                })
                return
            self._on_event({
                "type": "task_log_appended",
                "task_id": tid,
                "text": line,
                "seq": next(self._log_seq),
            })

        slot.tailer = LogTailer(log_path, _on_task_log)
        slot.tailer.start()

        # PP6.4 → PR #37: monitor_state.json 变化 → SSE monitor_progress (增量协议)
        # payload 是 delta（appended_losses/lr/samples + 最新 step/speed/...），
        # 客户端首次 GET /api/state 拿快照后用这个增量持续 merge。
        def _on_state_delta(delta: dict[str, Any]) -> None:
            self._on_event({
                "type": "monitor_progress",
                "task_id": tid,
                "delta": delta,
            })

        slot.state_poller = MonitorStatePoller(monitor_state_path, _on_state_delta)
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
            # 结构化事件标记：worker 写 `__EVENT__:type:json_payload` 让 supervisor
            # publish 成 typed SSE 事件（不进 job log）。比专门搭 IPC 通道轻，比
            # 让前端按文本 grep 日志靠谱。job_id / project_id 由 supervisor 注入。
            if line.startswith(_EVENT_MARKER):
                try:
                    rest = line[len(_EVENT_MARKER):]
                    evt_type, payload_str = rest.split(":", 1)
                    import json as _json
                    payload = _json.loads(payload_str) if payload_str else {}
                    self._on_event({
                        "type": evt_type,
                        "job_id": jid,
                        "project_id": pid_,
                        "version_id": vid,
                        "kind": kind,
                        **payload,
                    })
                except Exception:
                    logger.exception("malformed event marker: %r", line[:200])
                return  # 不当成日志推

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

    # ----------------------------------------------- daemon 路径 (commit 9)
    def _submit_to_daemon(self, task: dict[str, Any]) -> None:
        """把一条 generate task 推给 inference daemon。

        和 _spawn_task 平行的入口；没有 _Slot 概念，daemon 自己管 active task。
        """
        import json as _json

        task_id = int(task["id"])
        cfg_path_str = task.get("config_path")
        cfg_path = Path(cfg_path_str) if cfg_path_str else None
        if cfg_path is None or not cfg_path.exists():
            self._fail_daemon_task(
                task_id, f"config not found: {cfg_path_str or '<none>'}",
            )
            return

        try:
            cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            self._fail_daemon_task(task_id, f"failed to read config: {e}")
            return

        # output_dir：cfg 里给的（enqueue_generate 写到 anima_gen_{tid}）兜底也行
        output_dir = (
            cfg.get("output_dir")
            or str(STUDIO_DATA / "monitors" / f"task_{task_id}")
        )

        # monitor_state.json：让 daemon 写文件，supervisor 起 poller 推 SSE
        monitor_state_path = _resolve_monitor_state_path(task)
        cfg["__monitor_state_file"] = str(monitor_state_path)

        daemon = get_daemon()
        if daemon.state == _DAEMON_STOPPED:
            try:
                daemon.start()
            except Exception as e:
                logger.exception("daemon start failed")
                self._fail_daemon_task(task_id, f"daemon start failed: {e}")
                return

        if not self._daemon_listener_registered:
            daemon.add_global_listener(self._on_daemon_global_event)
            self._daemon_listener_registered = True

        with self._daemon_lock:
            self._daemon_active_task_id = task_id
            self._daemon_cancel_pending = False

        # poller：daemon 写 monitor_state.json → SSE monitor_progress (增量协议)
        def _on_state_delta(delta: dict[str, Any]) -> None:
            self._on_event({
                "type": "monitor_progress",
                "task_id": task_id,
                "delta": delta,
            })

        self._daemon_state_poller = MonitorStatePoller(monitor_state_path, _on_state_delta)
        self._daemon_state_poller.start()

        with db.connection_for(self._db_path) as conn:
            db.update_task(
                conn,
                task_id,
                status="running",
                started_at=time.time(),
                monitor_state_path=str(monitor_state_path),
            )
        self._on_event({
            "type": "task_state_changed",
            "task_id": task_id,
            "status": "running",
        })

        try:
            daemon.submit_task(
                task_id=task_id,
                config=cfg,
                output_dir=output_dir,
                on_event=self._on_daemon_task_event,
            )
            logger.info("submitted generate task %d to daemon", task_id)
            self._emit_daemon_state()
        except Exception as e:
            logger.exception("daemon submit failed")
            self._on_daemon_task_event({
                "kind": "error",
                "task_id": task_id,
                "message": f"daemon submit failed: {e}",
            })

    def _on_daemon_task_event(self, event: dict[str, Any]) -> None:
        """daemon 推回的 task 级事件（image_done / done / error / preview_step）。"""
        kind = event.get("kind")
        tid = int(event.get("task_id") or 0)
        if kind == "started":
            self._emit_daemon_state()
            return
        if kind in ("image_done", "image_error"):
            return
        if kind == "preview_step":
            # commit 14：中间步进度 + 可选预览。step/total 永远有，image_b64
            # 取决于 settings.preview_every_n_steps + TAEFlux 是否可用
            self._on_event({
                "type": "generate_preview_step",
                "task_id": tid,
                "step": event.get("step"),
                "total": event.get("total"),
                "image_b64": event.get("image_b64"),
            })
            return
        if kind == "image_started":
            # 多张图（XY 或 count>1）：当前进度到第几张
            self._on_event({
                "type": "generate_image_started",
                "task_id": tid,
                "batch_idx": event.get("batch_idx"),
                "batch_total": event.get("batch_total"),
                "total_steps": event.get("total_steps"),
            })
            return
        if kind == "done":
            self._finalize_daemon_task(tid, status="done")
            self._emit_daemon_state()
        elif kind == "error":
            self._finalize_daemon_task(
                tid, status="failed", error_msg=str(event.get("message") or "daemon error"),
            )
            self._emit_daemon_state()

    def _on_daemon_global_event(self, event: dict[str, Any]) -> None:
        """daemon 进程级事件（loaded / unloaded / stopped）。"""
        kind = event.get("kind")
        if kind in ("loaded", "unloaded"):
            self._emit_daemon_state()
            return
        if kind == "stopped":
            with self._daemon_lock:
                tid = self._daemon_active_task_id
                cancel_pending = self._daemon_cancel_pending
            if tid is not None:
                if cancel_pending:
                    self._finalize_daemon_task(tid, status="canceled")
                else:
                    self._finalize_daemon_task(
                        tid, status="failed",
                        error_msg=f"daemon exited (rc={event.get('rc')})",
                    )
            self._emit_daemon_state()

    def _emit_daemon_state(self) -> None:
        """commit 13：广播 daemon 当前状态给 SSE 订阅者（前端 status pill）。"""
        daemon = get_daemon()
        with self._daemon_lock:
            active_tid = self._daemon_active_task_id
        try:
            self._on_event({
                "type": "daemon_state_changed",
                "state": daemon.state,
                "model_loaded": daemon.is_model_loaded,
                "busy": daemon.is_busy,
                "active_task_id": active_tid,
            })
        except Exception:
            logger.exception("emit daemon state failed")

    def _finalize_daemon_task(
        self,
        task_id: int,
        *,
        status: str,
        error_msg: Optional[str] = None,
    ) -> None:
        """daemon 上 task 终态收尾：标 db 状态 + 停 poller + 清 active 标记。

        commit 10 起：图本身在 server 内存 cache（非磁盘），不在这里清 ——
        让客户端断连 / LRU / lifespan 决定（commit 11）。这里只清 task
        在磁盘上的小附属物：
          - anima_gen_{tid}/config.json + 空目录
          - monitors/task_{tid}/state.json（如果 fallback 路径写过）
        """
        with self._daemon_lock:
            if self._daemon_active_task_id == task_id:
                self._daemon_active_task_id = None
                self._daemon_cancel_pending = False
            poller = self._daemon_state_poller
            self._daemon_state_poller = None
        if poller is not None:
            try:
                poller.stop()
            except Exception:
                pass

        fields: dict[str, Any] = {
            "status": status,
            "finished_at": time.time(),
            "pid": None,
        }
        if error_msg:
            fields["error_msg"] = error_msg
        with db.connection_for(self._db_path) as conn:
            db.update_task(conn, task_id, **fields)

        try:
            from .services.inference_core import cleanup_generate_tempdir
            cleanup_generate_tempdir(task_id)
        except Exception as e:
            logger.warning("cleanup generate tempdir failed: %s", e)

        self._on_event({
            "type": "task_state_changed",
            "task_id": task_id,
            "status": status,
        })
        logger.info("daemon task %d finished: %s", task_id, status)

    def _fail_daemon_task(self, task_id: int, msg: str) -> None:
        """generate task 在派给 daemon 之前的失败（config 缺失等）。"""
        with self._daemon_lock:
            if self._daemon_active_task_id == task_id:
                self._daemon_active_task_id = None
        with db.connection_for(self._db_path) as conn:
            db.update_task(
                conn, task_id,
                status="failed",
                started_at=time.time(),
                finished_at=time.time(),
                error_msg=msg,
            )
        self._on_event({
            "type": "task_state_changed",
            "task_id": task_id,
            "status": "failed",
        })

    # ---- 子进程通用 -----------------------------------------------------------
    def _popen(
        self,
        cmd: list[str],
        log_fp: Any,
        extra_env: Optional[dict[str, str]] = None,
    ) -> subprocess.Popen:
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
        try:
            wandb_cfg = _secrets.load().wandb
            if wandb_cfg.enabled:
                env.setdefault("WANDB_ENABLED", "1")
                env.setdefault("WANDB_MODE", wandb_cfg.mode)
                env.setdefault("WANDB_LOG_SAMPLES", "1" if wandb_cfg.log_samples else "0")
                env.setdefault("WANDB_SAMPLE_MAX_SIDE", str(wandb_cfg.sample_max_side))
                env.setdefault("WANDB_SAMPLE_EVERY_N_STEPS", str(wandb_cfg.sample_every_n_steps))
                if wandb_cfg.api_key:
                    env.setdefault("WANDB_API_KEY", wandb_cfg.api_key)
                if wandb_cfg.project:
                    env.setdefault("WANDB_PROJECT", wandb_cfg.project)
                if wandb_cfg.entity:
                    env.setdefault("WANDB_ENTITY", wandb_cfg.entity)
                if wandb_cfg.base_url:
                    env.setdefault("WANDB_BASE_URL", wandb_cfg.base_url)
        except Exception:
            logger.exception("failed to load wandb settings")
        if extra_env:
            env.update(extra_env)
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

        # ADR 0006 PR-2 三元分流（原来是二元 canceled vs done/failed）。
        # paused 优先级最高 — pause_pending=True 且子进程 emit 了 pause_state
        # （state_path / config_path 都到位）= 真正成功暂停。
        # 缺 pause_state_path = 子进程退出前没来得及 emit（IO 慢 / 异常 /
        # 强 kill）→ 降级 canceled（ADR §4.3 modal "强制取消保存进度" 兜底）。
        if slot.pause_pending and slot.pause_state_path:
            status = "paused"
        elif slot.cancel_pending:
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
                elif status == "paused":
                    fields["paused_state_path"] = slot.pause_state_path
                    fields["paused_config_path"] = slot.pause_config_path
                    fields["paused_step"] = slot.pause_step
                    fields["paused_at"] = time.time()
                db.update_task(conn, cid, **fields)
                # PP6.3：训练成功时回填 version.output_lora_path + 推 stage=done
                if status == "done":
                    _maybe_finalize_version(conn, cid)
            # commit 10 起：generate task 走 daemon 不进 SLOT_TRAIN，
            # 这条 _finish_slot 路径只跑 train / reg_ai；不再需要 generate
            # tempdir 清理（已搬到 _finalize_daemon_task）。
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
        """Cancel 软终止信号。

        ADR 0006 PR-2：Windows 不再发 CTRL_BREAK_EVENT — 跟 pause 信号撞
        （pause 占用 CTRL_BREAK_EVENT），cancel 的语义本来就是硬中断，
        直接走 taskkill /T /F。POSIX 没这个冲突，继续 SIGTERM。

        `_signal_terminate_async` 后续仍有 grace timer，Windows 上 proc 早就
        被 taskkill 杀掉了、grace 第一次 poll 就 return；不浪费时间。
        """
        try:
            if os.name == "nt":
                _kill_process_tree(proc.pid)
            else:
                proc.terminate()
        except Exception:
            logger.exception("send terminate signal failed")

    @staticmethod
    def _send_pause_signal(proc: subprocess.Popen) -> None:
        """Pause 软信号 — 子进程 handle_interrupt 接住保 state。

        Windows：`CTRL_BREAK_EVENT` 送达 CREATE_NEW_PROCESS_GROUP 子进程组，
        Python 端映射成 SIGBREAK（sig=21），由 resume phase 注册的 handler 捕获。
        POSIX：`SIGINT` — 跟 SIGTERM 分流，cancel 走 SIGTERM 不撞。

        Spike 报告 docs/design/queue-pause-spike-report.md 验证过链路通。
        """
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            logger.exception("send pause signal failed")

    def _signal_pause_async(self, slot: _Slot) -> None:
        """非阻塞：发暂停信号，不带 grace 强杀。

        跟 `_signal_terminate_async` 的关键差别：暂停**不超时降级**。
        ADR §4.3：30s 阈值由 UI 端 modal 决定下一步（再等 30s / 强制取消
        保存进度 / 终止任务），supervisor 不主动 kill 进程 — kill 决策由
        cancel API（用户从 modal 上选择后再调）下达。
        """
        if not slot.proc:
            return
        slot.pause_pending = True
        self._send_pause_signal(slot.proc)

    def _clear_pause_artifacts(self, task_id: int) -> None:
        """删 pause 文件对 + 清 db `paused_*` 字段（ADR §5.5）。

        调用点：
          - resume_state_loaded 事件（cmd_builder 成功 load 后）
          - cancel paused → canceled
          - 删除 paused task（未来 PR）

        故意 **不改 status** — caller 决定要写什么状态。文件 unlink 兜 missing_ok
        以容忍用户手动删 / 磁盘异常；db update 是 single transaction。
        """
        with db.connection_for(self._db_path) as conn:
            task = db.get_task(conn, task_id)
            if not task:
                return
            for col in ("paused_state_path", "paused_config_path"):
                p = task.get(col)
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except Exception:
                        logger.exception("failed to remove pause file %s", p)
            db.update_task(
                conn, task_id,
                paused_state_path=None,
                paused_config_path=None,
                paused_step=None,
                paused_at=None,
            )


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
