"""Supervisor 主类 — PR-4 从 supervisor.py 抽出（行为零变更）。

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

主类**不拆**（保 1100 行单类）：37 个 method 全部 read/write 共享 self
字段（`_slots / _daemon_* / _stop / _thread / _db_path`），状态耦合极高
且缺乏清晰子域边界 — 拆 Mixin/helper class 反而增加未来扩展成本（详
tmp/0.11.0_planning.md PR-4 决策日志）。叶子 helper（_Slot / 默认 cmd
builder / _maybe_finalize_version / _kill_process_tree）已搬到 sibling
模块，本文件仅保 Supervisor class 主体。
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .. import db, secrets as _secrets
from ..services import eval_auto, eval_validation
from ..services.runtime import xformers as _xformers_svc
from ..services.projects import jobs as project_jobs
from ..infrastructure.log_tail import LogTailer, MonitorStatePoller, clean_log_line
from ..infrastructure.logging import LOG_LEVEL_ENV, LOG_LINE_RE
from ..infrastructure.log_messages import UI_LANG_ENV
from ..paths import (
    LOGS_DIR,
    REPO_ROOT,
    STUDIO_DATA,
    STUDIO_DB,
    USER_PRESETS_DIR,
    task_dir,
    task_log_path,
)
from ..services.inference.daemon import (
    InferenceDaemon,
    STATE_STOPPED as _DAEMON_STOPPED,
    get_daemon,
)
from .resources import (
    RESOURCE_EXCLUSIVE,
    RESOURCE_LIGHT,
    job_resource_class,
)
from .cmd_builder import (
    _EVENT_MARKER,
    CmdBuilder,
    EventCallback,
    JobCmdBuilder,
    _default_cmd_builder,
    _default_job_cmd_builder,
    _resolve_monitor_state_path,
)
from .finalizer import _maybe_finalize_version
from .process import _kill_process_tree
from .slot import SLOT_DATA, SLOT_TRAIN, _Slot

logger = logging.getLogger(__name__)


class _ErrorThrottle:
    """轮询站点错误节流（R8）：首条全文 exception，之后同异常类型 60s 一条
    计数 WARNING，异常类型变化重新全文记一次，恢复时一条 INFO。

    supervisor 的 _poll=1.0s，持续故障下逐条 exception 是 3600 条带
    traceback/小时 × 站点数，会吃穿 studio.log 配额并把诊断包冲成噪音。
    """

    def __init__(self, site: str, window: float = 60.0) -> None:
        self._site = site
        self._window = window
        self._count = 0
        self._last_type: Optional[str] = None
        self._last_report = 0.0

    def failed(self) -> None:
        exc = sys.exc_info()[1]
        exc_type = type(exc).__name__ if exc is not None else "?"
        now = time.monotonic()
        self._count += 1
        if exc_type != self._last_type:
            self._last_type = exc_type
            self._last_report = now
            logger.exception("%s failed", self._site)
        elif now - self._last_report >= self._window:
            self._last_report = now
            logger.warning(
                "%s still failing: count=%d err=%s",
                self._site, self._count, exc_type,
            )

    def recovered(self) -> None:
        if self._count:
            logger.info(
                "%s recovered after %d failures", self._site, self._count,
            )
            self._count = 0
            self._last_type = None



_ERROR_MSG_TAIL_BYTES = 256 * 1024


def _parse_event_marker(line: str) -> tuple[str, dict[str, Any]]:
    """`__EVENT__:type:json` → (type, payload)；格式不对抛异常由 caller 处理。"""
    rest = line[len(_EVENT_MARKER):]
    evt_type, payload_str = rest.split(":", 1)
    payload = json.loads(payload_str) if payload_str else {}
    if not isinstance(payload, dict):
        raise ValueError("event payload must be a JSON object")
    return evt_type, payload


def _tail_log_for_error_msg(log_path: Path, max_lines: int = 12, max_chars: int = 800) -> str:
    """失败 task 的 db.error_msg：从 run.log 尾部取「最后一个错误块」。

    行契约（LOG_LINE_RE）落地后 run.log 每条记录有行头，续行（traceback）无前缀：
      1. 找最后一条 ERROR/CRITICAL 记录（行头 + 其续行，去掉行头前缀）；
      2. 找最后一个裸 `Traceback` 行（子进程未捕获异常直接打到 stderr，不经 logger）；
      3. 两者取位置靠后的那个；都没有则退回末 N 行。
    只读文件尾 256KB（长训练 run.log 上百 MB 不能整读）；截断到 max_chars 适配 UI。
    失败兜底返 ""（caller 用 "exit code N" 默认值）。
    """
    try:
        if not log_path.exists():
            return ""
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if size > _ERROR_MSG_TAIL_BYTES:
                f.seek(size - _ERROR_MSG_TAIL_BYTES)
                f.readline()  # 丢掉切在中间的半行
            lines = [clean_log_line(b) for b in f.read().split(b"\n")]
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return ""
        err_start = tb_start = None
        for i in range(len(lines) - 1, -1, -1):
            m = LOG_LINE_RE.match(lines[i])
            if m and err_start is None and m["level"] in ("ERROR", "CRITICAL"):
                err_start = i
            if tb_start is None and lines[i].startswith("Traceback"):
                tb_start = i
            if err_start is not None and tb_start is not None:
                break
        err_end = None
        if err_start is not None:
            # 记录块 = 行头（去前缀只留 msg）+ 到下一条记录行头之前的续行
            err_end = err_start + 1
            while err_end < len(lines) and not LOG_LINE_RE.match(lines[err_end]):
                err_end += 1
        # 裸 Traceback 若落在 ERROR 记录块内（logger.exception 的续行）就属于该块，
        # 不单拎；只有块外更靠后的裸 Traceback（子进程未捕获异常）才盖过记录块
        if err_start is not None and (tb_start is None or tb_start < err_end):
            head = LOG_LINE_RE.match(lines[err_start])
            snippet_lines = [head["msg"] if head else lines[err_start], *lines[err_start + 1:err_end]]
        elif tb_start is not None:
            snippet_lines = lines[tb_start:]
        else:
            snippet_lines = lines[-max_lines:]
        out = "\n".join(snippet_lines).strip()
        if len(out) > max_chars:
            out = "..." + out[-(max_chars - 3):]
        return out
    except Exception:
        logger.warning(
            "tail run log failed: path=%s; error_msg stays empty for this task",
            log_path, exc_info=True,
        )
        return ""


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
        # 轮询站点错误节流（R8），各站点独立计数
        self._tick_throttle = _ErrorThrottle("supervisor tick")
        self._promote_throttle = _ErrorThrottle("promote_due_scheduled")
        self._queue_held_throttle = _ErrorThrottle("read queue_held")
        self._unload_throttle = _ErrorThrottle("daemon unload request")
        # 启动对账走到哪一步（reconcile 失败时进日志，见 _reconcile_orphans）
        self._reconcile_stage = "init"
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
        # 0.17 item1：generate task 走 daemon 无 run.log → LogTab 空。派活时开该 task
        # 的 run.log，把 daemon 在其运行期间的日志落盘 + emit task_log_appended（daemon
        # 串行跑一个，归属清晰）；finalize 时关。log 线程与 supervisor 线程都碰它，
        # 一律在 _daemon_lock 下访问。
        self._daemon_log_fp: Optional[Any] = None

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
            logger.warning(
                "inference daemon stop failed during shutdown; the supervisor "
                "exits anyway", exc_info=True,
            )
        if self._thread:
            self._thread.join(timeout=timeout)

    def _find_slot(self, *, kind: str, id: int) -> Optional[_Slot]:
        for slot in self._slots:
            if slot.kind == kind and slot.id == id:
                return slot
        return None

    def cancel(self, task_id: int) -> bool:
        """取消 task：pending/scheduled → status=canceled；running → 异步发信号立即返回。

        ADR 0006 PR-2：paused task 也可被取消，状态从 paused 直接改 canceled。
        ADR Addendum 2：恢复点文件保留（canceled 之后仍可 resume），只清
        paused_* 字段。

        异步路径关键：**不阻塞 web 请求线程**。supervisor 主循环会自然 poll
        proc.poll() 拿到退出码并走 `_finish_slot` 流程，把 status 写为
        canceled。后台 grace timer 在 30s 后还没退就强杀整棵进程树。
        """
        with db.connection_for(self._db_path) as conn:
            task = db.get_task(conn, task_id)
            if not task:
                return False
            if task["status"] in ("pending", "scheduled"):
                db.update_task(
                    conn, task_id, status="canceled", finished_at=time.time()
                )
                self._on_event(
                    {"type": "task_state_changed", "task_id": task_id, "status": "canceled"}
                )
                return True
        if task["status"] == "paused":
            # 进程已退出，无需发信号 — 清 paused_* 字段再单独写 status=canceled
            # + finished_at。ADR Addendum 2：恢复点文件保留，canceled 后仍可
            # resume（last_state_* 字段还在）。
            # 故意走 with 块外：_clear_pause_fields 内部开自己 conn，避免嵌套。
            self._clear_pause_fields(task_id)
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
            # R-5：台账合并后 running 的可能是数据作业（DATA 槽 kind="job"）——
            # 统一从这个入口取消，SIGTERM 语义同 cancel_job。
            slot = (
                self._find_slot(kind="task", id=task_id)
                or self._find_slot(kind="job", id=task_id)
            )
            if slot is not None:
                self._signal_terminate_async(slot)
                return True
            with self._daemon_lock:
                is_daemon_task = self._daemon_active_task_id == task_id
                if is_daemon_task:
                    self._daemon_cancel_pending = True
            if is_daemon_task:
                if get_daemon().cancel_active_task(task_id):
                    return True
                logger.warning(
                    "daemon cancel request missed: task_id=%s; the task may keep "
                    "running", task_id,
                )
            return True
        return False

    def is_task_pausable(self, task_id: int) -> bool:
        """ADR §8.1 + Addendum 1: UI is_pausable 信号。

        条件：task 在 slot 上 running、`train_loop_started` 事件已收到、
        **`last_auto_epoch_state_path` 已设置**（即首个 epoch 已写完 auto backup）、
        没有 pause / cancel pending。任一不满足 → UI 应隐藏暂停按钮。

        ADR 0006 Addendum 1：首 epoch 未结束时禁用 pause 是关键防护 —— 没有
        auto_epoch_state.pt 时按 pause 会让 supervisor 走 cancel 兜底，无可恢复进度，
        UI 端直接隐藏按钮避免误操作。
        """
        slot = self._find_slot(kind="task", id=task_id)
        if slot is None:
            return False
        return (
            slot.proc is not None
            and slot.train_loop_started
            and slot.last_auto_epoch_state_path is not None
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
            logger.exception(
                "startup reconcile failed: stage=%s; orphan tasks may stay in the "
                "running state until the next restart", self._reconcile_stage,
            )
        while not self._stop.is_set():
            try:
                self._tick()
                self._tick_throttle.recovered()
            except Exception:
                self._tick_throttle.failed()
            self._stop.wait(self._poll)

    def _reconcile_orphans(self) -> None:
        # ADR 0006 PR-2 兼容性 note：此处 list_tasks(status="running") 精确按
        # status 过滤，paused task（status='paused'）天然不进 this loop —
        # 跨 supervisor 重启的 paused task 保持状态不变（ADR §8.4）。
        # 失败时上面那条 startup reconcile 行要说清卡在哪一步（G 清单「就近来源 —」）
        self._reconcile_stage = "tasks"
        orphan_tasks = 0
        with db.connection_for(self._db_path) as conn:
            for t in db.list_tasks(conn, status="running"):
                orphan_tasks += 1
                logger.debug("orphan running task marked failed: task_id=%d", t["id"])
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
            self._reconcile_stage = "jobs"
            n = project_jobs.cleanup_orphan_running(conn)
        # 上次进程没干净退出的证据 —— 任务与作业合成一条，不再逐个孤儿刷屏
        if orphan_tasks or n:
            logger.warning(
                "orphan running tasks and jobs marked failed on startup: "
                "tasks=%d jobs=%d (the previous process did not exit cleanly)",
                orphan_tasks, n,
            )
        self._reconcile_stage = "done"

    def _tick(self) -> None:
        # 0) 0.17 P-B：到点的 scheduled task 提升为 pending，让下面的 dispatch
        #    看得见。不看 queue_held —— hold 语义是"停派活"，提升只是状态澄清，
        #    提升后的 pending 照样被 hold 拦住。
        self._promote_due_scheduled()

        # 1) 先收尸：所有 busy 槽位 poll 一遍，退出的走 _finish_slot
        for slot in self._slots:
            if not slot.busy:
                continue
            assert slot.proc is not None
            rc = slot.proc.poll()
            if rc is not None:
                self._finish_slot(slot, rc)

        # 2) 给空闲槽位派活（按槽位职责分工）。R-1：generate 并入 exclusive
        #    统一派发（同表 FIFO + 集中准入），不再有独立的第 3 步。
        for slot in self._slots:
            if slot.busy:
                continue
            if slot.name == SLOT_TRAIN:
                self._dispatch_exclusive_tasks(slot)
            elif slot.name == SLOT_DATA:
                self._dispatch_data(slot)

    def _promote_due_scheduled(self) -> None:
        """0.17 P-B：scheduled_at 到点的 task → pending + publish 状态事件。"""
        try:
            with db.connection_for(self._db_path) as conn:
                promoted = db.promote_due_scheduled(conn)
            self._promote_throttle.recovered()
        except Exception:
            self._promote_throttle.failed()
            return
        for tid in promoted:
            logger.info("scheduled task due: task_id=%d → pending", tid)
            self._on_event(
                {"type": "task_state_changed", "task_id": tid, "status": "pending"}
            )

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

    # ---- R-1 资源档位准入（docs/design/queue-resource-model-0.17.md §3） ----

    def _daemon_active(self) -> bool:
        """daemon 是否有 active generate task（提交后到 finalize 前）。"""
        with self._daemon_lock:
            return self._daemon_active_task_id is not None

    def _data_slot_exclusive_busy(self) -> bool:
        """DATA 槽是否正在跑 exclusive 档 job（eval_samples，底模级显存）。"""
        for slot in self._slots:
            if (
                slot.name == SLOT_DATA and slot.busy
                and slot.job_kind is not None
                and job_resource_class(slot.job_kind) == RESOURCE_EXCLUSIVE
            ):
                return True
        return False

    def _exclusive_busy(self) -> bool:
        """全系统是否有 exclusive 档工作在跑（同时最多 1 个的准入前提）。

        三个执行位逐一检查：TRAIN 槽（train/reg_ai）、daemon（active generate）、
        DATA 槽（eval_samples）。修 L1（generate 与训练互斥缺后端守卫）/
        L2（训练不躲正在跑的 eval_samples）的共同根。
        """
        return (
            self._train_busy()
            or self._daemon_active()
            or self._data_slot_exclusive_busy()
        )

    def _dispatch_exclusive_tasks(self, slot: _Slot) -> None:
        """exclusive 档统一派发（tasks 表：train / reg_ai / generate 同表 FIFO）。

        D-R3 平级 FIFO：三类之间无优先级，按 `priority DESC, created_at ASC`
        取队首；running 永不被中断。路由：train/reg_ai → TRAIN 槽子进程；
        generate → daemon（daemon 是 exclusive 档的执行器之一，不是独立车道）。

        eval_session（一次评估一个作业，#465）也在这条 FIFO 里：它先出图再算指标，
        整段占底模栈，跟 train / generate 同规格竞争。执行位在 DATA 槽（worker 子进程）。

        ADR 0006 PR-2：queue_held=True 时跳过本次派发（ADR §3.2）。
        """
        if self._queue_held():
            return
        if self._exclusive_busy():
            return
        task = self._next_pending_task_in(
            ("train", "reg_ai", "generate", "eval_session")
        )
        if task is None:
            return
        ttype = task.get("task_type") or "train"
        if ttype == "generate":
            # enqueue_generate 先 create_task(pending) 再写 config.json 落
            # config_path —— 两步之间这条 task 已 pending 但 config_path 还是
            # NULL。此时别提交（daemon 会报 "config not found"），等下个 tick。
            # FIFO 语义：不越过它取后面的任务（窗口 <1s）。
            if not task.get("config_path"):
                return
            self._submit_to_daemon(task)
            return
        if ttype == "eval_session":
            # exclusive 档数据作业。排队语义与 train/generate 同一 FIFO（D-R3 跨类型
            # 平级），执行位在 DATA 槽（worker 子进程）。DATA 槽被 light 作业占着时
            # 等它结束（light 都是短任务），不越队。
            data_slot = next(
                (s for s in self._slots if s.name == SLOT_DATA), None
            )
            if data_slot is None or data_slot.busy:
                return
            if self._maybe_yield_daemon():
                return
            self._spawn_job(data_slot, project_jobs.as_job(dict(task)) or task)
            return
        # train / reg_ai：daemon 常驻模型是 exclusive 租约，spawn 前必须吊销
        # （unload 释放 VRAM）。daemon 在跑 generate 的情况已被 _exclusive_busy
        # 拦下，这里只处理 idle-but-loaded 的租约。
        if self._maybe_yield_daemon():
            return  # daemon 还占 VRAM，等下次 tick 派
        self._spawn_task(slot, task)

    def _queue_held(self) -> bool:
        """ADR §3.2 queue hold 开关，跨 supervisor 重启保留（db kv）。"""
        try:
            with db.connection_for(self._db_path) as conn:
                held = db.get_queue_held(conn)
            self._queue_held_throttle.recovered()
            return held
        except Exception:
            self._queue_held_throttle.failed()
            return False  # 读失败默认放行，安全降级

    def _maybe_yield_daemon(self) -> bool:
        """daemon 占着 VRAM → 触发 unload，调用方应跳过这次派发。

        R-1：daemon 常驻模型 = exclusive 租约。要派 exclusive 档工作
        （train / reg_ai / eval_samples）前必须吊销租约——**不再受任何开关
        豁免**（老 allow_gpu_during_train 会放行「训练 + 常驻底模」并存，
        是 L3 的一部分）。light 档开关关闭时的保守路径也复用本函数。

        返回值：
          - True：daemon 还占着 VRAM（在跑 generate 或刚发了 unload 请求），
                  调用方不应该派，等下次 tick 重检
          - False：daemon 没占 GPU（未起 / 已 unloaded），可立刻派
        """
        daemon = get_daemon()
        if not daemon.is_model_loaded:
            return False
        if daemon.is_busy:
            # 用户主动触发的 generate 不强中断；等它跑完
            return True
        try:
            daemon.request_unload()
            logger.debug("requested daemon unload to yield GPU")
            self._unload_throttle.recovered()
        except Exception:
            self._unload_throttle.failed()
        return True

    def _dispatch_data(self, slot: _Slot) -> None:
        """DATA 槽：跑 project_jobs。R-1 按资源档位准入（修 L2/L3）：

        - io（download）：恒放行（仅受 queue_held 约束）
        - light（tag / preprocess / reg_build）：无 exclusive 运行时恒放行（daemon
          idle 常驻模型无碍——小模型体量）；有 exclusive 运行时看
          `queue.light_tasks_during_train`（默认开）。开关**关闭**时保守等同
          旧默认：额外要求 daemon 租约已释放
        - exclusive（eval_session，底模级）：不在这里派——见
          `_dispatch_exclusive_tasks` 的统一 FIFO

        ADR 0006 PR-2：queue_held=True 时跳过本次派发，包含 download。语义上
        hold 是"全队列暂停新派活"，不区分档位。
        """
        if self._queue_held():
            return
        exclusive_busy = self._exclusive_busy()
        light_parallel = self._light_tasks_during_train()
        with db.connection_for(self._db_path) as conn:
            pending = project_jobs.list_pending_fifo(conn)
        for job in pending:
            cls = job_resource_class(job["kind"])
            if cls == RESOURCE_EXCLUSIVE:
                # eval_session 走 exclusive 统一 FIFO（_dispatch_exclusive_tasks
                # 与 train/generate 平级排队），本函数只管 light + io。
                continue
            if cls == RESOURCE_LIGHT:
                if exclusive_busy and not light_parallel:
                    continue
                if not light_parallel and self._maybe_yield_daemon():
                    continue  # 保守模式：等 daemon 卸载
            self._spawn_job(slot, job)
            return

    def _train_busy(self) -> bool:
        for slot in self._slots:
            if slot.name == SLOT_TRAIN and slot.busy:
                return True
        return False

    def _light_tasks_during_train(self) -> bool:
        """R-1：exclusive 运行时是否放行 light 档（默认开；读失败取 schema 默认）。"""
        try:
            return bool(_secrets.load().queue.light_tasks_during_train)
        except Exception:
            return False

    # -------------------------------------------------------------- 子进程
    def _spawn_task(self, slot: _Slot, task: dict[str, Any]) -> None:
        # ADR-0009 PR-1 C6 trace_id 跨进程贯穿：
        #   1) task.request_trace_id 由 API endpoint 入 task 时存（HTTP 请求那一刻
        #      TraceIdMiddleware bind 的 contextvar）；老 task / 没存的兜底 bg-{uuid}
        #   2) bind 到 ContextVar 让 supervisor 整段 _spawn_task 内 logger.x 都带
        #   3) 注入 ANIMA_TRACE_ID / ANIMA_PROCESS_NAME env 给 worker 子进程
        from ..infrastructure.logging import (
            PROCESS_ENV, TRACE_ENV,
            bind_trace_id, new_trace_id, reset_trace_id,
        )
        trace_id = task.get("request_trace_id") or f"bg-{new_trace_id()}"
        kind = task.get("task_type") or "train"
        process_name = f"worker:{kind}/{task['id']}"
        _trace_token = bind_trace_id(trace_id)
        try:
            cfg_path = self._resolve_task_config_path(task)
            if not cfg_path.exists():
                self._fail_task_config_missing(task, cfg_path)
                return

            self._freeze_task_snapshot(int(task["id"]), cfg_path)

            # task-scoped 档案：monitor state 一律落 tasks/<id>/monitor/state.json，
            # 跟 version 解耦（之前在 versions/<label>/monitor/task_<id>/state.json，
            # 删 version 会一并丢掉 task 历史）
            monitor_state_path = _resolve_monitor_state_path(task)
            # 提前注入到 task dict 供 cmd_builder 用，以及落库
            task = dict(task)
            task["monitor_state_path"] = str(monitor_state_path)

            # task-scoped 档案：日志落 tasks/<id>/run.log，跟 monitor / samples /
            # snapshot 同根。老 task 跑过的 studio_data/logs/<id>.log 由 logs.py
            # fallback 读，不再写新文件到那。
            log_path = task_log_path(task["id"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_path, "wb")

            # 训练前：若 task 启用了验证集指标且设了分隔比例，从 train/ 划 held-out
            # 到 validation/（移动，不参与训练）。失败不阻断训练，只记日志。
            self._maybe_split_validation(task, cfg_path, log_fp)

            cmd = self._cmd_builder(task, cfg_path)
            # ADR 0006 PR-1：LORA_TASK_ID 注入让训练子进程把用户周期 save 写到
            # output_dir/state/task_<TID>/ 子目录，避免同 version 多 task 互覆盖。
            # Addendum 2 起 auto_epoch_state.pt 改落 task 档案 tasks/<id>/state/
            # —— 路径由子进程从 --monitor-state-file 推出（bootstrap，同 samples/）。
            # ADR-0009 PR-1 C6：TRACE_ENV + PROCESS_ENV 让 worker bootstrap 拿到。
            proc = self._popen(cmd, log_fp, extra_env={
                "LORA_TASK_ID": str(task["id"]),
                TRACE_ENV: trace_id,
                PROCESS_ENV: process_name,
            })

            slot.proc = proc
            slot.kind = "task"
            slot.id = task["id"]
            slot.log_fp = log_fp
            slot.cancel_pending = False

            tid = task["id"]

            # PP6.4 — log tail → SSE（取代前端 2s 轮询 /api/logs/{id}）
            slot.tailer = LogTailer(log_path, self._make_task_log_callback(slot, tid))
            slot.tailer.start()

            # PP6.4 → PR #37: monitor_state.json 变化 → SSE monitor_progress (增量协议)
            slot.state_poller = MonitorStatePoller(
                monitor_state_path, self._make_monitor_callback(tid)
            )
            slot.state_poller.start()

            self._write_task_running_to_db(task, proc.pid, monitor_state_path)

            self._on_event(
                {
                    "type": "task_state_changed",
                    "task_id": task["id"],
                    "status": "running",
                }
            )
            logger.info(
                "started task: task_id=%d slot=%s pid=%d",
                task["id"], slot.name, proc.pid,
            )
        finally:
            reset_trace_id(_trace_token)

    def _resolve_task_config_path(self, task: dict[str, Any]) -> Path:
        """PP6.3：优先用 task.config_path（version 私有 config 绝对路径）；
        没有时降级到老路径 _configs_dir / {config_name}.yaml。
        """
        explicit_cfg = task.get("config_path")
        if explicit_cfg:
            return Path(explicit_cfg)
        return self._configs_dir / f"{task['config_name']}.yaml"

    def _maybe_split_validation(
        self, task: dict[str, Any], cfg_path: Path, log_fp: Any
    ) -> None:
        """训练前把 held-out 验证集从 train/ 划到 validation/（移动）。

        仅当 task 启用 eval_validation 且 ratio>0 时生效；按比例补足、够了不动、
        永不移回。失败只记日志，不阻断训练。
        """
        try:
            with db.connection_for(self._db_path) as conn:
                summary = eval_validation.split_for_task(conn, task, cfg_path)
        except Exception:
            logger.warning(
                "validation split failed: task_id=%s; training continues without "
                "a validation set", task.get("id"), exc_info=True,
            )
            return
        if summary and summary.get("moved"):
            try:
                log_fp.write(
                    f"[eval-validation] moved {summary['moved']} image(s) to "
                    f"validation/ (train={summary['train']}, "
                    f"validation={summary['validation']})\n".encode("utf-8")
                )
                log_fp.flush()
            except Exception:
                pass

    def _fail_task_config_missing(
        self, task: dict[str, Any], cfg_path: Path
    ) -> None:
        """config 不存在时把 task 标 failed 并 publish 事件。"""
        explicit_cfg = task.get("config_path")
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

    def _freeze_task_snapshot(self, task_id: int, cfg_path: Path) -> None:
        """ADR-0007 §11.7 / PR-3 commit 4：task 启动 → 冻结当时的 config
        到 studio_data/tasks/{tid}/snapshot/config.yaml。失败不阻 task
        启动（snapshot 是 forensics 不是必需）。
        """
        try:
            from ..services import task_snapshot
            task_snapshot.freeze_config(task_id, cfg_path)
        except Exception:
            logger.warning(
                "config snapshot freeze failed: task_id=%s; the task runs against "
                "the live config", task_id, exc_info=True,
            )

    def _make_task_log_callback(
        self, slot: _Slot, tid: int
    ) -> Callable[[str, int], None]:
        """LogTailer 回调：识别 __EVENT__: 协议 → 镜像状态到 slot + publish SSE；
        普通行 → task_log_appended（带 seq + end_offset，前端断线按 end_offset 补拉）。

        ADR 0006 PR-2：训练 worker 通过 __EVENT__: 协议跟 supervisor 通信
        （pause_state / train_loop_started / auto_epoch_backup_written /
        resume_state_loaded）。跟 jobs 的 _on_line 路径对齐。
        """
        def _on_task_log(line: str, end_offset: int) -> None:
            if line.startswith(_EVENT_MARKER):
                try:
                    evt_type, payload = _parse_event_marker(line)
                except Exception:
                    # B-4.4: malformed event 静默丢导致 UI pause_state 永远收不到
                    # → 暂停按钮永远灰。logger.exception 进 studio.log；
                    # SSE event_malformed 让前端可见（不阻断 task）。
                    logger.warning(
                        "malformed event marker on the task stream: "
                        "task_id=%s raw=%r", tid, line[:200], exc_info=True,
                    )
                    self._on_event({
                        "type": "event_malformed",
                        "task_id": tid,
                        "raw_preview": line[:200],
                    })
                    return  # 不当 log 推
                # 状态机镜像（ADR §8.1 / §`_on_line` / Addendum 1 §supervisor）
                if evt_type == "pause_state":
                    # ADR Addendum 1 方案 Δ：state_path 为 None / 空 = 首 epoch 内暂停
                    # → 走 _finish_slot 的 cancel 分支（pause_state_path 空 → 降级 canceled）。
                    slot.pause_state_path = str(payload.get("state_path") or "")
                    slot.pause_config_path = str(payload.get("config_path") or "")
                    slot.pause_step = payload.get("step")
                elif evt_type == "train_loop_started":
                    slot.train_loop_started = True
                elif evt_type == "auto_epoch_backup_written":
                    # ADR 0006 Addendum 1：每 epoch 末 loop.py emit 一次 → 标记 slot
                    # 字段 → is_pausable 升级条件满足 → SSE 解锁 UI 暂停按钮。
                    slot.last_auto_epoch_state_path = str(payload.get("state_path") or "") or None
                    slot.last_auto_epoch_config_path = str(payload.get("config_path") or "") or None
                    # ADR Addendum 2：同步落 DB。slot 字段是内存态，进程 / 机器
                    # 一死即丢；落 DB 后 task 之后 failed（崩溃 / 关机）或
                    # canceled 时恢复点路径仍可查，resume endpoint 据此放行。
                    self._persist_last_state(tid, payload)
                elif evt_type == "resume_state_loaded":
                    # ADR §5.5 / PR-3：训练子进程 load_training_state 成功 →
                    # paused_* 字段已被消费完，清 db 字段避免 stale。
                    # ADR Addendum 2：**不再删文件** —— auto_epoch_state.pt 是
                    # 覆盖式单文件不会堆积，删了反而造成「resume 后到下一
                    # epoch 末之间无恢复点」的窗口。
                    self._clear_pause_fields(tid)
                elif evt_type == "eval_training_finished":
                    slot.eval_training_finished_payload = dict(payload)
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
                "end_offset": end_offset,
            })
        return _on_task_log

    def _queue_auto_eval_after_training(
        self, tid: int, payload: dict[str, Any]
    ) -> None:
        """训练完成 → 建一个 EvalSession（#465：不再逐 checkpoint fan-out 子作业）。"""
        try:
            with db.connection_for(self._db_path) as conn:
                task = db.get_task(conn, tid)
                if not task:
                    return
                session = eval_auto.queue_training_finished_eval(conn, task, payload)
        except Exception:
            logger.warning(
                "after-training auto eval enqueue failed: task_id=%s; training "
                "finished and was saved, no evaluation was queued", tid, exc_info=True,
            )
            return
        if not session:
            return
        plan = session.get("plan") or {}
        candidates = len(plan.get("candidates") or [])
        if plan.get("baseline", {}).get("enabled"):
            candidates += 1
        self._on_event({
            "type": "eval_auto_after_training_queued",
            "task_id": tid,
            "session_id": session.get("id"),
            "eval_task_id": session.get("task_id"),
            "count": candidates,
        })

    def _make_monitor_callback(
        self, tid: int
    ) -> Callable[[dict[str, Any]], None]:
        """MonitorStatePoller 回调：把 monitor_state.json 的 delta publish 成
        SSE monitor_progress（PR #37 增量协议）。

        payload 是 delta（appended_losses/lr/samples + 最新 step/speed/...），
        客户端首次 GET /api/state 拿快照后用这个增量持续 merge。
        """
        def _on_state_delta(delta: dict[str, Any]) -> None:
            self._on_event({
                "type": "monitor_progress",
                "task_id": tid,
                "delta": delta,
            })
        return _on_state_delta

    def _write_task_running_to_db(
        self, task: dict[str, Any], pid: int, monitor_state_path: Path
    ) -> None:
        """task spawn 后的 db 写入：task.status=running + version.status=training
        （ADR-0007 §11.3-B 双写）。
        """
        with db.connection_for(self._db_path) as conn:
            db.update_task(
                conn,
                task["id"],
                status="running",
                started_at=time.time(),
                pid=pid,
                monitor_state_path=str(monitor_state_path),
            )
            vid = task.get("version_id")
            if vid:
                try:
                    from ..services.projects import versions as _versions
                    _versions.update_version(
                        conn, int(vid),
                        status=_versions.VersionStatus.TRAINING,
                    )
                except Exception:
                    logger.warning(
                        "version status write failed: task_id=%s; the status is "
                        "corrected on the next read", task["id"], exc_info=True,
                    )

    def _spawn_job(self, slot: _Slot, job: dict[str, Any]) -> None:
        log_path = Path(job.get("log_path") or project_jobs.log_path_for(job["id"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # worker 自己 append 模式开 log，supervisor 这里只挂个 stdout 转发到同一文件
        log_fp = open(log_path, "ab")

        # trace / process 注入与 _spawn_task 对齐：jobs 表没有 request_trace_id 列，
        # 用 bg-{uuid} 兜底，至少让 worker 侧 studio.log / run.log 的 trace_id 与
        # supervisor 这段 spawn 日志对得上（之前 job 路径完全不注入，worker 只能自造）。
        from ..infrastructure.logging import (
            PROCESS_ENV, TRACE_ENV, bind_trace_id, new_trace_id, reset_trace_id,
        )
        trace_id = job.get("request_trace_id") or f"bg-{new_trace_id()}"
        _trace_token = bind_trace_id(trace_id)
        try:
            cmd = self._job_cmd_builder(job)
            proc = self._popen(cmd, log_fp, extra_env={
                TRACE_ENV: trace_id,
                PROCESS_ENV: f"worker:{job['kind']}/{job['id']}",
            })
        finally:
            reset_trace_id(_trace_token)

        with db.connection_for(self._db_path) as conn:
            project_jobs.mark_running(conn, job["id"], pid=proc.pid)

        slot.proc = proc
        slot.kind = "job"
        slot.id = job["id"]
        slot.job_kind = job["kind"]  # R-1：供 _exclusive_busy 判档位
        slot.log_fp = log_fp
        slot.cancel_pending = False

        jid = job["id"]
        pid_ = job["project_id"]
        vid = job.get("version_id")
        kind = job["kind"]

        slot.tailer = LogTailer(
            log_path, self._make_job_log_callback(jid, pid_, vid, kind)
        )
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
            "started job: job_id=%d slot=%s kind=%s pid=%d",
            jid, slot.name, kind, proc.pid,
        )

    def _make_job_log_callback(
        self,
        jid: int,
        pid_: Optional[int],
        vid: Optional[int],
        kind: str,
    ) -> Callable[[str, int], None]:
        """LogTailer 回调：识别 __EVENT__: 协议 publish typed SSE；普通行
        → job_log_appended（带 seq + end_offset）。

        结构化事件标记：worker 写 `__EVENT__:type:json_payload` 让 supervisor
        publish 成 typed SSE 事件（不进 job log）。比专门搭 IPC 通道轻，比
        让前端按文本 grep 日志靠谱。job_id / project_id 由 supervisor 注入。
        """
        def _on_line(line: str, end_offset: int) -> None:
            if line.startswith(_EVENT_MARKER):
                # try 只包解析：广播自身抛错不能被误报成 malformed marker
                try:
                    evt_type, payload = _parse_event_marker(line)
                except Exception:
                    # 与 task 路径对齐（B-4.4 之前只落了一半）：进 studio.log +
                    # SSE event_malformed 让前端可见
                    logger.warning(
                        "malformed event marker on the job stream: "
                        "job_id=%s project_id=%s raw=%r",
                        jid, pid_, line[:200], exc_info=True,
                    )
                    self._on_event({
                        "type": "event_malformed",
                        "job_id": jid,
                        "project_id": pid_,
                        "version_id": vid,
                        "kind": kind,
                        "raw_preview": line[:200],
                    })
                    return
                self._on_event({
                    "type": evt_type,
                    "job_id": jid,
                    "project_id": pid_,
                    "version_id": vid,
                    "kind": kind,
                    **payload,
                })
                return  # 不当成日志推

            self._on_event({
                "type": "job_log_appended",
                "job_id": jid,
                "project_id": pid_,
                "version_id": vid,
                "kind": kind,
                "text": line,
                "seq": next(self._log_seq),
                "end_offset": end_offset,
            })
        return _on_line

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
                logger.exception(
                    "inference daemon start failed: task_id=%s; the task is "
                    "marked failed", task_id,
                )
                self._fail_daemon_task(task_id, f"daemon start failed: {e}")
                return

        # spawn 后立刻把 idle timeout 从 secrets 同步进 daemon，避免首个 task 出图后
        # 模型常驻；用户在 settings 改值后下一次 dispatch 也会重新读取。
        daemon.sync_idle_timeout_from_secrets()

        if not self._daemon_listener_registered:
            daemon.add_global_listener(self._on_daemon_global_event)
            daemon.add_log_listener(self._on_daemon_log_line)
            self._daemon_listener_registered = True

        with self._daemon_lock:
            self._daemon_active_task_id = task_id
            self._daemon_cancel_pending = False
            # 0.17 item1：开该 generate task 的 run.log（LogTab 读 /api/logs →
            # tasks/<id>/run.log）。daemon 串行跑一个，其间的日志都归这个 task。
            lp = None
            try:
                lp = task_log_path(task_id)
                lp.parent.mkdir(parents=True, exist_ok=True)
                self._daemon_log_fp = open(lp, "ab")
            except Exception:
                logger.warning(
                    "open daemon task log failed: task_id=%s path=%s; this task "
                    "has no run.log, generation continues",
                    task_id, lp, exc_info=True,
                )
                self._daemon_log_fp = None

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
            logger.info("submitted generate task to daemon: task_id=%d", task_id)
            self._emit_daemon_state()
        except Exception as e:
            logger.exception(
                "daemon submit failed: task_id=%s; the task is marked failed", task_id,
            )
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
        if kind == "phase":
            # 出图阶段（load/clip/sample/vae）→ 前端进度条覆盖非采样阶段（不再卡 0%/100%）
            self._on_event({
                "type": "generate_phase",
                "task_id": tid,
                "name": event.get("name"),
            })
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
        elif kind == "canceled":
            self._finalize_daemon_task(tid, status="canceled")
            self._emit_daemon_state()
        elif kind == "error":
            self._finalize_daemon_task(
                tid, status="failed", error_msg=str(event.get("message") or "daemon error"),
            )
            self._emit_daemon_state()

    def _on_daemon_log_line(self, entry: dict[str, Any]) -> None:
        """daemon stderr 增量行 → SSE daemon_log_line（前端日志抽屉用）。

        0.17 item1：同时落当前 active generate task 的 run.log + emit task_log_appended
        （LogTab 实时更新）。file 写在锁内避免与 finalize 的 close 竞态；SSE 在锁外发。
        """
        line = entry.get("line")
        self._on_event({
            "type": "daemon_log_line",
            "ts": entry.get("ts"),
            "seq": entry.get("seq"),
            "line": line,
        })
        end_offset: int | None = None
        with self._daemon_lock:
            tid = self._daemon_active_task_id
            fp = self._daemon_log_fp
            if tid is not None and isinstance(line, str):
                if fp is not None:
                    try:
                        fp.write((line + "\n").encode("utf-8", errors="replace"))
                        fp.flush()
                        end_offset = fp.tell()
                    except Exception:
                        # 自我放大回路摘除（R8）：本分支处在**每一行 daemon 日志**
                        # 的路径上，写失败一次就关闭句柄置 None——后续行零日志、
                        # SSE 广播继续（end_offset=None，前端已兼容）。这条失败
                        # 记录走 logger（studio.log），与失败目标（run.log 句柄）
                        # 是不同通道；绝不能改成写 run.log，否则是死循环。
                        logger.warning(
                            "write daemon task log failed: task_id=%s; run.log "
                            "for this task is closed, the SSE log stream "
                            "continues", tid, exc_info=True,
                        )
                        try:
                            fp.close()
                        except Exception:
                            pass
                        self._daemon_log_fp = None
            else:
                tid = None
        if tid is not None and isinstance(line, str):
            # 与 LogTailer 路径同形状（seq + end_offset），前端不用区分来源
            self._on_event({
                "type": "task_log_appended",
                "task_id": tid,
                "text": line,
                "seq": next(self._log_seq),
                "end_offset": end_offset,
            })

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
            logger.warning(
                "emit daemon state failed; the next state change re-publishes it",
                exc_info=True,
            )

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
            log_fp = self._daemon_log_fp
            self._daemon_log_fp = None
        if poller is not None:
            try:
                poller.stop()
            except Exception:
                pass
        if log_fp is not None:  # 0.17 item1：关该 task 的 run.log
            try:
                log_fp.close()
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

        # 落盘失败节流（T6）的收尾汇总：逐图 DEBUG 之后在这里给一条计数 WARNING
        try:
            from ..services.generate_storage import flush_disk_store_summary
            flush_disk_store_summary(task_id)
        except Exception:  # noqa: BLE001 — 收尾日志不能反过来把收尾流程炸掉
            pass

        try:
            from ..services.inference.core import cleanup_generate_tempdir, generate_tempdir
            cleanup_generate_tempdir(task_id)
        except Exception:
            try:
                _tmpdir: Any = generate_tempdir(task_id)
            except Exception:
                _tmpdir = None
            logger.warning(
                "cleanup generate tempdir failed: task_id=%s path=%s",
                task_id, _tmpdir, exc_info=True,
            )

        self._on_event({
            "type": "task_state_changed",
            "task_id": task_id,
            "status": status,
        })
        logger.info("daemon task finished: task_id=%d status=%s", task_id, status)

    def _fail_daemon_task(self, task_id: int, msg: str) -> None:
        """generate task 在派给 daemon 之前的失败（config 缺失等）。"""
        with self._daemon_lock:
            if self._daemon_active_task_id == task_id:
                self._daemon_active_task_id = None
            log_fp = self._daemon_log_fp
            self._daemon_log_fp = None
        if log_fp is not None:  # 0.17 item1：兜底关 run.log（一般此路径未开）
            try:
                log_fp.close()
            except Exception:
                pass
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
        # 子进程的 stderr 就是 run.log：记录不过滤、显示才过滤（docs/design/
        # logging-target-state.md D1），所以 console 级别给 DEBUG 让调试行落盘；
        # setdefault 保留外部显式设的值。
        env.setdefault(LOG_LEVEL_ENV, "DEBUG")
        # xformers 的 triton 探测会把无害的 ImportError traceback 打进 task log
        # （Windows 无官方 triton wheel），被失败摘要误当失败原因；本 app 的
        # xformers 路径不用 triton kernel，无条件短路。
        _xformers_svc.disable_triton_probe(env)
        # 训练侧内存/显存水位保护开关（Settings → 训练 → 训练参数，v0.23.1
        # 起默认关）。runtime 侧 env 缺省=开（CLI 直跑的安全兜底），只有
        # 关闭时才需要显式传；训练与正则 AI 子进程读它，generate daemon
        # 走自己的 cfg.ram_guard 不受影响。
        # 子进程日志语言（Q1 i18n 字典口径）：INFO 叙事行按 UI 语言输出
        try:
            env.setdefault(UI_LANG_ENV, str(_secrets.load().system.ui_language))
        except Exception:
            pass
        try:
            if not _secrets.load().training.ram_guard:
                env.setdefault("LORA_RAM_GUARD", "0")
        except Exception:
            logger.warning(
                "read training ram_guard setting failed; using the default",
                exc_info=True,
            )
        try:
            wandb_cfg = _secrets.load().wandb
            if wandb_cfg.enabled:
                # 0.18 预设化：enabled 是顶层总开关，其余字段读当前选中的 preset。
                wb = wandb_cfg.active
                env.setdefault("WANDB_ENABLED", "1")
                env.setdefault("WANDB_MODE", wb.mode)
                env.setdefault("WANDB_LOG_SAMPLES", "1" if wb.log_samples else "0")
                env.setdefault("WANDB_SAMPLE_MAX_SIDE", str(wb.sample_max_side))
                env.setdefault("WANDB_SAMPLE_EVERY_N_STEPS", str(wb.sample_every_n_steps))
                env.setdefault("WANDB_UPLOAD_MODEL", "1" if wb.upload_model else "0")
                env.setdefault("WANDB_UPLOAD_MODEL_POLICY", wb.upload_model_policy)
                env.setdefault("WANDB_UPLOAD_STATE_MANUAL", "1" if wb.upload_state_manual else "0")
                env.setdefault("WANDB_UPLOAD_STATE_MANUAL_POLICY", wb.upload_state_manual_policy)
                env.setdefault("WANDB_UPLOAD_STATE_AUTO", "1" if wb.upload_state_auto else "0")
                env.setdefault("WANDB_UPLOAD_STATE_AUTO_POLICY", wb.upload_state_auto_policy)
                if wb.api_key:
                    env.setdefault("WANDB_API_KEY", wb.api_key)
                if wb.project:
                    env.setdefault("WANDB_PROJECT", wb.project)
                if wb.entity:
                    env.setdefault("WANDB_ENTITY", wb.entity)
                if wb.base_url:
                    env.setdefault("WANDB_BASE_URL", wb.base_url)
        except Exception:
            logger.warning(
                "read wandb settings failed; this run reports nothing to wandb",
                exc_info=True,
            )
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

        # ADR 0006 PR-2 + Addendum 1 三元分流（原来二元 canceled vs done/failed）。
        # paused 优先级最高 — pause_pending=True 且子进程 emit 了 pause_state
        # （state_path / config_path 都到位）= 真正成功暂停。
        # ADR Addendum 1 方案 Δ：pause_pending=True 但 pause_state_path 空 = 首 epoch
        # 内暂停或子进程退出前没来得及 emit（IO 慢 / 异常 / 强 kill）→ 降级 canceled
        # （ADR §4.3 modal "强制取消保存进度" 兜底）。
        if slot.pause_pending and slot.pause_state_path:
            status = "paused"
        elif slot.pause_pending or slot.cancel_pending:
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
                    # B-1.6: tail task run.log 末 12 行（含 Traceback 优先）
                    # 拼到 error_msg，UI Task 列表能直接看到根因，不必每次翻 trace。
                    # 新 task 走 tasks/<id>/run.log；老 task fallback 到旧 logs/<id>.log。
                    new_log = task_log_path(cid)
                    tail_src = new_log if new_log.exists() else self._logs_dir / f"{cid}.log"
                    tail = _tail_log_for_error_msg(tail_src)
                    fields["error_msg"] = (
                        f"exit code {rc}\n{tail}" if tail else f"exit code {rc}"
                    )
                elif status == "paused":
                    fields["paused_state_path"] = slot.pause_state_path
                    fields["paused_config_path"] = slot.pause_config_path
                    fields["paused_step"] = slot.pause_step
                    fields["paused_at"] = time.time()
                db.update_task(conn, cid, **fields)
                # ADR-0007 §11.3-B：task 终态（done/failed/canceled）独立映射到
                # version.status。paused 不进（task 还能 resume，§11.3-A）。
                if status in ("done", "failed", "canceled"):
                    _maybe_finalize_version(conn, cid, status)
            # commit 10 起：generate task 走 daemon 不进 SLOT_TRAIN，
            # 这条 _finish_slot 路径只跑 train / reg_ai；不再需要 generate
            # tempdir 清理（已搬到 _finalize_daemon_task）。
            self._on_event(
                {"type": "task_state_changed", "task_id": cid, "status": status}
            )
            logger.info("task finished: task_id=%d status=%s rc=%d", cid, status, rc)
            if status == "done" and slot.eval_training_finished_payload is not None:
                self._queue_auto_eval_after_training(
                    cid,
                    slot.eval_training_finished_payload,
                )
        else:  # job
            with db.connection_for(self._db_path) as conn:
                if status == "done":
                    project_jobs.mark_done(conn, cid)
                elif status == "canceled":
                    project_jobs.mark_canceled(conn, cid)
                else:
                    # B-1.6: 同 task — tail 作业 log 拼 error_msg。
                    # R-3：作业日志已随台账合并搬到 tasks/<id>/run.log。
                    tail = _tail_log_for_error_msg(project_jobs.log_path_for(cid))
                    err_msg = f"exit code {rc}\n{tail}" if tail else f"exit code {rc}"
                    project_jobs.mark_failed(conn, cid, err_msg)
                job = project_jobs.get_job(conn, cid)
            self._on_event({
                "type": "job_state_changed",
                "job_id": cid,
                "project_id": job["project_id"] if job else None,
                "version_id": job.get("version_id") if job else None,
                "kind": job["kind"] if job else None,
                "status": status,
            })
            logger.info("job finished: job_id=%d status=%s rc=%d", cid, status, rc)

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
                "%s did not exit in %.1fs; killing process tree: "
                "%s_id=%s slot=%s pid=%d",
                slot.kind, self._grace, slot.kind, slot.id, slot.name, proc.pid,
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
                    "process did not exit in %.1fs; killing process tree: pid=%d",
                    grace, proc.pid,
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
            logger.warning(
                "send terminate signal failed: pid=%d; falling back to force kill",
                proc.pid, exc_info=True,
            )

    @staticmethod
    def _send_pause_signal(
        proc: subprocess.Popen, task_id: Optional[int] = None,
    ) -> None:
        """Pause 软信号 — 子进程 handle_interrupt 接住保 state。

        Windows：`CTRL_BREAK_EVENT` 送达 CREATE_NEW_PROCESS_GROUP 子进程组，
        Python 端映射成 SIGBREAK（sig=21），由 resume phase 注册的 handler 捕获。
        POSIX：`SIGINT` — 跟 SIGTERM 分流，cancel 走 SIGTERM 不撞。

        信号链路经 spike 验证（决策见 ADR 0006）。
        """
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            logger.warning(
                "send pause signal failed: pid=%d task_id=%s; the task keeps running",
                proc.pid, task_id, exc_info=True,
            )

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
        self._send_pause_signal(slot.proc, slot.id)

    def _persist_last_state(self, task_id: int, payload: dict[str, Any]) -> None:
        """把 auto_epoch_backup_written 的恢复点信息写进 tasks 行（ADR Addendum 2）。

        每 epoch 一次 UPDATE，开销可忽略；失败只 log 不抛 —— 训练不能因
        DB 写入异常受影响，丢一拍下个 epoch 会再写。
        """
        state_path = str(payload.get("state_path") or "") or None
        if not state_path:
            return
        try:
            with db.connection_for(self._db_path) as conn:
                db.update_task(
                    conn, task_id,
                    last_state_path=state_path,
                    last_config_path=str(payload.get("config_path") or "") or None,
                    last_state_epoch=payload.get("epoch"),
                    last_state_step=payload.get("step"),
                )
        except Exception:
            logger.warning(
                "resume state persist failed: task_id=%s; this task cannot be "
                "resumed later", task_id, exc_info=True,
            )

    def _clear_pause_fields(self, task_id: int) -> None:
        """清 db `paused_*` 字段（ADR §5.5 / Addendum 2 修订）。

        调用点：
          - resume_state_loaded 事件（cmd_builder 成功 load 后）
          - cancel paused → canceled

        Addendum 2 起 **不再删恢复点文件**：auto_epoch_state.pt 覆盖式单文件
        不会堆积，保留它让 canceled task 可 resume、resume 后立刻仍有恢复点。
        文件清理统一挪到 DELETE task（lifecycle.delete_queue_item）。
        故意 **不改 status** — caller 决定要写什么状态。
        """
        with db.connection_for(self._db_path) as conn:
            task = db.get_task(conn, task_id)
            if not task:
                return
            db.update_task(
                conn, task_id,
                paused_state_path=None,
                paused_config_path=None,
                paused_step=None,
                paused_at=None,
            )
