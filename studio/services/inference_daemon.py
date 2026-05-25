"""测试出图常驻 daemon：复用模型加载，避免每次出图 30-60s reload。

设计要点：
  - daemon 是个常驻 subprocess（runtime/anima_daemon.py），由 server 进程内的
    InferenceDaemon 类管理；JSON-over-stdio 协议，stderr 走日志
  - lazy spawn：第一次有 generate task 来时才起；起来后保持 alive 直到
    server 关闭、用户主动 unload、或 GPU 让位（commit 12）
  - 一次跑一个 task（队列由 supervisor 喂；daemon 内部不排队），完成后回 idle
  - 协议（line-delimited JSON）：
      stdin  → {"id": "<req_id>", "action": "generate"|"unload"|"ping", ...}
      stdout → {"id": "<req_id>"|"_evt", "kind": "started"|"image_done"|
                 "done"|"error"|"loaded"|"unloaded", ...}
  - image_done 事件 payload 含 base64 PNG bytes（commit 10 起）；reader
    把它解码进 generate_cache，再把"瘦身版"事件（去 b64）转发给 supervisor
    callback，避免大 payload 进日志/SSE 链路
  - reader thread 把 stdout 事件分发回 callback；调用方（supervisor）注册 callback
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..paths import REPO_ROOT
from . import generate_cache

logger = logging.getLogger(__name__)

# Daemon 状态机
STATE_STOPPED = "stopped"      # 子进程未启动 / 已退出
STATE_STARTING = "starting"    # spawn 中，未收到 ready 信号
STATE_IDLE = "idle"            # daemon 活着等命令；模型可能已 load 也可能未 load
STATE_BUSY = "busy"            # daemon 正在跑一个 task
STATE_UNLOADING = "unloading"  # 收到 unload 指令，等 unloaded 事件


# Daemon 进程脚本路径
_DAEMON_SCRIPT = REPO_ROOT / "runtime" / "anima_daemon.py"


EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class _ActiveTask:
    """daemon 当前在跑的 task（或刚提交还没收到 started 事件的 task）。"""
    task_id: int
    request_id: str
    on_event: EventCallback
    started_at: float = field(default_factory=time.time)


class InferenceDaemon:
    """测试出图 daemon 的服务端代理。线程安全。

    使用模式（singleton）：
        d = InferenceDaemon()
        d.start()
        d.submit_task(task_id=42, config={...}, on_event=cb)
        # ...等 cb 收到 done 事件
        d.stop()

    `on_event` 收到的事件 dict 形如：
        {"kind": "started", "task_id": 42}
        {"kind": "image_done", "task_id": 42, "filename": "gen_0000_p0_c0_s42.png",
                                "path": "/tmp/anima_gen_42/..."}
        {"kind": "done", "task_id": 42}
        {"kind": "error", "task_id": 42, "message": "..."}
    """

    READY_TIMEOUT = 30.0  # 子进程 import 完成给 ready 的最长等待
    UNLOAD_TIMEOUT = 60.0  # unload 后等 unloaded 事件最长

    def __init__(self, *, script_path: Optional[Path] = None) -> None:
        self._script = script_path or _DAEMON_SCRIPT
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._state: str = STATE_STOPPED
        self._model_loaded: bool = False
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._active: Optional[_ActiveTask] = None
        self._req_seq = 0
        # 全局 listener（用于 daemon 状态变化：loaded / unloaded / 进程崩溃）
        self._global_listeners: list[EventCallback] = []

    # ---------------------------------------------------------------- 状态
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def is_busy(self) -> bool:
        return self.state == STATE_BUSY

    @property
    def is_alive(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def is_model_loaded(self) -> bool:
        """模型是否在 VRAM 里（commit 12 GPU 让位判定用）。"""
        with self._lock:
            return self._model_loaded

    def add_global_listener(self, cb: EventCallback) -> None:
        with self._lock:
            self._global_listeners.append(cb)

    # --------------------------------------------------------------- 生命周期
    def start(self) -> None:
        """spawn daemon 子进程；已在跑直接返回。"""
        with self._lock:
            if self._state != STATE_STOPPED:
                return
            self._state = STATE_STARTING

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TRANSFORMERS_VERBOSITY", "error")
        env.setdefault("DIFFUSERS_VERBOSITY", "error")

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        cmd = [sys.executable, str(self._script)]
        logger.info("spawning inference daemon: %s", " ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(REPO_ROOT),
                env=env,
                creationflags=creationflags,
                bufsize=1,
                text=True,
                encoding="utf-8",
            )
        except Exception:
            with self._lock:
                self._state = STATE_STOPPED
            logger.exception("failed to spawn daemon")
            raise

        with self._lock:
            self._proc = proc
            # reader thread 处理 stdout（协议）
            self._reader_thread = threading.Thread(
                target=self._read_stdout_loop,
                args=(proc,),
                name="inference-daemon-stdout",
                daemon=True,
            )
            self._reader_thread.start()
            # stderr thread 把日志转发到本进程 logger
            self._stderr_thread = threading.Thread(
                target=self._read_stderr_loop,
                args=(proc,),
                name="inference-daemon-stderr",
                daemon=True,
            )
            self._stderr_thread.start()

        # 等 ready
        deadline = time.time() + self.READY_TIMEOUT
        while time.time() < deadline:
            with self._lock:
                if self._state == STATE_IDLE:
                    return
                if self._state == STATE_STOPPED:
                    raise RuntimeError("daemon exited before ready")
            time.sleep(0.05)
        raise TimeoutError(f"daemon not ready in {self.READY_TIMEOUT}s")

    def stop(self, timeout: float = 10.0) -> None:
        """关闭 daemon 子进程。优雅 → 强杀。"""
        with self._lock:
            proc = self._proc
            if proc is None:
                self._state = STATE_STOPPED
                return
        # 关 stdin → daemon 主循环 EOF 退出
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("daemon didn't exit in %.1fs, killing", timeout)
            try:
                proc.kill()
                proc.wait(timeout=3.0)
            except Exception:
                pass
        with self._lock:
            self._proc = None
            self._state = STATE_STOPPED
            self._model_loaded = False
            self._active = None

    # ----------------------------------------------------------------- 提交
    def submit_task(
        self,
        *,
        task_id: int,
        config: dict[str, Any],
        output_dir: str,
        on_event: EventCallback,
    ) -> str:
        """提交一个 generate task 给 daemon。daemon 必须 idle。

        返回 request_id。同步发命令；后续事件通过 on_event 异步推。
        """
        with self._lock:
            if self._state != STATE_IDLE:
                raise RuntimeError(
                    f"daemon not ready to accept task (state={self._state})"
                )
            self._req_seq += 1
            req_id = f"task-{task_id}-{self._req_seq}"
            self._active = _ActiveTask(
                task_id=task_id, request_id=req_id, on_event=on_event,
            )
            self._state = STATE_BUSY
            assert self._proc is not None and self._proc.stdin is not None
            stdin = self._proc.stdin

        msg = {
            "id": req_id,
            "action": "generate",
            "task_id": task_id,
            "config": config,
            "output_dir": output_dir,
        }
        try:
            stdin.write(json.dumps(msg) + "\n")
            stdin.flush()
        except Exception as e:
            logger.exception("failed to send task to daemon")
            with self._lock:
                self._state = STATE_IDLE
                self._active = None
            raise RuntimeError(f"daemon write failed: {e}") from e
        return req_id

    def cancel_active_task(self, task_id: int) -> bool:
        """请求取消当前 generate task；daemon 保持常驻，模型缓存不卸载。"""
        with self._lock:
            active = self._active
            if self._state != STATE_BUSY or active is None or active.task_id != task_id:
                return False
            assert self._proc is not None and self._proc.stdin is not None
            stdin = self._proc.stdin
            req_id = active.request_id

        try:
            stdin.write(json.dumps({
                "id": f"cancel-{task_id}",
                "action": "cancel",
                "target_id": req_id,
            }) + "\n")
            stdin.flush()
        except Exception:
            logger.exception("failed to send cancel")
            return False
        return True

    def request_unload(self) -> None:
        """通知 daemon 卸载模型（释放 VRAM）。daemon 处理完会推 unloaded 事件。

        commit 9 不暴露给前端；为 commit 12 GPU 让位 / commit 13 手动卸载预留。
        """
        with self._lock:
            if self._state == STATE_STOPPED:
                return
            if self._state == STATE_BUSY:
                logger.warning("unload requested while busy; ignored")
                return
            assert self._proc is not None and self._proc.stdin is not None
            stdin = self._proc.stdin
            self._state = STATE_UNLOADING
        try:
            stdin.write(json.dumps({"id": "_unload", "action": "unload"}) + "\n")
            stdin.flush()
        except Exception:
            logger.exception("failed to send unload")

    # ----------------------------------------------------------- 内部 reader
    def _read_stdout_loop(self, proc: subprocess.Popen) -> None:
        """读 daemon stdout 行 → JSON 解析 → 分发。"""
        assert proc.stdout is not None
        try:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("daemon stdout non-JSON: %r", line[:200])
                    continue
                self._handle_event(msg)
        except Exception:
            logger.exception("daemon stdout reader crashed")
        finally:
            self._handle_proc_exit(proc)

    def _read_stderr_loop(self, proc: subprocess.Popen) -> None:
        """daemon stderr → 本进程 logger.info（保留原日志层级前缀）。"""
        assert proc.stderr is not None
        try:
            for raw_line in proc.stderr:
                line = raw_line.rstrip()
                if line:
                    logger.info("[daemon] %s", line)
        except Exception:
            logger.exception("daemon stderr reader crashed")

    def _handle_event(self, msg: dict[str, Any]) -> None:
        """分发协议消息。task 事件路由到 _active.on_event；全局事件给 listeners。"""
        kind = msg.get("kind")
        msg_id = msg.get("id")

        if msg_id == "_evt":
            # daemon 全局状态事件
            with self._lock:
                if kind == "ready":
                    self._state = STATE_IDLE
                    self._model_loaded = False
                elif kind == "loaded":
                    self._model_loaded = True  # 状态保持 IDLE
                elif kind == "unloaded":
                    self._state = STATE_IDLE
                    self._model_loaded = False
            for cb in list(self._global_listeners):
                try:
                    cb(msg)
                except Exception:
                    logger.exception("global listener failed")
            return

        # task 事件
        with self._lock:
            active = self._active
        if active is None or active.request_id != msg_id:
            logger.warning("event for unknown request: %s", msg_id)
            return

        # commit 10：image_done 含 base64 PNG → 入 cache，转发瘦身版（无 b64）
        # commit 14：preview_step 含 base64 JPEG → 直接透传给 callback（不入 cache，
        #   前端 SSE 收到立刻 <img src="data:..."> 显示当前步预览；done/最终图
        #   会替换它）
        forward_msg = msg
        if kind == "image_done" and "image_b64" in msg:
            filename = msg.get("filename") or ""
            try:
                data = base64.b64decode(msg["image_b64"])
                generate_cache.cache_image(active.task_id, filename, data)
            except Exception:
                logger.exception("cache_image failed for %s", filename)
            forward_msg = {k: v for k, v in msg.items() if k != "image_b64"}

        # done/error/canceled 先切状态，再回调 —— 让 callback 内查询 is_busy/state 时
        # 看到准确的 IDLE 状态（commit 13 daemon_state_changed 依赖这个顺序）
        if kind in ("done", "error", "canceled"):
            with self._lock:
                self._active = None
                self._state = STATE_IDLE

        try:
            active.on_event({**forward_msg, "task_id": active.task_id})
        except Exception:
            logger.exception("task on_event handler failed")

    def _handle_proc_exit(self, proc: subprocess.Popen) -> None:
        """子进程退出处理：标 STOPPED + 给 active task 推 error + 通知 listeners。"""
        rc = proc.wait()
        logger.warning("inference daemon exited rc=%d", rc)
        with self._lock:
            self._proc = None
            prev_state = self._state
            self._state = STATE_STOPPED
            self._model_loaded = False
            active = self._active
            self._active = None
            listeners = list(self._global_listeners)

        if active is not None and prev_state != STATE_UNLOADING:
            try:
                active.on_event({
                    "kind": "error",
                    "task_id": active.task_id,
                    "message": f"daemon exited unexpectedly (rc={rc})",
                })
            except Exception:
                logger.exception("error handler failed")

        for cb in listeners:
            try:
                cb({"id": "_evt", "kind": "stopped", "rc": rc})
            except Exception:
                logger.exception("listener failed on proc exit")


# Singleton 句柄；server 启动时初始化（lazy spawn）
_INSTANCE: Optional[InferenceDaemon] = None
_INSTANCE_LOCK = threading.Lock()


def get_daemon() -> InferenceDaemon:
    """返回单例 daemon 实例（懒构造）。"""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = InferenceDaemon()
        return _INSTANCE


def reset_daemon_for_test() -> None:
    """测试用：清掉 singleton 让下个测试拿干净实例。"""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            try:
                _INSTANCE.stop(timeout=3.0)
            except Exception:
                pass
        _INSTANCE = None
