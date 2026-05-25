"""Log file tailer：把追加到 log 文件的字节流增量推到 callback。

用于 supervisor 跟踪 worker 子进程的日志，按行 publish 到 SSE。

PP6.4：增加 MonitorStatePoller —— 监听 monitor_state.json mtime，
变化时 publish 整个 state 给 SSE 订阅者（取代前端 1Hz 轮询 /api/state）。
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

# C++ 库（典型如 onnxruntime）有时直接往 worker 进程的 fd 2 写带 ANSI 颜色码
# 的日志，前端 <pre> 不解析 ANSI，会渲染成 `日[1;31m...` 之类的乱码。Windows
# 上还会塞 UTF-16 风格的 NUL 字节，让一行 ASCII 看起来字间夹空格。统一在
# tail 阶段剥掉，让前端拿到的就是干净文本。
_ANSI_CSI_RE = re.compile(r"\x1b\[[\d;?]*[A-Za-z]")


class LogTailer:
    """轮询 log 文件，把新增字节按行送给 `on_line(line)`。

    线程安全；start/stop 各调一次；不抛错（IO 失败静默重试）。
    """

    def __init__(
        self,
        path: Path,
        on_line: Callable[[str], None],
        *,
        poll_interval: float = 0.3,
    ) -> None:
        self._path = path
        self._on_line = on_line
        self._poll = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0
        self._buffer = ""

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"log-tail-{self._path.name}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        # 收尾：flush 残余 buffer 作为最后一行
        if self._buffer.strip():
            try:
                self._on_line(self._buffer.rstrip("\r\n"))
            finally:
                self._buffer = ""

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._read_chunk()
            except Exception:
                # IO 异常不向上抛，避免拖死 supervisor
                pass
            self._stop.wait(self._poll)
        # 退出前再 flush 一次，捕获结束瞬间的输出
        try:
            self._read_chunk()
        except Exception:
            pass

    def _read_chunk(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read()
            if not chunk:
                return
            self._offset += len(chunk)
        raw = chunk.decode("utf-8", errors="replace")
        # 剥 ANSI CSI 转义 + NUL 字节（onnxruntime 等 C++ 库直写 fd 2 的副产物）
        cleaned = _ANSI_CSI_RE.sub("", raw).replace("\x00", "")
        text = self._buffer + cleaned
        # 拆行；最后一段不完整就留在 buffer 里下次拼
        lines = text.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            self._on_line(line.rstrip("\r"))


class MonitorStatePoller:
    """轮询 monitor_state.json 的 mtime，变化时**构造增量 delta** 推给 callback。

    协议（PR #37 改造）：早期版本每次推全量 state（losses/lr 数组每步都全
    量重传，2000 步训练单次推 ~200KB）。云部署跨公网时这是 O(N²) 浪费。

    新设计：poller 维护 last_step / last_loss_count / last_lr_count /
    last_optimizer_metrics_count / last_sample_count，每次只把「自上次发布以来的新增」打成 delta：

        {
          "step": 234, "total_steps": 2000,
          "epoch": 3, "total_epochs": 10,
          "speed": 1.2, "start_time": 1234567890.0,
          "appended_losses": [{step, loss, time}],   # 可能为空数组
          "appended_lr":     [{step, lr}],
          "appended_optimizer_metrics": [{step, actual_lr, d, ...}],
          "appended_samples":[{path, step, time, xy?}],
          "config": {...},        # 仅在变化时携带（首次推送 / config 改）
        }

    Throttle 规则（混合 step + 时间）：
    - poll_interval 0.5s 探测 mtime
    - min_publish_interval 1.0s 强制下界 — 即使训练每 100ms 一步，也不会推
      超过 1Hz；累积的 loss 点会在下次推送里 batch 一起送
    - 「step 没变 + 没新 sample + config 没变」时跳过推送，避免空 delta
      （例如训练只更新了 speed 这种衍生指标）
    """

    def __init__(
        self,
        path: Path,
        on_delta: Callable[[dict[str, Any]], None],
        *,
        poll_interval: float = 0.5,
        min_publish_interval: float = 1.0,
    ) -> None:
        self._path = path
        self._on_delta = on_delta
        self._poll = poll_interval
        self._min_pub = min_publish_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_mtime: float = 0.0
        self._last_publish_at: float = 0.0

        # 增量追踪
        self._last_step: int = -1
        self._last_loss_count: int = 0
        self._last_lr_count: int = 0
        self._last_optimizer_metrics_count: int = 0
        self._last_sample_count: int = 0
        self._last_config: dict[str, Any] | None = None

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"monitor-state-{self._path.name}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._check_once()
            except Exception:
                # IO/解析异常静默重试，避免拖死 supervisor
                pass
            self._stop.wait(self._poll)
        # 退出前再读一次，捕获结束瞬间的最终 state；带 force=True 绕过 throttle
        try:
            self._check_once(force=True)
        except Exception:
            pass

    def _check_once(self, *, force: bool = False) -> None:
        if not self._path.exists():
            return
        mtime = self._path.stat().st_mtime
        if mtime <= self._last_mtime:
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 写一半的 JSON / 临时锁住 → 下一轮再试
            return
        self._last_mtime = mtime

        # ── 节流：未到最小发布间隔时退出，下一轮再检查（含新累积的数据） ──
        now = time.time()
        if not force and (now - self._last_publish_at) < self._min_pub:
            return

        # ── 计算 delta ──
        step = int(data.get("step", 0) or 0)
        losses = data.get("losses") or []
        lr_hist = data.get("lr_history") or []
        optimizer_metrics = data.get("optimizer_metrics_history") or []
        samples = data.get("samples") or []
        config = data.get("config") or {}

        appended_losses = losses[self._last_loss_count:]
        appended_lr = lr_hist[self._last_lr_count:]
        appended_optimizer_metrics = optimizer_metrics[self._last_optimizer_metrics_count:]
        appended_samples = samples[self._last_sample_count:]
        config_changed = config != self._last_config

        has_progress = (
            step != self._last_step
            or appended_losses
            or appended_lr
            or appended_optimizer_metrics
            or appended_samples
            or config_changed
        )
        if not (force or has_progress):
            return

        delta: dict[str, Any] = {
            "step": step,
            "total_steps": int(data.get("total_steps", 0) or 0),
            "epoch": int(data.get("epoch", 0) or 0),
            "total_epochs": int(data.get("total_epochs", 0) or 0),
            "speed": float(data.get("speed", 0.0) or 0.0),
            "start_time": data.get("start_time"),
            "appended_losses": appended_losses,
            "appended_lr": appended_lr,
            "appended_optimizer_metrics": appended_optimizer_metrics,
            "appended_samples": appended_samples,
        }
        if config_changed:
            delta["config"] = config

        # 推送 + 更新游标
        self._last_step = step
        self._last_loss_count = len(losses)
        self._last_lr_count = len(lr_hist)
        self._last_optimizer_metrics_count = len(optimizer_metrics)
        self._last_sample_count = len(samples)
        self._last_config = config
        self._last_publish_at = now

        self._on_delta(delta)
