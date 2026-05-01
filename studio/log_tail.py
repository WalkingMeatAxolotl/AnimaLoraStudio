"""Log file tailer：把追加到 log 文件的字节流增量推到 callback。

用于 supervisor 跟踪 worker 子进程的日志，按行 publish 到 SSE。

PP6.4：增加 MonitorStatePoller —— 监听 monitor_state.json mtime，
变化时 publish 整个 state 给 SSE 订阅者（取代前端 1Hz 轮询 /api/state）。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable


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
        text = self._buffer + chunk.decode("utf-8", errors="replace")
        # 拆行；最后一段不完整就留在 buffer 里下次拼
        lines = text.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            self._on_line(line.rstrip("\r"))


class MonitorStatePoller:
    """轮询 monitor_state.json 的 mtime，变化时把解析后的 dict 推给 callback。

    设计取舍：
    - 不用 watchdog/inotify：跨平台一致性 + 0 依赖；anima_train 写入频率 ~1Hz，
      poll 0.5s 足够低延迟。
    - 直接 publish 整个 state（含 losses 数组）：服务端读一次 → 所有 SSE
      订阅者共享，比 N 个客户端各自 GET /api/state 便宜得多。
    - mtime 没变就不读、不 publish；文件不存在静默跳过。
    """

    def __init__(
        self,
        path: Path,
        on_change: Callable[[dict[str, Any]], None],
        *,
        poll_interval: float = 0.5,
    ) -> None:
        self._path = path
        self._on_change = on_change
        self._poll = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_mtime: float = 0.0

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
        # 退出前再读一次，捕获结束瞬间的最终 state
        try:
            self._check_once()
        except Exception:
            pass

    def _check_once(self) -> None:
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
        self._on_change(data)
