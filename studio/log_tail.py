"""Log file tailer：把追加到 log 文件的字节流增量推到 callback。

用于 supervisor 跟踪 worker 子进程的日志，按行 publish 到 SSE。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable


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
