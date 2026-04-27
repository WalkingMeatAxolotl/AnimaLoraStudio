"""线程安全的事件总线：supervisor（同步线程）→ FastAPI SSE（asyncio）。

使用方式：
    bus = EventBus()
    # 在 FastAPI lifespan 启动时绑定 event loop
    bus.attach_loop(asyncio.get_running_loop())

    # SSE 连接
    q = await bus.subscribe()
    try:
        evt = await q.get()
    finally:
        bus.unsubscribe(q)

    # 任意线程发布
    bus.publish({"type": "task_state_changed", "task_id": 7, "status": "done"})
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional


class EventBus:
    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """在 FastAPI 启动时调用一次，绑定主事件循环。"""
        self._loop = loop

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        with self._lock:
            self._queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._queues.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        """线程安全：同步代码（如 supervisor 线程）也能调用。"""
        loop = self._loop
        with self._lock:
            queues = list(self._queues)
        if not loop or not queues:
            return
        for q in queues:
            try:
                loop.call_soon_threadsafe(_safe_put, q, event)
            except RuntimeError:
                # loop 已经停了
                pass


def _safe_put(q: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        pass  # 慢消费者：丢弃，不阻塞 publisher


# 进程内单例（server.py 用）
bus = EventBus()
