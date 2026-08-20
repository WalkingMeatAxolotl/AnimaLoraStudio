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
import logging
import threading
import weakref
from typing import Any, Callable, Optional

_drop_logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # commit 11：连接生命周期钩子（generate cache 用 last → 30s timer 清）
        self._on_first_subscribe: Optional[Callable[[], None]] = None
        self._on_last_unsubscribe: Optional[Callable[[], None]] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """在 FastAPI 启动时调用一次，绑定主事件循环。"""
        self._loop = loop

    def set_connection_callbacks(
        self,
        on_first_subscribe: Optional[Callable[[], None]] = None,
        on_last_unsubscribe: Optional[Callable[[], None]] = None,
    ) -> None:
        """设连接首/末事件钩子。调用方应处理重入与异常 —— bus 不 catch。"""
        self._on_first_subscribe = on_first_subscribe
        self._on_last_unsubscribe = on_last_unsubscribe

    def connection_count(self) -> int:
        with self._lock:
            return len(self._queues)

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        with self._lock:
            was_empty = len(self._queues) == 0
            self._queues.add(q)
        if was_empty and self._on_first_subscribe is not None:
            self._on_first_subscribe()
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._queues.discard(q)
            now_empty = len(self._queues) == 0
        if now_empty and self._on_last_unsubscribe is not None:
            self._on_last_unsubscribe()

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


# B-1.5 + R8：慢消费者丢事件是**持续状态**不是离散事件——训练进度风暴时逐条
# WARNING 能刷上千行黄。状态化：进入丢弃态 1 条、丢弃期间只计数、恢复时 1 条带
# 累计数（一次拥塞恒定 2 条）。_safe_put 只在 event loop 线程执行，无竞态；
# WeakKeyDictionary 随队列 unsubscribe 自动清。
_drop_states: "weakref.WeakKeyDictionary[asyncio.Queue, dict[str, Any]]" = (
    weakref.WeakKeyDictionary()
)


def _safe_put(q: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    st = _drop_states.get(q)
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        if st is None:
            st = _drop_states[q] = {"dropped": 0, "last_type": "?"}
        if st["dropped"] == 0:
            # 用 module logger 走 studio.log，自带 trace_id（publisher 在 request ctx 内时）
            _drop_logger.warning(
                "event_bus slow consumer: queue full, dropping events "
                "(type=%s queue_maxsize=%d)",
                event.get("type", "?"), q.maxsize,
            )
        st["dropped"] += 1
        st["last_type"] = event.get("type", "?")
        return
    if st is not None and st["dropped"]:
        _drop_logger.warning(
            "event_bus slow consumer recovered: dropped %d event(s) "
            "(last type=%s)",
            st["dropped"], st["last_type"],
        )
        st["dropped"] = 0


# 进程内单例（server.py 用）
bus = EventBus()
