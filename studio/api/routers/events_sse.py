"""SSE 事件流（PR-5 从 server.py 抽出）。

1 route：
    GET /api/events    SSE — 广播 task/job 状态变化 + monitor delta + daemon state 等
                       给所有订阅者。15s keepalive 防代理超时；await is_disconnected()
                       立刻 break 不残留 queue（防内存泄漏）。
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ...event_bus import bus

router = APIRouter()


@router.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    """SSE：广播任务状态变化事件给所有订阅者。"""
    queue = await bus.subscribe()

    async def gen() -> AsyncIterator[bytes]:
        try:
            yield b": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(evt)}\n\n".encode("utf-8")
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")
