"""TraceIdMiddleware（ADR-0009 §3.2，PR-1 C5）。

Pure ASGI middleware（不继承 BaseHTTPMiddleware），原因：
  - BaseHTTPMiddleware 用 starlette 内置的 anyio.Stream wrapping，ContextVar
    跨 thread 跳跃在 starlette 0.36+ 有已知问题
  - Pure ASGI 直接拿 receive/send，ContextVar 在请求生命周期内稳定

每个 HTTP 请求开头：
  1. 读 X-Trace-Id header，无则 new_trace_id()
  2. bind_trace_id → 整个 request scope 的 logger.x 自动带
  3. response.headers 写回 X-Trace-Id（前端 client.ts 拿到能存 atom）
  4. 请求结束 reset_trace_id

为什么 middleware 不 router：router scope 是 path-after-match；404 / 异常
路径前的请求拿不到 trace_id。Middleware 是 ASGI 最外层，所有响应都过。
"""
from __future__ import annotations

from typing import Awaitable, Callable

from ..infrastructure.logging import (
    TRACE_HEADER,
    bind_trace_id,
    new_trace_id,
    reset_trace_id,
)

_TRACE_HEADER_LOWER = TRACE_HEADER.lower().encode("ascii")


class TraceIdMiddleware:
    """ASGI middleware；只处理 http scope，websocket / lifespan 透传。"""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # 读 header（scope["headers"] 是 List[(bytes, bytes)]）
        trace_id: str | None = None
        for k, v in scope["headers"]:
            if k.lower() == _TRACE_HEADER_LOWER:
                try:
                    trace_id = v.decode("ascii", "replace").strip()
                except Exception:
                    trace_id = None
                if trace_id:
                    break
        if not trace_id:
            trace_id = new_trace_id()

        # 把 trace_id 写到 scope["state"]，让 ServerErrorMiddleware 外层的
        # fallback Exception handler 能拿到（contextvar 在 finally 里 reset，
        # 外层 handler 跑时 contextvar 已空）。DomainError handler 在 ExceptionMiddleware
        # 内层，靠 contextvar 仍可用；fallback 必须靠 scope state。
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["trace_id"] = trace_id

        token = bind_trace_id(trace_id)

        async def send_wrapper(message):
            # 给 response 写回 X-Trace-Id header
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # 防重复：先去掉已有同名 header（罕见但 starlette 可能加 trace_id_middleware 嵌套）
                headers = [(k, v) for k, v in headers if k.lower() != _TRACE_HEADER_LOWER]
                headers.append((_TRACE_HEADER_LOWER, trace_id.encode("ascii", "replace")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            reset_trace_id(token)
