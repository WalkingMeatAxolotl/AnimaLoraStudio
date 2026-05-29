"""统一 exception handler 注册（ADR-0009 §4 / PR-2 C2）。

3 个 handler:

  1. DomainError → dual-write envelope:
        {"detail": <message>,                # legacy contract（前端 client.ts
                                              #   3 处解析点 + 5 测试 11 处断言）
         "error": {"code", "message", "trace_id", "details"?}}  # 新结构化
     4xx 不打 stack，5xx 才打 logger.exception（ADR-0009 §4.1）。

  2. RequestValidationError → 保 starlette 默认 `{"detail": [...]}` 不动
     （pydantic body 校验失败 — 前端有专门处理；改 envelope 破现状）。
     middleware 已经自动加 X-Trace-Id header。

  3. Exception fallback → 500 + dual-write envelope，message 脱敏：
        {"detail": "Internal Server Error",
         "error": {"code": "internal.server_error",
                   "message": "Internal Server Error (see trace_id in server log)",
                   "trace_id": "..."}}
     原始 traceback **不**进 response（防 leak）；进 studio.log 让开发者按
     trace_id grep。

HTTPException 不重新注册 — starlette 默认 handler 已经处理（仅返 `{"detail":
<orig>}`），X-Trace-Id header 由 TraceIdMiddleware 自动加。不动它保现有
175 处 raise HTTPException(...) 完全不破。

dual-write 渐进迁移路径（ADR-0009 §错误 envelope 渐进迁移）：
  Phase 1 (0.12.0 / 本 PR): dual-write 同时填 detail + error
  Phase 2 (0.13.0): raise HTTPException 加 deprecation log；前端迁 body.error.*
  Phase 3 (0.14.0): handler 删 detail key
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..domain.errors import DomainError
from ..infrastructure.logging import get_trace_id

logger = logging.getLogger(__name__)


def _trace_id_from(req: Optional[Request]) -> Optional[str]:
    """优先 request.scope state（TraceIdMiddleware 写入，跨外层 handler 仍可用）；
    fallback contextvar（同进程同 scope）。fallback handler 跑在 ServerErrorMiddleware
    层，contextvar 已 reset — 必须靠 scope state。
    """
    if req is not None:
        state = req.scope.get("state") if hasattr(req, "scope") else None
        if state and state.get("trace_id"):
            return state["trace_id"]
    return get_trace_id()


def _error_envelope(
    *, message: str, code: str,
    details: Optional[Dict[str, Any]] = None,
    req: Optional[Request] = None,
) -> Dict[str, Any]:
    """dual-write body: detail (legacy str) + error (structured)。"""
    err: Dict[str, Any] = {
        "code": code,
        "message": message,
        "trace_id": _trace_id_from(req),
    }
    if details:
        err["details"] = details
    return {"detail": message, "error": err}


async def _domain_error_handler(req: Request, exc: DomainError) -> JSONResponse:
    # 4xx 业务异常用 info（非异常路径，是契约的一部分）；5xx 才 exception。
    if exc.http_status >= 500:
        logger.exception("domain error %s: %s", exc.code, exc.message)
    else:
        logger.info("domain error %s: %s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.http_status,
        content=_error_envelope(
            message=exc.message, code=exc.code, details=exc.details, req=req,
        ),
    )


async def _request_validation_handler(
    _req: Request, exc: RequestValidationError,
) -> JSONResponse:
    # pydantic 默认 detail 是 list[dict]；保现状（前端有专门处理）。
    # 不 dual-write 因为 body validation 不是 DomainError，不强行套 envelope。
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


async def _fallback_handler(req: Request, exc: Exception) -> JSONResponse:
    # 未捕获异常 — 进 logger.exception 带完整 traceback + trace_id 给开发查；
    # response body 脱敏不含 traceback 防 leak。
    logger.exception(
        "unhandled exception in %s %s", req.method, req.url.path,
    )
    return JSONResponse(
        status_code=500,
        content=_error_envelope(
            message="Internal Server Error (see trace_id in server log)",
            code="internal.server_error",
            req=req,
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """app.py 启动时调一次。

    顺序无关（FastAPI 按异常类型最具体匹配）。HTTPException **不**注册 — 让
    starlette 默认 handler 跑（保现有 175 处 raise HTTPException 形状不变）。
    """
    app.add_exception_handler(DomainError, _domain_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
    app.add_exception_handler(Exception, _fallback_handler)
