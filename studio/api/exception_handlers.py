"""统一 exception handler 注册（ADR-0009 §4 / PR-2 C2）。

4 个 handler（Phase 3 起：错误响应只发 `error` 信封，legacy `detail` 已移除）:

  1. DomainError → `{"error": {"code", "message", "trace_id", "details"?}}`。
     4xx 不打 stack，5xx 才打 logger.exception（ADR-0009 §4.1）。

  2. RequestValidationError → 保 starlette 默认 `{"detail": [...]}` 不动
     （pydantic body 校验失败 — 前端有专门处理；这是唯一保留 detail 的路径）。
     middleware 已经自动加 X-Trace-Id header。

  3. HTTPException（backstop）→ 给未迁移 / 框架 HTTPException 也补 `{"error": {...}}`
     （code=`http.<status>`）；dict/list detail 放进 error.details。

  4. Exception fallback → 500 + `{"error": {...}}`，message 脱敏：
        {"error": {"code": "internal.server_error",
                   "message": "Internal Server Error (see trace_id in server log)",
                   "trace_id": "..."}}
     原始 traceback **不**进 response（防 leak）；进 studio.log 让开发者按
     trace_id grep。

ADR-0009 §错误 envelope 渐进迁移（完成）：
  Phase 1 (0.12.0): dual-write 同时填 detail + error —— 已发布
  Phase 2 (0.15.0): backstop handler 让 body.error 全覆盖 + ~330 处 raise 迁 DomainError
    带语义 code + 前端按 code 查 errors.* i18n —— 已实现
  Phase 3 (0.15.0): 删 legacy detail key，错误响应只发 error（RequestValidationError
    的 422 list 除外）—— 本次
  详见 docs/todo/error-envelope-detail-key-removal.md
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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
    """单一 error 信封（ADR-0009 Phase 3：legacy `detail` key 已移除，只发 error）。"""
    err: Dict[str, Any] = {
        "code": code,
        "message": message,
        "trace_id": _trace_id_from(req),
    }
    if details:
        err["details"] = details
    return {"error": err}


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


async def _http_exception_handler(
    req: Request, exc: StarletteHTTPException,
) -> JSONResponse:
    """ADR-0009 Phase 2/3：给裸 HTTPException 也补 error 信封，让 body.error 覆盖
    所有错误响应（Phase 3 起只发 error，不再 dual-write legacy detail）。前端一律
    读 body.error.code → i18n。

    - detail 是 str → message=detail，code=`http.<status>`（无语义 code 兜底；
      已迁移到 DomainError 的端点带语义 code，不走这里；剩下多是框架/未迁移）。
    - detail 是 dict/list（罕见，业务迁移后已无来源）→ 放进 error.details 保留结构，
      message 取 dict.message/error 兜底。
    保留 exc.headers（如 401 WWW-Authenticate）。
    """
    detail = exc.detail
    err: Dict[str, Any] = {
        "code": f"http.{exc.status_code}",
        "message": "Request failed",
        "trace_id": _trace_id_from(req),
    }
    if isinstance(detail, str):
        err["message"] = detail
    elif isinstance(detail, dict):
        err["message"] = str(detail.get("message") or detail.get("error") or "Request failed")
        err["details"] = detail
    elif detail is not None:
        err["details"] = {"detail": detail}
    return JSONResponse(
        status_code=exc.status_code, content={"error": err},
        headers=getattr(exc, "headers", None),
    )


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

    顺序无关（FastAPI 按异常类型最具体匹配）。HTTPException 注册 backstop handler
    （ADR-0009 Phase 2）：未迁移到 DomainError 的裸 HTTPException 也补上 error 信封，
    detail 原样保留不破现有形状。
    """
    app.add_exception_handler(DomainError, _domain_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _fallback_handler)
