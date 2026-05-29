"""前端错误上报端点（ADR-0009 §5.2 / PR-3 C1）。

POST /api/client-errors  — 前端 ErrorBoundary / window.onerror /
unhandledrejection 三路捕获到的浏览器端错误上报到这里。

body 是任意 dict — 前端可自由扩展字段。后端只识别：
    kind           "react.boundary" | "window.error" | "unhandledrejection" | "manual"
    message        string
    stack          string?           Error.stack
    componentStack string?           react boundary 的组件栈
    source         string?           script URL（window.error）
    line / col     int?              window.error 行列号
    url            string            location.href
    user_agent     string            navigator.userAgent
    client_ts      string            ISO8601
    app_version    string
    build_hash     string?
    trace_id_last_4xx  string?       前端记录的最近一次 4xx 响应 trace_id
                                     （让开发查 server 端那次失败的上下文）

返 **204 No Content** — 永远成功；上报失败不能级联前端 UI（前端 silent
swallow）。

per-IP 限流：内存令牌桶，10 次 / 分钟。超过 silently drop（仍返 204）防
ad-blocker / 用户离线 / network flap 雪崩。

记录走 `studio.client` logger → studio.log JSON line。开发者：
    jq 'select(.logger == "studio.client")' studio_data/logs/studio.log
"""
from __future__ import annotations

import logging
import time
from collections import deque
from threading import Lock
from typing import Any, Deque, Dict

from fastapi import APIRouter, Request, Response

logger = logging.getLogger("studio.client")
router = APIRouter()

_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_MAX_PER_WINDOW = 10
_ip_buckets: Dict[str, Deque[float]] = {}
_bucket_lock = Lock()


def _rate_limit_ok(ip: str, *, now: float | None = None) -> bool:
    """True = 在限额内可上报；False = 超 10/min 拒收（silently 204）。

    `now` 参数仅给测试用，覆盖时间。
    """
    t = now if now is not None else time.monotonic()
    cutoff = t - _RATE_LIMIT_WINDOW_SECONDS
    with _bucket_lock:
        bucket = _ip_buckets.setdefault(ip, deque())
        # 清出窗口外的
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_MAX_PER_WINDOW:
            return False
        bucket.append(t)
        return True


def _client_ip(request: Request) -> str:
    """IP 兜底优先级：X-Forwarded-For 第一个 → request.client.host → 'unknown'。

    单机部署 client 全 127.0.0.1 共享限额（10/min 上限对单人也足够）；
    反代部署看 X-Forwarded-For。
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@router.post("/api/client-errors", status_code=204)
async def report_client_error(request: Request) -> Response:
    """前端上报。**永远 204** — 不让上报失败级联前端 UI。"""
    ip = _client_ip(request)
    if not _rate_limit_ok(ip):
        # 超限 silently drop；偶尔记一行 INFO 防完全失明
        logger.info("client_errors rate-limit drop ip=%s", ip)
        return Response(status_code=204)

    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        # 非 JSON body — 不上报
        logger.warning("client_errors malformed body from ip=%s", ip)
        return Response(status_code=204)

    if not isinstance(body, dict):
        return Response(status_code=204)

    kind = str(body.get("kind") or "manual")[:64]
    message = str(body.get("message") or "(no message)")[:1000]
    # 把识别的字段拆出来，rest 进 extra
    extra: Dict[str, Any] = {
        "client_kind": kind,
        "client_ip": ip,
        "client_url": str(body.get("url") or "")[:500],
        "client_user_agent": str(body.get("user_agent") or "")[:300],
        "client_app_version": str(body.get("app_version") or "")[:64],
        "client_build_hash": str(body.get("build_hash") or "")[:32],
        "client_ts": str(body.get("client_ts") or "")[:32],
    }
    # 可选 stack / componentStack — 截到合理长度防 log file 爆
    for k in ("stack", "componentStack", "source"):
        v = body.get(k)
        if v:
            extra[f"client_{k}"] = str(v)[:4000]
    for k in ("line", "col"):
        v = body.get(k)
        if isinstance(v, int):
            extra[f"client_{k}"] = v
    if body.get("trace_id_last_4xx"):
        extra["client_trace_id_last_4xx"] = str(body["trace_id_last_4xx"])[:64]

    # logger.error 进 studio.log；JsonLineFormatter 把 extra dict 摊到 .extra
    logger.error("[%s] %s", kind, message, extra=extra)
    return Response(status_code=204)


def _reset_rate_limit_for_tests() -> None:
    """测试钩子 — 清 ip_buckets 让每个测试独立。"""
    with _bucket_lock:
        _ip_buckets.clear()
