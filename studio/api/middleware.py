"""FastAPI middleware（PR-5 从 server.py 抽出）。"""
from __future__ import annotations

from fastapi.middleware.gzip import GZipMiddleware

# 给 JSON / 文本响应自动 gzip 压缩。对 /api/state（10k 步训练 ~500KB → ~100KB）
# 这种大数组高度可压；对小响应 (< 1000B) 跳过避免 framing overhead 反而变大。
#
# 显式排除两类路径：
#   - /api/events：SSE 流。GZipMiddleware 会 buffer chunks 到 minimum_size
#     才发，破坏 SSE 实时性 + 某些 EventSource 实现解析 gzip 流有兼容问题
#   - /samples/*：图片字节（PNG/JPEG/WEBP 已经是压缩格式），gzip 再压净浪费
#     CPU 且 size 略增
_GZIP_SKIP_PREFIXES = ("/api/events", "/samples/")


class _SelectiveGZipMiddleware(GZipMiddleware):
    """GZipMiddleware 的子类，按 path 前缀绕过指定路由。"""

    async def __call__(self, scope, receive, send):  # type: ignore[override]
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if any(path.startswith(p) for p in _GZIP_SKIP_PREFIXES):
                await self.app(scope, receive, send)
                return
        await super().__call__(scope, receive, send)
