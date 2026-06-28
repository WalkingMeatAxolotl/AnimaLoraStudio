"""SPA 静态文件 mount（PR-5 从 server.py 抽出）。"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Windows 注册表常把 .js 标成 text/plain（IIS / 杀软 / 旧装机污染），
# 导致 ES module strict MIME check 拒载 → 启动白屏。issue #228
# import time 主动覆盖，确保跨平台一致。
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/svg+xml", ".svg")

# ADR 0012：SPA 挂到根路径后，本 catch-all 接管整个 URL 空间。这些是服务端拥有
# 的根级命名空间——其下未命中的请求必须保持干净 404，不能被 index.html 兜底
# （否则 /api/typo 这类拼写错误会返 200 误导调用方）。
# **新增非 /api 的根级路由时，必须把其首段补进这里。**
_RESERVED_SEGMENTS = frozenset({"api", "samples"})


class SPAStaticFiles(StaticFiles):
    """SPA 路由兜底：未命中实际文件且不像静态资产时，返回 index.html。

    这样直接刷新 `/studio/projects/1/v/1/curate` 这种 react-router 路由
    也能拿到 index.html，让 BrowserRouter 在前端解析路径。
    带文件扩展名的请求（.js/.css/.png 等）保持原 404 行为，避免把缺失的
    资源吞成 200 误导浏览器。
    """

    async def get_response(self, path, scope):  # type: ignore[override]
        from starlette.exceptions import HTTPException as StarletteHTTPException
        # StaticFiles.get_path 走 os.path.normpath，Windows 上分隔符是反斜杠——
        # 先归一化成 "/" 再判断，否则下面的命名空间 / 扩展名判定在 Windows 失效。
        norm = path.replace("\\", "/")
        # 服务端命名空间（/api、/samples）下未命中显式路由的请求 → 统一干净 404
        # （任何 method），不进 StaticFiles（ADR 0012）。必须在 super() 之前拦：
        # StaticFiles 对非 GET/HEAD 会先判 405，且不会兜底 index.html——挂到根路径
        # 后 `PUT /api/typo` 这类未匹配请求会变 405 而非应有的 404。
        first_seg = norm.split("/", 1)[0]
        if first_seg in _RESERVED_SEGMENTS:
            raise StarletteHTTPException(status_code=404)
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            # 末段含 "." → 视为静态资产请求，不兜底
            last = norm.rsplit("/", 1)[-1]
            if "." in last:
                raise
            return FileResponse(Path(self.directory) / "index.html")
