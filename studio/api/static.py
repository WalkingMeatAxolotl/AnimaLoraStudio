"""SPA 静态文件 mount（PR-5 从 server.py 抽出）。"""
from __future__ import annotations

from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


class SPAStaticFiles(StaticFiles):
    """SPA 路由兜底：未命中实际文件且不像静态资产时，返回 index.html。

    这样直接刷新 `/studio/projects/1/v/1/curate` 这种 react-router 路由
    也能拿到 index.html，让 BrowserRouter 在前端解析路径。
    带文件扩展名的请求（.js/.css/.png 等）保持原 404 行为，避免把缺失的
    资源吞成 200 误导浏览器。
    """

    async def get_response(self, path, scope):  # type: ignore[override]
        from starlette.exceptions import HTTPException as StarletteHTTPException
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            # 末段含 "." → 视为静态资产请求，不兜底
            last = path.rsplit("/", 1)[-1]
            if "." in last:
                raise
            return FileResponse(Path(self.directory) / "index.html")
