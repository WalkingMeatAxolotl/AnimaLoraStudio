"""根路径 redirect（PR-6 commit 1 从 server.py 抽出）。

1 route：
    GET /    302 → /studio/（前端 SPA 入口），dist 缺失时返 JSON 提示
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, RedirectResponse

from ...paths import WEB_DIST

router = APIRouter()


@router.get("/", response_model=None)
def root() -> RedirectResponse | JSONResponse:
    """根路径 302 跳转到 React 应用 `/studio/`。

    若前端尚未构建（dist 缺失），返回 JSON 提示。"""
    if WEB_DIST.exists():
        return RedirectResponse(url="/studio/", status_code=302)
    return JSONResponse(
        {
            "message": "AnimaStudio is running. Build the React app at studio/web/ "
            "(npm install && npm run build) to enable the new UI."
        }
    )
