"""根路径 + legacy `/studio/*` 兼容跳转（ADR 0012）。

routes：
    GET /                       dist 已构建 → 直接吐 index.html（SPA 入口，不再重定向）；
                                dist 缺失 → 返 JSON 构建提示
    GET /studio                 一次性 legacy 跳转 → / （307，保留 query）
    GET /studio/{rest:path}     一次性 legacy 跳转 → /{rest}

历史上 SPA 挂在 `/studio/` 子路径、`/` 302 跳过去（给已删除的 monitor_smooth.html
等根级页面让路）。ADR 0012 起 SPA 挂回根路径：`/` 不再重定向，消除「反代容器端口」
平台（ModelScope 创空间 / HF Space）网关尾斜杠归一化叠加 mount 307 的死循环（issue
#330）。`/studio/*` 跳转保留一个 release，给老书签 / 自更新后仍停在 `/studio/...` 的
老标签页一个平滑落点，下版删除。
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from ...paths import WEB_DIST

router = APIRouter()


@router.get("/", response_model=None, include_in_schema=False)
def root() -> FileResponse | JSONResponse:
    """根路径 SPA 入口。

    dist 已构建 → 直接返回 index.html（ADR 0012：不再 302 跳 /studio/）；
    其余前端资源 / 深链由 server.py 挂在 `/` 的 SPAStaticFiles 兜底。
    dist 缺失 → 返回 JSON 提示。"""
    index = WEB_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {
            "message": "AnimaStudio is running. Build the React app at studio/web/ "
            "(npm install && npm run build) to enable the new UI."
        }
    )


@router.get("/studio", response_model=None, include_in_schema=False)
@router.get("/studio/{rest:path}", response_model=None, include_in_schema=False)
def legacy_studio_redirect(request: Request, rest: str = "") -> RedirectResponse:
    """ADR 0012 legacy：老 `/studio/...` 链接一次性 307 跳到根路径，保留一个 release。

    307 保 method + 不被永久缓存（下版删除时不会有顽固 301 缓存残留）；query 透传。"""
    qs = ("?" + request.url.query) if request.url.query else ""
    target = "/" + rest.lstrip("/") if rest else "/"
    return RedirectResponse(url=target + qs, status_code=307)
