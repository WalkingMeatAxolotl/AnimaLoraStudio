"""共享响应常量 / 响应工厂（PR-5 起从 server.py 抽出）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse

from .. import thumb_cache

# /api/state 在 task_id 不存在 / 没 task / state.json 缺失时返回的空 state，
# 保持前端 monitor 页能稳定渲染（不报错也不显示 "loading"）。
EMPTY_STATE: dict[str, Any] = {
    "losses": [],
    "lr_history": [],
    "epoch": 0,
    "total_epochs": 0,
    "step": 0,
    "total_steps": 0,
    "speed": 0.0,
    "samples": [],
    "start_time": None,
    "config": {},
}


def _thumb_response(src: Path, size: int) -> FileResponse:
    """统一 thumb 响应：弱 etag（基于 src mtime+size）+ no-cache 强制重验。

    早先用 `Cache-Control: public, max-age=86400` 会让浏览器记住所有响应 24h，
    包括重启过渡期的失败响应；用户视角就是「重启后图片加载不了」。改用 etag +
    no-cache 后，浏览器每次发条件请求，命中走 304 几 ms，错过响应不再阻塞。

    PR-6：从 server.py 抽到 api/responses.py 给 samples router 和 server.py 内的
    project_thumb（PR-6.5 之前还留 server.py）共用。
    """
    out = thumb_cache.get_or_make_thumb(src, size)
    try:
        mtime_ns = out.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    etag = f'W/"{mtime_ns}-{size}"'
    return FileResponse(
        out,
        headers={
            "Cache-Control": "no-cache, must-revalidate",
            "ETag": etag,
        },
    )
