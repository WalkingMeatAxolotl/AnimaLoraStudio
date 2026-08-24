"""共享响应常量 / 响应工厂（PR-5 起从 server.py 抽出）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from fastapi import BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from ..services.dataset import thumb_cache

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


def packaged_zip_response(
    path: Path, filename: str, background: BackgroundTasks,
) -> StreamingResponse:
    """现场打包的 zip 一次性下载：整流发送，显式不支持 Range。

    这类 zip 每次请求都重新生成，两次生成的字节不保证一致（manifest 里有
    time.time() 时间戳，float repr 长度还会波动 ±1 字节）。FileResponse 默认
    宣告 `Accept-Ranges: bytes` 并响应 206——多线程下载器（IDM / Ghost
    Downloader 等，接管 localhost 一样生效）按 Range 分块并行下载时，各块
    来自不同的打包实例，拼出来的 zip 必然损坏（真实案例：分段间错位 ±1
    字节，导入报「文件已损坏」）。所以忽略 Range 头 + `Accept-Ranges: none`，
    下载器退化为单连接整流下载，字节一致性由单次打包保证。

    Content-Length 显式给出（打包已完成、大小固定），浏览器保留进度条。
    """
    size = path.stat().st_size

    def _iter() -> Iterator[bytes]:
        with path.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                yield chunk

    # filename 可能含非 ASCII（task/project 名），同 starlette 的编码策略
    quoted = quote(filename)
    disposition = (
        f"attachment; filename*=utf-8''{quoted}"
        if quoted != filename
        else f'attachment; filename="{filename}"'
    )
    return StreamingResponse(
        _iter(),
        media_type="application/zip",
        headers={
            "Content-Length": str(size),
            "Content-Disposition": disposition,
            "Accept-Ranges": "none",
        },
        background=background,
    )


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
