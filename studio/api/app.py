"""FastAPI 应用工厂（PR-5 从 server.py 抽出）。

只创建 `app` 实例并配 middleware + lifespan。**路由注册不在这里**：
    - 老路由仍在 `studio/server.py` 通过 `@app.get(...)` 装饰到本实例
      （PR-5 commit 1 范围；后续 commit 把 router 逐批搬到 api/routers/）
    - 新路由走 `api/routers/<name>.py` + `app.include_router(...)`

只要至少 import 一次 `studio.server`，全部 130 个老 route decorator
就会注册到本 app 上，`uvicorn studio.server:app` 或 `studio.api.app:app`
启动都拿到同一 FastAPI 实例。
"""
from __future__ import annotations

from fastapi import FastAPI

from .. import __version__
from .lifespan import lifespan
from .middleware import _SelectiveGZipMiddleware

app = FastAPI(title="AnimaStudio", version=__version__, lifespan=lifespan)
app.add_middleware(_SelectiveGZipMiddleware, minimum_size=1000)
