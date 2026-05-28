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
from .routers import (
    browse,
    data_exports,
    events_sse,
    health,
    installs,
    jobs,
    logs,
    models,
    presets,
    root,
    samples,
    secrets as secrets_router,
    tagger,
    upscalers,
)

app = FastAPI(title="AnimaStudio", version=__version__, lifespan=lifespan)
app.add_middleware(_SelectiveGZipMiddleware, minimum_size=1000)

# Router 注册顺序无所谓（FastAPI 按 path 精确匹配，include_router 先后只影响
# include_in_schema=False 的 catch-all 顺序）。按 PR / 字母序排列方便审查。
# PR-5 commit 2: health / presets / browse / events_sse
app.include_router(health.router)
app.include_router(presets.router)
app.include_router(browse.router)
app.include_router(events_sse.router)
# PR-6 commit 1: 5 个小 router（root / samples / logs / data_exports / tagger）
app.include_router(root.router)
app.include_router(samples.router)
app.include_router(logs.router)
app.include_router(data_exports.router)
app.include_router(tagger.router)
# PR-6 commit 2: 4 个 admin router（jobs / secrets / models / upscalers）
app.include_router(jobs.router)
app.include_router(secrets_router.router)
app.include_router(models.router)
app.include_router(upscalers.router)
# PR-6 commit 3: installs router（10 routes: wd14/torch/flash-attn/xformers/llm-tagger admin）
app.include_router(installs.router)
