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
from .exception_handlers import register_exception_handlers
from .lifespan import lifespan
from .middleware import _SelectiveGZipMiddleware
from .trace_middleware import TraceIdMiddleware
from .routers import (
    announcements,
    browse,
    client_errors,
    data_exports,
    events_sse,
    generate,
    health,
    installs,
    logs,
    models,
    models_storage,
    presets,
    root,
    samples,
    secrets as secrets_router,
    studio_data,
    system,
    tag_dictionary,
    tagger,
    upscalers,
)
from .routers.projects import crud as projects_crud
from .routers.projects import exports as projects_exports
from .routers.projects import curation as projects_curation
from .routers.projects import eval_metrics as projects_eval_metrics
from .routers.projects import eval_samples as projects_eval_samples
from .routers.projects import ingestion as projects_ingestion
from .routers.projects import training as projects_training
from .routers.queue import io as queue_io_router
from .routers.queue import lifecycle as queue_lifecycle
from .routers.queue import outputs as queue_outputs

app = FastAPI(title="AnimaStudio", version=__version__, lifespan=lifespan)
# Middleware 注册顺序：starlette 后注册的 middleware 在 stack 外层。
# TraceIdMiddleware 最先注册 → 实际包在 GZip 外层 → trace_id 在 GZip
# 之前 bind，gzip handler 内 logger.x 也能拿到。
app.add_middleware(_SelectiveGZipMiddleware, minimum_size=1000)
app.add_middleware(TraceIdMiddleware)
# ADR-0009 PR-2 C2: 装 3 个 exception handler（DomainError / RequestValidation /
# Exception fallback）— dual-write envelope 保 detail contract + 加 error 结构化字段。
# HTTPException 不注册，让 starlette 默认 handler 跑保现有 175 处 raise 形状不变。
register_exception_handlers(app)

# Router 注册顺序无所谓（FastAPI 按 path 精确匹配，include_router 先后只影响
# include_in_schema=False 的 catch-all 顺序）。按 PR / 字母序排列方便审查。
# PR-5 commit 2: health / presets / browse / events_sse
app.include_router(health.router)
app.include_router(presets.router)
app.include_router(browse.router)
app.include_router(events_sse.router)
# ADR-0009 PR-3 C1: 前端错误上报 (ErrorBoundary / window.onerror / unhandledrejection)
app.include_router(client_errors.router)
# 公告栏（announcement-center Phase 1）：docs/announcements/ → GET /api/announcements
app.include_router(announcements.router)
# PR-6 commit 1: 5 个小 router（root / samples / logs / data_exports / tagger）
app.include_router(root.router)
app.include_router(samples.router)
app.include_router(logs.router)
app.include_router(data_exports.router)
app.include_router(tagger.router)
# PR-6 commit 2: admin router（secrets / models / upscalers）。jobs router 已删（R-5 台账合并，数据作业走 /api/queue）
app.include_router(secrets_router.router)
app.include_router(models.router)
app.include_router(upscalers.router)
app.include_router(tag_dictionary.router)
# PR-6 commit 3: installs router（10 routes: wd14/torch/flash-attn/xformers/llm-tagger admin）
app.include_router(installs.router)
# PR-6 commit 4: system router（11 routes: restart / update / rollback / preflight / etc.）
app.include_router(studio_data.router)
app.include_router(models_storage.router)
app.include_router(system.router)
# PR-6 commit 5: generate router（8 routes: 出图 + daemon 状态 + TAEFlux）
app.include_router(generate.router)
# PR-6 commit 6: queue 子包 3 文件（lifecycle 12 + io 3 + outputs 5 = 20 routes）
# 注册顺序：io 必须在 lifecycle 之前（FastAPI 按定义顺序匹配 path，"export" /
# "import" 字符串否则会被 `/api/queue/{task_id}` 的整数解析截胡 422）
app.include_router(queue_io_router.router)
app.include_router(queue_lifecycle.router)
app.include_router(queue_outputs.router)
# PR-6.5 commit 1: projects/versions CRUD 子包第一刀（16 routes）
app.include_router(projects_crud.router)
# ADR-0011: eval sample runs + manual run trigger (task-scoped)
app.include_router(projects_eval_samples.router)
# ADR-0011: metric result endpoints and concrete metric runner triggers
app.include_router(projects_eval_metrics.router)
# PR-6.5 commit 2: train.zip / bundle.zip / export-bundle / import-bundle (path/upload) /
# import-train（6 routes）
app.include_router(projects_exports.router)
# PR-6.5 commit 3: download/upload + preprocess (14 routes)
app.include_router(projects_ingestion.router)
# PR-6.5 commit 4: files/thumb + curation + duplicates (12 routes)
app.include_router(projects_curation.router)
# PR-6.5 commit 5: tag + captions + reg + reg_ai + version_config + queue training + version_thumb (23 routes)
app.include_router(projects_training.router)
