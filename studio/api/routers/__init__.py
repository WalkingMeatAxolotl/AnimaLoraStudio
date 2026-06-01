"""HTTP routers — PR-5 起从 studio/server.py 逐批搬过来。

每个文件 = 一个域。api/app.py 一次性 `app.include_router` 全部。

## 16 顶层 router + 1 子包（27 文件）

| router 文件 | routes | 说明 |
|---|---:|---|
| `health.py`        | 3   | health / system stats / training monitor state |
| `presets.py`       | 13  | preset CRUD + import/export + schema + configs redirect |
| `browse.py`        | 3   | datasets / browse / dataset thumbnail |
| `events_sse.py`    | 1   | /api/events SSE |
| `root.py`          | 1   | / → /studio/ 302 redirect |
| `samples.py`       | 1   | /samples/{filename} |
| `logs.py`          | 1   | /api/logs/{task_id} |
| `data_exports.py`  | 1   | /api/data-exports |
| `tagger.py`        | 1   | /api/tagger/{name}/check |
| `jobs.py`          | 3   | /api/jobs/{jid} / log / cancel |
| `secrets.py`       | 2   | secrets read/update |
| `models.py`        | 3   | models catalog / path-defaults / download |
| `upscalers.py`     | 2   | upscaler select / custom download |
| `installs.py`      | 10  | wd14 + torch + flash-attn + xformers + llm-tagger admin |
| `system.py`        | 11  | restart / update / rollback / preflight / dev_commits / release_notes |
| `generate.py`      | 8   | 测试出图 + daemon 控制 + TAEFlux |
| `queue/`           | 20  | 内拆 lifecycle (12) / io (3) / outputs (5) |
| `projects/`        | 71  | 内拆 crud (16) / exports (6) / ingestion (14) / curation (12) / training (23) |

**合计**：155 routes（+ 5 非 APIRoute：SPA mount + openapi/docs/redoc 等）= 160。

## 子包内部 helpers

- `queue/__init__.py` — 子包说明
- `projects/_shared.py` — projects 域 8 个共用 helper（_project_payload /
  _publish_*_state / _version_dir_or_404 / 等），仅 projects sub-router 内部 import

## include 顺序约束

queue/ 的 io 必须在 lifecycle 之前 include（FastAPI 按 path 定义顺序匹配，
`/api/queue/export` / `/api/queue/import` 否则会被 `/api/queue/{task_id}` 的
整数解析截胡 422）。
"""
