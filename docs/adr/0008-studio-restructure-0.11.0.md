# 0008 — studio/ 4 层重构（0.11.0）

**状态**：Accepted
**日期**：2026-05-28
**决策者**：@WalkingMeatAxolotl

## 决策

把 `studio/` 从 25k 行平铺单包重构为 4 层架构。`server.py` 4657 → 51 行，130 个 `@app` 装饰器分到 27 个 router 文件，全部 routes / models / 业务 / 基础设施按职责分包。

本 ADR 主要用途：**告诉未来贡献者代码应该写在哪里**。

---

## 当前结构（0.11.0 起）

```
studio/
├── api/                  HTTP 表面（FastAPI）
│   ├── app.py            FastAPI 实例 + middleware + 全部 include_router
│   ├── lifespan.py       startup/shutdown：ensure_dirs / db.init_db / supervisor 启停 / SSE
│   ├── main.py           uvicorn 启动入口（main()）
│   ├── middleware.py     _SelectiveGZipMiddleware
│   ├── errors.py         4 个 HTTPException helper（_safe_join_or_400 / _preset_err_code / ...）
│   ├── responses.py      EMPTY_STATE 常量 + _thumb_response
│   ├── static.py         SPAStaticFiles（react-router 兜底）
│   ├── deps.py           跨 router 共享 helper（_supervisor / _resolve_anima_model_paths）
│   ├── schemas/          每 router 的 inline BaseModel 抽出
│   │   ├── presets.py / models.py / installs.py / system.py / generate.py
│   │   ├── queue.py / curation.py / ingestion.py / projects.py / exports.py / training.py
│   └── routers/          27 个 router 文件
│       ├── health.py / presets.py / browse.py / events_sse.py
│       ├── root.py / samples.py / logs.py / data_exports.py / tagger.py
│       ├── jobs.py / secrets.py / models.py / upscalers.py / installs.py / system.py / generate.py
│       ├── queue/        子包：lifecycle (12) + io (3) + outputs (5)
│       └── projects/     子包：crud (16) + exports (6) + ingestion (14) + curation (12) + training (23)
│                         └── _shared.py（projects 域内共用 helper）
│
├── services/             业务服务（不依赖 db；db 调用在 caller 端）
│   ├── booru/            api + pool + downloader
│   ├── tagging/          wd14 / cltagger / llm / joycaption / caption_format / caption_snapshot
│   │                     / onnx_base / base (tagger factory)
│   ├── reg/              builder + analysis + postprocess
│   ├── inference/        core (LoRA apply) + daemon + cache + upscaler
│   ├── models/           catalog + paths + sources + downloader (PR-3.8 4-way 拆)
│   ├── preprocess/       core + duplicates + manifest
│   ├── projects/         projects + versions + jobs + phase + curation
│   ├── dataset/          scan + browse + thumb_cache + tagedit + uploads
│   ├── presets/          io + fork/save 流程
│   ├── runtime/          onnxruntime / torch / flash_attention / xformers / pending_install / updater
│   ├── data_io/          train_io（train.zip / bundle.zip 导入导出）
│   ├── queue_io.py       queue 任务 import/export
│   ├── task_snapshot.py  task 启动 freeze config
│   ├── version_config.py per-version yaml config CRUD
│   ├── release_notes.py  release_notes.yaml 解析
│   └── system_stats.py   CPU/GPU 采样
│
├── domain/               pydantic 模型（前端 schema 契约源）
│   ├── training.py       TrainingConfig（643 行单类，PR-2 决策不拆 mixin）
│   ├── lora.py           LoraEntry
│   ├── xy_matrix.py      XYAxisSpec / XYMatrixSpec
│   ├── generate.py       GenerateConfig
│   ├── reg.py            RegAiConfig
│   ├── migrations.py     字段名 / 字段值历史迁移
│   └── common.py         GROUP_ORDER / AttentionBackend
│
├── infrastructure/       路径 / DB / 配置 / 日志
│   ├── paths.py          REPO_ROOT 等路径常量 + safe_join / validate_path_component
│   ├── db.py             SQLite 连接 + tasks 表 CRUD
│   ├── migrations/       _v2 ~ _v9 schema migration
│   ├── secrets.py        secrets.yaml 全部 model + load/save + legacy migration
│   ├── event_bus.py      进程内 SSE 总线
│   ├── log_tail.py       per-task 日志增量读 + monitor state 轮询
│   ├── argparse_bridge.py pydantic 模型 → argparse 参数派生
│   └── llm_presets.py    builtin LLM caption preset 加载
│
├── supervisor/           任务调度守护线程（跨层使用）
│   ├── core.py           Supervisor 主类（1100 行单类，PR-4 决策不拆 mixin）
│   ├── slot.py           _Slot dataclass
│   ├── cmd_builder.py    默认 cmd builder + monitor_state_path
│   ├── finalizer.py      task 终态 → version.status 映射
│   └── process.py        _kill_process_tree
│
├── workers/              4 个子进程入口（跨层使用）
│   ├── _base.py          worker_main + reconfigure_console_utf8 公共模板
│   ├── download_worker.py / tag_worker.py / preprocess_worker.py / reg_build_worker.py
│
├── api/__init__.py 等 4 个 docstring 加强  目录导航（详 PR-9）
├── server.py             51 行 shim（FastAPI app + main + SPA mount + test fixture re-imports）
├── db.py / paths.py / secrets.py  各 10 行 shim（test fixture monkeypatch 兼容）
├── cli.py                841 行 launcher（python -m studio run/dev/build/test）— 未来 0.11.1 拆
├── __init__.py           __version__ + 顶层架构 docstring
├── __main__.py           python -m studio 入口
├── llm_presets/*.json    builtin LLM caption preset 数据
└── web/                  React 前端（Vite 构建）
```

**层依赖方向（严格单向，不允许反向）**：

```
api/  →  services/  →  domain/
                ↓
        infrastructure/
```

`supervisor/` 和 `workers/` 是跨层使用者（不归 4 层之一）。

---

## 未来开发指南

### 加新的 HTTP route

1. 找 `api/routers/<domain>.py` 或 `api/routers/<domain>/` 子包；没有就新建
2. inline `BaseModel` 全部抽到 `api/schemas/<domain>.py`
3. 路由 handler 不直接调 db / file system；通过 `services/` 调
4. 在 `api/app.py` 加 `app.include_router(<domain>.router)`
5. 跨 router 共用 helper → `api/deps.py`；单域内 helper → `api/routers/<domain>/_shared.py`
6. 测试用 FastAPI TestClient，fixture monkeypatch 走 **新模块路径**（不是 `server.X`）

**路由顺序约束**：FastAPI 按 path 定义顺序匹配。`/api/queue/export` 必须在 `/api/queue/{task_id}` 之前 include（否则 "export" 被当 task_id 整数解析报 422）。已在 `api/app.py` 注释 codify。

### 加新的业务功能

在 `services/<domain>/` 下加 module。规则：
- 纯函数 + dataclass，不依赖 fastapi / db connection（db connection 由 caller 传入）
- 失败抛 domain-specific Exception（如 `CurationError`），让 api/routers 翻译成 HTTPException
- 跨 service 共享 helper → 抽到 `services/<domain>/__init__.py` 或新建 `services/<domain>/_shared.py`

### 加新的数据模型

加到 `domain/<name>.py` 作独立 pydantic BaseModel。如果是历史字段名 / 字段值迁移，加到 `domain/migrations.py`。

**注意**：`TrainingConfig` 643 行单类**不要拆 mixin**（PR-2 决策 — 影响 pydantic field declaration order，破坏前端 schema 字段顺序）。新字段直接加到 `domain/training.py` 现有字段组。

### 加新的基础设施

如新增 path 常量 / db helper / event 类型，加到 `infrastructure/<相应模块>.py`。**不要在 studio/ 顶层加新文件**。

### 加新的子进程 worker

1. 写 `studio/workers/<name>_worker.py`：定义 `run(job_id) -> int`
2. 底部 `if __name__ == "__main__": from ._base import worker_main; worker_main(run)`
3. 不直接读写 db connection；通过 `studio.infrastructure.db.connection_for()`
4. 日志只走 stdout（supervisor 重定向到 log 文件）
5. supervisor 通过 `python -m studio.workers.<name>_worker --job-id N` 启动

### 测试

- **不要** patch `server.X`（除非是永久 shim 的 X 之一：`db / paths / secrets / OUTPUT_DIR / WEB_DIST / STUDIO_DB / USER_PRESETS_DIR / LOGS_DIR / REPO_ROOT`）；直接 patch handler 内 import 的真实模块
- 跨 router fixture 复用 → `conftest.py`
- 涉及 supervisor 的 test 用 `_StubSupervisor` 注入 `app.state.supervisor`（参考 `tests/test_studio_queue_endpoints.py`）

### 永久保留的 4 shim（**不要删**）

- `studio/server.py` (51 行) — FastAPI `app` + `main` + SPA mount + `HTTPException` + 6 path 常量 + `db` re-import 给 test fixture
- `studio/db.py` / `studio/paths.py` / `studio/secrets.py` (各 10 行) — test fixture `monkeypatch.setattr(server.db, ...)` 大量使用

这 4 个是架构的一部分，删除 = 改 30+ 测试文件 + 高 ImportError 风险，ROI 不值。其它 45 个 PR-3/7 留下的 shim 已在 PR-9 全删完。

### 命名 / 约定速查

| 主题 | 约定 |
|---|---|
| 新 router 文件 | `api/routers/<domain>.py`，对应 `api/schemas/<domain>.py` |
| 新业务 service | `services/<domain>/<feature>.py`，导出函数 + dataclass |
| 新 pydantic 模型 | `domain/<name>.py` |
| 新 path / db / config | `infrastructure/<topic>.py` |
| 新 worker | `studio/workers/<kind>_worker.py` + `worker_main(run)` |
| inline BaseModel | 抽到 `api/schemas/`，handler 文件不 inline |
| 跨 router helper | `api/deps.py`（真正跨域）或 `api/routers/<domain>/_shared.py`（域内）|
| Exception | 业务层抛 domain-specific（`CurationError`）；router 层翻 HTTPException |
| 错误码 | `api/errors.py:_preset_err_code` 模板：消息字符串 → HTTP code |

---

## 实施 lessons（PR-1..9 累积）

代码 review 时按需引用。完整背景在 git log + PR description。

1. **`sys.modules` 别名 shim 模式** — 包同名覆盖时 `_sys.modules[__name__] = _real` 让旧路径透明转发（PR-3）
2. **跨子模块调用走 `module.func()`** — `from .sub import func` 把名 bind 成本模块名，test patch `sub.func` 失效；改 `from . import sub as _sub; _sub.func()`（PR-3.8）
3. **fixture patch path 跟搬迁** — handler 搬走后老 `monkeypatch.setattr(server, X)` 看不到新位置；fixture 加新位置 patch（PR-5/6 反复）
4. **package shim 也用 module 文件 + sys.modules** — `studio/migrations.py` 单文件代理 `studio.infrastructure.migrations` 整个 package，子模块访问透明（PR-7）
5. **搬深一层补 `.parent`** — `Path(__file__).resolve().parent.parent` 搬到子目录后要加 `.parent`（PR-7）
6. **`del sys.modules + reimport` 在共享 app 实例下污染** — 重 import server.py 让 `@app` 装饰器对同一 app 重复注册，routes 数翻倍（PR-6）
7. **状态耦合高的类不拆** — `TrainingConfig` (pydantic order) / `Supervisor` (共享 self) 保单类，拆 mixin 增加未来扩展成本（PR-2/4）
8. **删 shim：fail loud not lazy fallback** — 不用 `__getattr__` lazy 回退（隐藏 ImportError），每 caller 显式改 canonical path（PR-9）

---

## 还的债（0.11.1+）

| 项 | 推迟理由 |
|---|---|
| `cli.py` 841 → 7 文件拆 | launcher 单文件可读；独立 PR 隔离风险 |
| `secrets.py` 763 → models/store/migrations 3 文件 | Pydantic v2 跨文件循环风险；3-way 收益主要是视觉隔离 |
| `db.py` 188 → connection/tasks/settings 3 文件 | 同上 |
| 统一 exception handler 替代 4 套 `err_code` helper | 行为变更，独立 PR 安全 |
| `secrets.py` 170 行 legacy migration 加 deprecation log | 跟 3-way 拆一起 |

---

## 参考

- 11 个 PR：#141 #142 #143 #144 #145 #147 #148 #149 #150 #151 #152 #153
- 关联 ADR：[#0003 anima_train.py 模块化重构](0003-anima-train-refactor.md)（同性质 runtime 端拆分）
- 关键文件 / 索引：`studio/__init__.py` / `studio/services/__init__.py` / `studio/api/routers/__init__.py` 三个 docstring 是顶层导航
