# 0009 — 统一日志 + 错误体系（0.12.0）

**状态**：Accepted
**日期**：2026-05-28
**决策者**：@WalkingMeatAxolotl（三方 agent review：架构师 / 审计员 / 迁移策略师，各两轮）
**落地**：PR #155 (PR-1 后端基础设施) / PR #156 (PR-2 错误体系) / PR #TBD (PR-3 前端 + CLI 收尾)

## 决策

把 5 surface（后端 Python / API HTTP / 前端 React / Subprocess workers / CLI）的日志与错误处理统一到：

- **stdlib `logging`** + `concurrent-log-handler`（唯一新依赖，跨进程文件锁）
- **ContextVar + HTTP header + 子进程 env** 三路同步传 `trace_id`（ULID 26 字符）
- **`DomainError` 基类** + 5 子类（NotFound / Validation / Conflict / Auth / Forbidden）放 `studio/domain/errors.py`
- **3 个 `exception_handler`** 统一翻译（DomainError / RequestValidationError / Exception fallback）
- **错误响应 envelope 走 dual-write**：保 `{"detail": ...}` legacy contract，新增 `{"error": {"code", "message", "trace_id"}}` 平行字段，分 3 release 渐进迁移
- **3 PR 落地**（后端基础设施 / 错误体系 / 前端+CLI），约 72h ≈ 2 工作周

本 ADR 主要用途：**告诉未来贡献者如何打日志、如何抛错、如何加新 surface 接入这套体系**。

同时关闭 ADR-0008 §"还的债" 第 4 项（"统一 exception handler 替代 4 套 err_code helper"）。

---

## 背景

### 现状（0.11.0 之后）

5 个 surface 的日志/错误处理状态（B agent 审计 31 个问题，P0 × 4 / P1 × 14 / P2 × 13）：

| Surface | 关键痛点 |
|---|---|
| 后端 Python | 21 文件 `getLogger(__name__)` + 101 处 `logger.x` 调用，**0 处 `basicConfig` / FileHandler** → INFO 全被 root WARNING 过滤；`except Exception: pass` 散落 13+ 文件 |
| API HTTP | **0 个 `exception_handler` 注册**；4 套独立 `_err_code` helper 靠中文字符串匹配（"不存在" → 404）；380 处 router try/except 重复模板 |
| 前端 React | `ErrorBoundary` 只 `console.error` 不上报；**0 处 `window.onerror` / `unhandledrejection`** |
| Subprocess workers | 4 worker 用 `print()` 当日志（无 level / ts）；`__EVENT__:` malformed payload 静默丢；`db.tasks.error_msg` 只存 `"exit code 1"` |
| CLI | 48 处 `print()`，无 verbose 级别控制 |
| **跨 surface** | **完全无 trace_id / request_id 概念** — 用户操作触发的 "router → service → supervisor → worker → SSE → toast" 全链路无贯穿 ID，oncall 无法 join |

### 触发动机

- ADR-0008 §"还的债" 第 4 项已经标记要统一 exception handler，但单独做会再次错过 trace_id 这条命脉
- 0.11.0 重构完成后架构清晰，是引入 cross-cutting 体系的好时机
- 用户报问题时只能给 task_id，开发者翻 4 处日志（jobs/<id>.log / uvicorn stderr / 浏览器 devtools / daemon ring buffer）反推时间窗口

---

## 当前结构（本 ADR 落地后）

```
studio/
├── infrastructure/
│   └── logging.py                # ⭐ 新建。单文件 ~200 LOC。
│                                 # 暴露：setup_logging / bind_trace_id / get_trace_id / new_trace_id
│                                 #       TRACE_HEADER / TRACE_ENV / PROCESS_ENV
│                                 # 内含：JsonLineFormatter / HumanConsoleFormatter / RotatingFileHandler 配置
│                                 #       contextvars._trace_id_var / _process_var / _job_id_var / _task_id_var
│                                 #       第三方库静音 list（asyncio/urllib3/PIL/...）
│                                 #       uvicorn.access / uvicorn.error handler 接管
│
├── domain/
│   └── errors.py                 # ⭐ 新建。DomainError 基类 + 5 子类。
│                                 # 不依赖 fastapi，纯 Python；services 可直接 raise
│
├── api/
│   ├── middleware/
│   │   └── trace.py              # ⭐ 新建。TraceIdMiddleware（pure ASGI）
│   │                             # 读 X-Trace-Id header → ContextVar → 写 response header
│   ├── exception_handlers.py     # ⭐ 新建。register_exception_handlers(app)
│   │                             # 注册 3 个：DomainError / RequestValidationError / Exception fallback
│   ├── errors.py                 # 改造。原 4 套 _err_code helper 退化为 thin wrapper（PR-2 内 8 commit 渐进删完）
│   ├── routers/
│   │   └── client_errors.py      # ⭐ 新建。POST /api/client-errors（前端 ErrorBoundary 上报）
│   └── app.py                    # 调 setup_logging("webui") + register_exception_handlers + 装 TraceIdMiddleware
│
├── supervisor/
│   └── core.py                   # 改造：_popen 注入 ANIMA_TRACE_ID + ANIMA_PROCESS_NAME env
│                                 #       _finish_slot tail jobs/<id>.log 末 10 行回写 db.tasks.error_msg
│                                 #       dispatcher spawn 时 var.set(task.request_trace_id)
│
├── workers/
│   ├── _base.py                  # 改造：worker_main() 开头调 setup_logging("worker:<kind>/<job_id>")
│   │                             #       从 ANIMA_TRACE_ID env 读 trace_id bind contextvar
│   │                             #       reconfigure_console_utf8 收编进 setup_logging
│   └── *_worker.py               # print → logger（保 __EVENT__: IPC 行不动）
│
├── services/
│   └── inference/daemon.py       # 改造：_read_stderr_loop thread 死亡 watchdog（restart 一次或标 STOPPED）
│
├── infrastructure/
│   ├── event_bus.py              # 改造：_safe_put QueueFull 加 logger.warning（不再静默）
│   └── migrations/
│       └── _vN_request_trace.py  # ⭐ 新 migration：tasks 表加 request_trace_id TEXT 列
│
├── cli.py                        # 改造：加 _say(msg, level="info") wrapper；48 处 print 走 _say
└── web/src/
    ├── components/ErrorBoundary.tsx     # 改造：componentDidCatch 上报到 /api/client-errors
    ├── main.tsx                          # 改造：装 window.addEventListener('error'|'unhandledrejection')
    ├── lib/errors/setup.ts               # ⭐ 新建。三路捕获 + reportClientError
    ├── lib/errors/report.ts              # ⭐ 新建。POST 上报（silent swallow on fail）
    └── api/client.ts                     # 改造：req() / xhrUpload() / importPreset 三处合一吃 X-Trace-Id
                                          #       toast 显示 "trace ab12cd34" 后缀
```

### 5 surface 数据流

```
┌─ 前端 (browser) ─────────────────────────────────────────────────┐
│ ErrorBoundary ─┐                                                  │
│ window.error  ─┼──► reportClientError ──► POST /api/client-errors │
│ unhandledrej. ─┘    (attach lastTraceId from atom)               │
└────────────────────────────────┬─────────────────────────────────┘
                                  │ HTTP + X-Trace-Id header (双向)
┌─────────────────────────────────▼────────────────────────────────┐
│ FastAPI (process="webui")                                        │
│   TraceIdMiddleware: header → ContextVar → response              │
│   exception_handler(DomainError) → JSON {detail:{...}, error:{}} │
│   logger.x ──► JsonLineFormatter ──┐                             │
└────────────────┬───────────────────│─────────────────────────────┘
                 │ supervisor.spawn   │
                 │ env: ANIMA_TRACE_ID=<id from task.request_trace_id>
                 │      ANIMA_PROCESS_NAME=worker:tag/42            │
                 ▼                    ▼
┌──────────────────────────┐    ┌───────────────────────────────────┐
│ Worker subprocess         │    │ logs/studio.log                  │
│  setup_logging("worker..")│    │  {ts, level, process, trace_id,  │
│  ContextVar bind          │    │   logger, msg, exc, extra}       │
│  stdout ──► supervisor    │    │  rotated *.1 ~ *.5 (50MB 每份)   │
│   redirect log_fp         │    └───────────────────────────────────┘
└──────────────┬────────────┘                    ▲
               │ stdout/stderr                    │ 单写到 jobs/<id>.log
               ▼                                  │ + supervisor logger 也写 studio.log
       logs/jobs/<task_id>.log ───────────────────┘
       (人读，给 SSE LogTailer + 前端 <pre>)
```

层依赖（严格单向）：

```
api/middleware/  ────►  api/exception_handlers  ────►  domain/errors
       ▲                         │
       │                         ▼
       │              infrastructure/logging
       │                         ▲
       └──── services/ ──────────┘
              ▲
       supervisor/, workers/, cli/
```

`domain/errors` 不依赖 fastapi（纯 Exception 子类），让 services 可以 raise 而不反向 import api。

---

## 三方 review 后的关键裁决

| 议题 | 候选 | 最终选择 | 否决理由 |
|---|---|---|---|
| 日志库 | stdlib / loguru / structlog | **stdlib + concurrent-log-handler** | loguru/structlog 是运行期硬依赖；subprocess 启动开销 + caplog 不正交；JSON schema 简单 50 行自写 |
| `trace_id` 长度/格式 | uuid4 hex[:16] / ULID 26 | **ULID 26**（含 `bg-{ULID}` 后台 spawn 前缀，长度统一） | uuid4 无 lexicographic sort；bg-uuid8 vs ULID 长度不一致 grep 痛 |
| `trace_id` 传播路径 | header / env / contextvar / db 列 | **四路全开** | header 仅前端拿；env 子进程拿；contextvar 同进程；db 列让 supervisor 后台 dispatcher 拿（请求时刻 ID = spawn 时刻 ID = worker log ID） |
| 错误 envelope | 新 schema / 保 contract / dual-write | **dual-write**（详 §envelope 渐进迁移） | 前端 3 解析点 + 11 测试断言依赖 `detail`；新 schema 立刻炸；header-only 用户截图不便 |
| `DomainError` 位置 | api/ / domain/ | **`domain/errors.py`** | services 反向依赖 api 是反模式；domain/ 纯 Exception 子类不带 fastapi |
| 子类数量 | 5 / 7 / 更多 | **5 核心**（NotFound/Validation/Conflict/Auth/Forbidden） | 7 子类一次落 services 7 文件都要改 base，diff 涨 200+ 行；剩余渐进 |
| Worker log | 单写 / 双写 | **0.12.0 单写**（worker stdout → supervisor 重定向） | Windows 跨进程文件锁复杂；supervisor `_finish_slot` 回写 `error_msg` 已解决 1.6 痛点；双写到 0.13.x 再演进 |
| `infrastructure/logging` | 单文件 / 子包 | **单文件 ~200 LOC** | 超 400 LOC 再拆子包；当前体量子包是 ceremony overhead |
| CLI 输出 | print / logger | **保 print + `_say()` wrapper** | CLI 5s 短命周期落盘价值低；用户终端看 `[studio] ...` 比 logger 默认 format 清爽；capsys 7 文件改"in 模糊匹配"代价低 |
| 第三方库 logger 噪音 | 一个个 silence / 不管 | **`setup_logging` 内显式 silence list** | 不 silence 合完 root level=INFO 后 stderr 噪音 10×，dev 一小时内 rollback |
| uvicorn access log | 沿用 uvicorn 自带 / 接管 | **接管为 JSON handler** | access log 跟业务 log 风格割裂，"请求→service→worker" join 不出来 |

---

## 错误 envelope 渐进迁移

| 阶段 | release | 后端 | 前端 |
|---|---|---|---|
| Phase 1 | 0.12.0 | dual-write：`{"detail": <legacy>, "error": {"code", "message", "trace_id"}}` | 优先读 `body.error.trace_id` 显 toast；fallback 读 `body.detail` |
| Phase 2 | 0.15.0 | 所有 `raise HTTPException` 加 deprecation log；front-end 完成全量迁移到 `body.error.*` | 删 `client.ts` 内 `body.detail` 解析路径，只剩单一 `ApiError` |
| Phase 3 | 0.16.0 | handler 删 `detail` key；测试 5 文件 11 处迁完 | — |

**关键**：Phase 1 多写 ~30 行（handler 同时填两个 key），给前端 toast trace_id body 可见性提前 3 release。

> **进度（2026-06-19 更新）**：Phase 1 随 0.12.0 发布；Phase 2/3 曾滑期（目标下调到 0.15.0 / 0.16.0）。**Phase 2 已实现**（分支 `feat/error-envelope-i18n`，待并入 0.15.0）：HTTPException backstop handler 让 `body.error` 全覆盖 + ~330 处 raise 迁 `DomainError` 带语义 code + 前端按 code 查 `errors.*` i18n（删 4 个中文子串匹配 helper）。实现与本表略有出入（用 backstop handler 替代 deprecation log，且全量迁移而非渐进）。Phase 3（删 legacy `detail`）现可安全做。详见 [`docs/todo/error-envelope-detail-key-removal.md`](../todo/error-envelope-detail-key-removal.md)。

---

## 实施计划（3 PR）

### PR-1: 后端基础设施（~28h / ~6 commit）

| Commit | 内容 |
|---|---|
| C1 | 加 `concurrent-log-handler` 依赖；新建空骨架 `infrastructure/logging.py` 仅暴露 `make_studio_log_handler` |
| C2 | `infrastructure/logging.py` 完整 `setup_logging` + JsonLineFormatter + HumanConsoleFormatter + 第三方库 silence list + uvicorn handler 接管 + utf8 reconfigure |
| C3 | `api/lifespan.py` + `cli.py` + `workers/_base.py` 三处调用 `setup_logging` |
| C4 | `infrastructure/logging.py` 加 ContextVar + Filter；`api/middleware/trace.py` 新建 TraceIdMiddleware；`api/app.py` 装 middleware |
| C5 | `supervisor/core.py:_popen` 注入 ANIMA_TRACE_ID env；新 migration `_vN_request_trace.py` 加 `tasks.request_trace_id` 列；dispatcher 读取并 var.set；API endpoint 入 task 时写 request_trace_id |
| C6 | workers/*.py print → logger；supervisor `_finish_slot` tail jobs/<id>.log 回写 error_msg（解 B-1.6）；`_on_task_log` malformed event 改 logger.error + emit SSE warning_event（解 B-4.4）；`services/inference/daemon.py:_read_stderr_loop` 加 thread watchdog + restart（解 B-4.5）；`event_bus._safe_put` QueueFull 加 logger.warning（解 B-1.5） |

**前置安全网**（C1 前一个 commit）：
- `tests/test_error_response_snapshot.py` — 锁现有 20 个 4xx/5xx endpoint 形状
- `tests/test_log_baseline.py` — caplog 验典型 logger.warning 路径
- `tests/test_worker_event_protocol.py` — 锁 `__EVENT__:` IPC 前缀过滤行为
- `tests/test_cli_stdout_baseline.py` — capsys 锁现有 CLI 输出关键字
- `tests/test_logging_import_inertia.py` — 验 `import studio.infrastructure.logging` 后 sys.excepthook 仍 default + root handlers 数仍 0

**回归网**：全套 `pytest -q tests/` + `npm test --silent` 每 commit 必跑。

**Rollback**：每 commit 单独 revert；revert C5 后 worker 端 `os.environ.get("ANIMA_TRACE_ID")` 永远安全；revert migration 用反向 migration（仅删列）。

### PR-2: 错误体系（~26h / ~5 commit）

| Commit | 内容 |
|---|---|
| C1 | `studio/domain/errors.py` 新建 DomainError 基类 + 5 子类（NotFound/Validation/Conflict/Auth/Forbidden），不依赖 fastapi |
| C2 | `studio/api/exception_handlers.py` 新建 + 注册到 `app.py`：DomainError handler / RequestValidationError handler / Exception fallback handler；dual-write envelope（`{"detail":<legacy>, "error":{"code","message","trace_id"}}`） |
| C3 | 5 个 service 错误类（PresetError/ProjectError/VersionError/CurationError/TrainIOError）加 DomainError base；4 套 _err_code helper 退化为 thin wrapper |
| C4 | router try/except 批量迁移 batch 1+2（preset / projects/* domain）— 删 `try: ... except XxxError: raise HTTPException(...)` 三明治，让 raise 直接进 handler |
| C5 | router try/except batch 3+4（curation / training / queue）；删 4 套 _err_code helper 文件；删 `api/errors.py` 内残留 |

**关键约束**：
- DomainError `message` 字段规约为**英文** + 前端用 `code` 查 i18n 表（避开 ADR-0008 §跨问题 D 中文匹配陷阱）
- HTTPException 老路径保 `{"detail": <string>}` 形态不变（trace_id 仅 header 兜底），handler 注册顺序保证 starlette 默认行为不破现有 175 处 raise

**Rollback**：每 commit 单独 revert；hotfix 路径（合后 1 周前端某 toast 解析挂）= 改 `_error_body` 强制 fallback 到 `{"detail": <string>}`（30 分钟 hotfix）。

### PR-3: 前端 + CLI 收尾（~18h / ~5 commit）

| Commit | 内容 |
|---|---|
| C1 | `studio/api/routers/client_errors.py` 新建 `POST /api/client-errors`（per-IP 10/min 限流，独立 `client_errors.jsonl`） |
| C2 | `web/src/lib/errors/{setup.ts, report.ts}` 新建；`main.tsx` 装 `window.error` + `unhandledrejection` 监听 |
| C3 | `ErrorBoundary.tsx::componentDidCatch` 改造上报；`api/client.ts` 三处 fetch 包装统一吃 X-Trace-Id；toast 显示 `trace ab12cd34` 后缀 |
| C4 | `cli.py` 加 `_say(msg, level="info")` wrapper；48 处 print 走 `_say`；启动消息保留 print 路径，诊断信息走 logger（`--verbose` 翻开） |
| C5 | capsys 7 文件改"in 模糊匹配"；ADR-0009 状态改 Accepted；ADR-0008 §"还的债" 第 4 项标 "已并入 ADR-0009" |

---

## 不在范围（推迟到 0.13.x+）

| 项 | 推迟理由 |
|---|---|
| `infrastructure/logging.py` 单文件 → 子包 | 当前 ~200 LOC，超 400 触发；纯文件搬动 PR 2h |
| Worker 单写 → 双写（worker 直接 emit `studio.log`） | `concurrent-log-handler` Windows + OneDrive + Defender 稳定性需先验证；当前单写 + supervisor 回写已经解 1.6 痛点；运维 grep `jobs/*.log studio.log` 也能聚合 |
| envelope Phase 2/3（删 `detail` legacy key） | 跨 release deprecation 周期；滑期后 Phase 2 → 0.15.0、Phase 3 → 0.16.0（见 `docs/todo/error-envelope-detail-key-removal.md`） |
| IndexedDB 离线上报 retry | 前端上报失败 silently swallow 是合理默认；离线场景属于增强 |
| `jobs/<id>.log` 真 rotation | 单 job 几 MB 量级可控；改 `_finish_slot` 删超 7 天 .log 即可（GC 而非 rotation），独立 follow-up |
| CLI cmd_run subprocess 不接管 stdout（B-5.2） | 跟 nssm/systemd wrapper 行为相关，本批不动 |
| CLI cmd_test 无 junit-xml（B-5.3） | DX feature 不是日志体系 |
| 前端 64 处 `console.error/warn/log` 散落统一（B-3.4） | 跟 i18n epic 一起做 |
| 中英文错误信息混杂（B 跨 D 完整解决） | i18n 单独 epic；本 ADR 只规约 DomainError message 英文 + code 查表，message 兜底 |

---

## 替代方案（已否决）

### A. 12 PR 极细切分
最初 C agent 推 8 → 12 PR 方案。否决理由：这不是 25k 行重构（参 ADR-0008 12 PR），是新增 cross-cutting 体系；细切 PR 数 review/合并节奏不匹配，反复合 → 反复 rebase。3 PR 按主题切，每 PR 一个清晰心智模型，stack 依次合并。

### B. envelope hard cutover 到 `{"error": {...}}`
A agent 首轮主张。否决理由：前端 client.ts 3 处 + Presets.tsx 1 处 + 5 测试文件 11 处依赖 `body.detail`；hard cutover 立刻炸 toast；dual-write 给 3 release 渐进窗口。

### C. Worker 双写到 `studio.log` + `jobs/<id>.log`
A agent 首轮主张。否决理由：Windows 跨进程文件锁复杂（`concurrent-log-handler` 在 OneDrive 同步路径 + Defender 场景未验证）；运维 trace_id grep `jobs/*.log studio.log` 也能聚合；0.13.x 可独立 PR 演进。

### D. CLI 全部 print → logger
A agent 首轮主张。否决理由：CLI 是 5s 短命周期落盘价值低；capsys 7 测试文件重写 caplog 工时 ≈ 4h；用户终端看 `2026-05-28 14:32 [INFO] studio.cli:` 一坨比 `[studio] ...` 难看。

### E. loguru / structlog
A agent 首轮考虑。否决理由：运行期硬依赖污染 requirements；subprocess 启动开销；caplog 不正交（需 `loguru-caplog` shim）；Windows 上 rotation 跟 stdlib 同样有锁问题；JSON schema 简单 50 行自写。

### F. `infrastructure/logging/` 子包 6 模块
A agent 首轮主张。否决理由：当前 ~200 LOC 单文件完全 hold；6 文件并行 review 难度反而高；后续超 400 LOC 再纯文件搬动拆，零风险。

---

## 后果

### 好处

- **跨进程 trace_id 全链路贯穿** — 用户报问题给 toast trace 后缀，`jq 'select(.trace_id=="...")' studio.log` 一行还原 webui → supervisor → worker → 错误抛出时间线
- **5 surface 统一日志格式** — JSON line 10 固定字段，jq / grep 跨 surface 检索
- **API 错误响应统一** — DomainError 体系替 4 套字符串匹配 helper，删 4 文件 helper
- **前端崩溃可观测** — ErrorBoundary + window.onerror + unhandledrejection 三路上报到后端 `client_errors.jsonl`
- **数据库错误根因可见** — supervisor tail jobs log 回写 `db.tasks.error_msg`，UI Task 列表从 "exit code 1" 升级到 traceback 摘要
- **关闭 ADR-0008 §"还的债" 第 4 项**

### 新增约束

- 写新 router 时**禁止**手写 `try/except XxxError: raise HTTPException`；改 raise DomainError 子类让 handler 兜
- 写新 service 错误时**必须**继承 DomainError 子类（或其再子类）
- 写新 worker 时**必须**调 `setup_logging(process="worker:<kind>/<job_id>")`，不调直接 `print`
- 写新 fetch 调用**禁止**直接 `fetch(url)`；走 `apiClient.req()` 包装拿 X-Trace-Id 自动注入
- DomainError `message` 字段**规约英文** + 前端用 `code` 查 i18n（防 ADR-0008 跨问题 D 中文匹配陷阱借 DomainError 复活）
- 第三方库新加依赖时，如果带 logger，**评估是否加进 silence list**（参考 `infrastructure/logging.py` 内现有列表）
- 永久保留 `studio.server` shim 不动（参 ADR-0008 §永久保留 4 shim）

### 还的债（0.13.x+）

| 项 | 触发条件 |
|---|---|
| `infrastructure/logging.py` 单文件 → 子包 | 文件超 400 LOC（预期 OpenTelemetry / Sentry 适配器引入时） |
| Worker 单写 → 双写（emit `studio.log`） | 跨进程查 trace 痛感明显时（运维反馈）+ `concurrent-log-handler` Windows 稳定性验证通过 |
| envelope Phase 2（deprecation log） | 前端完成 `body.error.*` 全量迁移后 |
| envelope Phase 3（删 `detail` legacy key） | Phase 2 一个 release 周期后 |
| `jobs/<id>.log` GC 超 7 天 | 独立 PR ~10 行（不算 rotation） |
| 中英文错误信息完整规约 | i18n epic 启动时 |
| 前端 64 处 `console.*` 统一抽象 | i18n epic 启动时 |
| CLI cmd_run / cmd_test 接管 stdout | wrapper（nssm / systemd）改造时 |

---

## 实施 lessons

代码 review 时按需引用。完整背景在 git log + PR description。

1. **TraceIdMiddleware 必须 pure ASGI 不能 BaseHTTPMiddleware** — starlette
   0.36+ 后者用 anyio.Stream wrapping，ContextVar 跨 thread 跳跃在 Python
   3.10+ 有边缘 case。pure ASGI 直接拿 receive/send，ContextVar 在请求生命
   周期内稳定。（PR-1 C5）

2. **Fallback `Exception` handler 跑在 ServerErrorMiddleware 外层 contextvar 已 reset** —
   FastAPI `app.add_exception_handler(Exception, ...)` 注册到 ServerError­Middleware
   (在 TraceIdMiddleware 外层)；DomainError 等具名异常注册到 ExceptionMiddleware
   (内层 contextvar 仍可用)。fallback 路径必须从 `request.scope["state"]["trace_id"]`
   读，靠 contextvar 拿不到。统一用 `_trace_id_from(req)` helper 优先 scope state
   兜底 contextvar。（PR-2 C2）

3. **ContextFilter 装 handler 而非 root logger** — stdlib `Logger.filter` 只在
   `Logger.handle` 顶层调一次，子 logger propagate 到 root 时**不**调
   `root.filter`，只调 `root.handlers[*].emit`。Filter 装 logger 上 → 子 logger
   record 完全绕过。装到每个 handler 才确保所有 record 经过 handler 时都
   有 ContextVar 注入。（PR-1 C5）

4. **`ANIMA_LOGGING_NO_BOOTSTRAP` env 守卫 + `ANIMA_LOG_DIR` env 隔离** — 业务入口
   (api/lifespan / cli.main / workers/_base.worker_main) 在被测试触发时会装真
   file handler 写 repo `studio_data/logs/`，污染 caplog 跟磁盘。conftest
   session fixture 设两个 env：业务 setup_logging 顶部 early return；
   ANIMA_LOG_DIR 兜底指向 tmp_path_factory。测 setup_logging 自身的 fixture
   用 `monkeypatch.delenv` 解除。（PR-1 C4）

5. **db.tasks.request_trace_id 列让 dispatcher 拿请求时刻 trace_id** — supervisor
   后台 dispatcher tick spawn worker 时**不**在 HTTP request ctx 内，contextvar
   trace_id 是 None。如果只兜底 `bg-{uuid}` 标后台触发，用户截图 toast 的
   trace_id (请求时刻) 跟 worker log 的 trace_id (spawn 时刻) 对不上，链路断。
   修：API endpoint 入 task 时 `get_trace_id()` 写 `tasks.request_trace_id` 列；
   dispatcher 拉起时读该列注入 worker env。（PR-1 C6）

6. **批量替换 print → _say 正则会误伤 _say 自身** — `replace_all "print(f\"[studio] "
   → "_say(f\""` 把 `_say` 内部实现里的 `print(f"[studio] {msg}", file=...)`
   也替换了，导致无限递归 + 错误 kwargs。修：`_say` 内部用 `print("[studio] " + str(msg), ...)`
   字符串拼接而非 f-string 避免 pattern 匹配。（PR-3 C4）

7. **route_snapshot.json 在新 endpoint 后必须 regenerate** — `test_route_snapshot`
   按 method+path+name+type 锁死全部 route 集合。新增 `/api/client-errors`
   时删 snapshot 文件 + 重跑即可生成新 baseline；commit 进 git 锁新基线。
   （PR-3 C1）

8. **前端上报 silent swallow 是强约束** — ErrorBoundary 已在 catch state，
   上报本身失败再 throw → 二次崩溃 → ErrorBoundary 自身死循环。`reportClientError`
   内全部 `try/catch` 吞（连 `console.warn` 都 try）；fetch keepalive:true 让 tab
   关闭瞬间也尽量送出。（PR-3 C2/C3）

---

## 参考

- 三 agent 两轮 review 文档：`tmp/log_unify_agent_{a,b,c}_{architect,audit,migration,round2}.md`
- 关联 ADR：
  - [#0008 studio/ 4 层重构（0.11.0）](0008-studio-restructure-0.11.0.md) — §"还的债" 第 4 项已并入本 ADR
- 关键文件 / 索引（落地后）：
  - `studio/infrastructure/logging.py` — 日志体系入口
  - `studio/domain/errors.py` — 错误体系入口
  - `studio/api/exception_handlers.py` — 3 个 handler 注册点
  - `studio/api/middleware/trace.py` — trace_id 入口
