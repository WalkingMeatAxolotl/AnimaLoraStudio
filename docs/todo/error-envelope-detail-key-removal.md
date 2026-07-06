# 错误 envelope `detail` key 移除（ADR 0009 Phase 2 / 3 收尾）

> ✅ **已完成** — 三阶段随 0.15.0 发布；保留作记录。

**创建于** 2026-06-19
**触发** 0.14.0 发版核对时发现 [ADR 0009](../adr/0009-logging-error-system.md) §错误 envelope 渐进迁移的 Phase 2/3 早已滑期：ADR 当时把 Phase 2 排到 0.13.0、Phase 3 排到 0.14.0，但实际只有 Phase 1 跟 0.12.0 发出去了，后两阶段一直没人做。已把目标版本下调（Phase 2 → 0.15.0、Phase 3 → 0.16.0）并立此条防再忘。
**当前状态** ✅ 全部完成。Phase 1 已发布（0.12.0）；Phase 2 已合 dev（PR #294）；**Phase 3 已实现**（分支 `feat/error-envelope-phase3`，删 legacy `detail` key，错误响应只发 `error`；唯一例外 RequestValidationError 的 422 list）。三阶段随 0.15.0 一起发布后本条可归档。

---

## 背景

API 错误信封从老格式 `{"detail": <str>}` 渐进迁到新结构化 `{"error": {"code", "message", "trace_id", ...}}`。为不一刀切炸前端，ADR 0009 定三步走：

| 阶段 | 目标版本 | 后端 | 前端 |
|---|---|---|---|
| Phase 1 | 0.12.0 ✅ | dual-write 同时填 `detail` + `error` | toast 优先读 `error.trace_id`，fallback `detail` |
| Phase 2 | **0.15.0**（原 0.13.0） | `raise HTTPException` 加 deprecation log；前端全量迁到 `body.error.*` | 删 `client.ts` 里 `body.detail` 解析路径，只剩单一 `ApiError` |
| Phase 3 | **0.16.0**（原 0.14.0） | handler 删 `detail` key；测试迁完 | — |

## 为什么不能直接做 Phase 3

现状（2026-06-19 核对）：

- 后端 `studio/api/exception_handlers.py` 仍 dual-write（`_error_envelope` 同时填 `detail` + `error`）—— Phase 1 状态。
- 前端 `studio/web/src/api/client.ts`（约 1488-1493、1566-1568 行）**仍以 `body.detail` 为主要错误文案来源**，`body.error.*` 只用来取 `trace_id`。
- 没有 HTTPException 的 deprecation log（handler 注释自己写了「HTTPException 不重新注册」）。

所以现在删 `detail` key（Phase 3）会让所有错误 toast 丢文案。**必须先做 Phase 2**（前端迁到 `error.*` + 后端加 deprecation log），Phase 3 才安全。

## Phase 2 已完成（分支 feat/error-envelope-i18n）

实现与原计划略有出入（更优）：用 HTTPException **backstop handler** 代替「deprecation log」，
一步让 `body.error` 全覆盖；并做了**全量**迁移而非渐进。

- [x] 前端 `client.ts`：4 处 fetch/XHR 错误解析收口为 `makeApiError`，主读 `body.error.code`
      查 `errors.*` i18n（带 details 插值），`body.detail` 退 fallback；结构化数据经
      `err.detail`(=error.details) 给 callsite（Presets 冲突 / Settings 运行中任务已迁）。
- [x] 后端：注册 HTTPException backstop handler（`exception_handlers.py`）→ `body.error` 覆盖所有
      错误响应，detail 原样保留。
- [x] ~330 处 raise（HTTPException + service 异常）迁到 `DomainError` 子类带语义 code + details；
      删 4 个中文子串匹配 helper（`_preset/_project/_curation/_duplicate_err_code`）；router 不再转
      HTTPException。新增 `errors.*` locale（120 code，中英，从 CATALOG 生成）。
- [x] 测试：18 文件断言对齐新信封（子类 + code + error.details）；preset 冲突 400→409。
- [x] 修 `preprocess_worker` 两处 `except PreprocessError`（迁移后改抛 InvalidPathError/
      ValidationError，原 catch 漏接会崩 job）→ 放宽 `except DomainError`。

设计目录（code → en/zh 总表 + 每处映射）见 `tmp/error_i18n/CATALOG.md`（gitignored 草稿，
Phase 3 / 后续错误 i18n 可复用，必要时移入 docs/）。

## Phase 3 已完成（分支 feat/error-envelope-phase3，提前到 0.15.0）

- [x] `exception_handlers.py`：`_error_envelope` + Exception fallback + HTTPException backstop 都只发 `error`，删顶层 `detail`。backstop 的 dict/list detail 改放 error.details。
- [x] RequestValidationError handler 保 `{"detail": [...]}`（唯一保留 detail 的路径）。
- [x] 收尾：~8 测试文件断言迁到 error 信封；error_response_baseline 契约测试改锁 error key；ADR / 注释更新。前端无需改（makeApiError 早已主读 error）。

## 同步点

落地各阶段时记得回改 [`docs/adr/0009-logging-error-system.md`](../adr/0009-logging-error-system.md) 的迁移表与 `studio/api/exception_handlers.py` 顶部注释里的阶段/版本。
