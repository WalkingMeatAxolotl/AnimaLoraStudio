# 错误 envelope `detail` key 移除（ADR 0009 Phase 2 / 3 收尾）

**创建于** 2026-06-19
**触发** 0.14.0 发版核对时发现 [ADR 0009](../adr/0009-logging-error-system.md) §错误 envelope 渐进迁移的 Phase 2/3 早已滑期：ADR 当时把 Phase 2 排到 0.13.0、Phase 3 排到 0.14.0，但实际只有 Phase 1 跟 0.12.0 发出去了，后两阶段一直没人做。已把目标版本下调（Phase 2 → 0.15.0、Phase 3 → 0.16.0）并立此条防再忘。
**当前状态** 🟡 Phase 1 已发布（0.12.0 起 dual-write）；Phase 2 待做（**下个版本 0.15.0**）；Phase 3 被 Phase 2 阻塞。

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

## Phase 2 待办（下个版本 0.15.0）

- [ ] 前端 `client.ts`：把错误文案主来源从 `body.detail` 改读 `body.error.message`，`body.detail` 退为 fallback；保留 `ApiError.traceId` 现有逻辑。核对所有 callsite（`e.detail` / `e.detail.error` 等结构化用法）。
- [ ] 后端：给老 `raise HTTPException` 路径加一次性 deprecation log（标识仍走 legacy detail-only 形态的 endpoint），方便盘点残留。
- [ ] 把还在 `raise HTTPException(...)` 的 router 逐步迁到 `DomainError`（走 dual-write handler），减少 Phase 3 时的 detail-only 残留面。
- [ ] 测试：更新依赖 `body.detail` 的断言（ADR 记为 5 文件 11 处）改读 `body.error.*`。

## Phase 3 待办（0.16.0，Phase 2 落地后）

- [ ] `exception_handlers.py` 的 `_error_envelope` 删 `detail` key，只留 `error`。
- [ ] RequestValidationError handler 仍保 starlette 默认 `{"detail": [...]}`（pydantic body 校验，前端有专门处理）——这条不在本迁移范围。
- [ ] 收尾测试 + ADR 0009 状态更新。

## 同步点

落地各阶段时记得回改 [`docs/adr/0009-logging-error-system.md`](../adr/0009-logging-error-system.md) 的迁移表与 `studio/api/exception_handlers.py` 顶部注释里的阶段/版本。
