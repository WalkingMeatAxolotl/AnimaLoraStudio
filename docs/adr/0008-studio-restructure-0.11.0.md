# 0008 — studio/ 4 层重构（0.11.0）

**状态**：Accepted
**日期**：2026-05-28
**决策者**：@WalkingMeatAxolotl

## 背景

0.10.x 末期 `studio/` 顶层是平铺的 25k 行单包，关键单文件大头：

- `server.py` 4657 行 / 130 个 `@app.<verb>` 装饰器 + lifespan + middleware + helper 杂糅
- `supervisor.py` 1431 行（`_spawn_task` 单 method 164 行）
- `schema.py` 976 行（`TrainingConfig` 单类 643 行）
- `services/` **平铺 36 个文件**，无分组
- `cli.py` 841 行 / `secrets.py` 763 行 / `model_downloader.py` 1068 行 / `reg_builder.py` 1107 行

具体痛点：

- **`server.py` 是 monolith**：130 routes + 49 inline `BaseModel` + 4 套 `_err_code` helper 一文件装；新加 route 一定要碰它
- **`services/` 平铺**：tagging / booru / inference / reg / preprocess 5+ 个不同主题混在同级；找代码要靠 grep
- **`import-time 副作用`**：`server.py` 顶层直接 `ensure_dirs()` + `db.init_db()`，`from studio.server import app` 立即写盘 + sqlite init —— 单元测试 / 工具脚本 import 时被迫接受这个 side effect
- **`schema.py` / `supervisor.py` / `model_downloader.py` 等单文件过大**：编辑器导航 / git blame / code review 都吃力

需要一次系统性重构把 `studio/` 切成清晰的层和子包。

## 候选方案

两轮 review 后讨论了 3 个目标态：

### A — 4 层完整架构（采纳）

```
api/             HTTP 表面（FastAPI app + router + schemas + deps）
  ↓
services/        业务服务（按主题分子包：tagging / booru / reg / inference / ...）
  ↓
domain/          pydantic 模型（TrainingConfig + LoRA / XY / Generate / RegAi + migrations）
  ↓
infrastructure/  路径常量 / 数据库 / event bus / secrets / 日志 / argparse 桥接
```

`supervisor/`（任务调度守护线程）和 `workers/`（4 个子进程入口）跨层使用，不归 4 层之一。

### B — 折中方案（仅拆 server.py + services/ 分组）

只动 `server.py` 抽 router + `services/` 平铺改 11 子包，不引入 `domain/` / `infrastructure/`。否决理由：

- `domain/` 缺位 → `schema.py` 跟业务代码同层，无法把"数据形状"和"运算"分开
- `infrastructure/` 缺位 → `paths.py` / `db.py` / `secrets.py` 等基础设施跟业务代码混居
- 25k 行项目用 4 层架构不算过度（参考 Kubernetes / Django 项目都用类似分层）

### C — 6 层 DDD 风格（domain / application / infrastructure / interfaces / adapters / shared）

否决：项目规模不到 DDD 阈值，引入会让贡献者必须学 hexagonal architecture 词汇才能改代码。

## 决策

采纳方案 A（4 层完整架构）。

11 PR 实施（PR-1..8 + 中途细切 PR-3.8/3.9 + 大型 PR-6.5 + PR-9 收尾）：

| PR | 状态 | server.py | 关键产出 |
|---|---|---:|---|
| PR-1 #141 | merged | 4657 | 安全网（route snapshot + import smoke + invariants）|
| PR-2 #142 | merged | 4657 | `schema.py` → `studio/domain/` 8 文件 |
| PR-3 #143 | merged | 4657 | `services/` 11 子包（36 文件分组）|
| PR-3.8 #144 | merged | 4657 | `model_downloader.py` 1068 → 4 文件 |
| PR-3.9 #145 | merged | 4657 | `reg/builder.py` 1108 → `analysis.py` 抽 |
| PR-4 #147 | merged | 4657 | `supervisor.py` 1431 → 6 文件 + `_spawn_task` 拆 helper |
| PR-5 #148 | merged | 4068 | `api/` 骨架 + lifespan 迁移 + 4 router (20 routes) |
| PR-6 #149 | merged | 2382 | 6 中小域 router (64 routes) + `queue/` 3 文件子包 |
| PR-6.5 #150 | merged | 253 | `projects/` 域 71 routes (5 子文件) |
| PR-7 #151 | merged | — | 顶层 8 文件搬 `infrastructure/` |
| PR-8 #152 | merged | 51 | workers `_base` + server.py 终极 shim |
| PR-9 #153 | merged | — | 45/49 shim 删除 + `__init__.py` docstring 加强 |

**核心成就**：

- **server.py: 4657 → 51 行 (−98.9%)**
- **`@app` 装饰器: 130 → 0** in `server.py`
- 全部 160 routes 分布在 27 个 `api/routers/` 文件
- 4 层架构完整落地

## 理由

**为什么 4 层 > 折中 B**：

- 0.11.0 之后会持续加 features（pause/resume PR-3、多 GPU 槽位、quota）。这些 feature 都涉及 service + persistence + api 三层。4 层架构让"加 feature = 加层叠加"，折中 B 会让 `services/` 同时混进业务和基础设施代码
- `domain/` 独立的真正价值是**前端 schema 生成路径稳定**：`/api/schema` 端点 → `studio.domain.TrainingConfig.model_json_schema()`，前端表单字段顺序依赖 pydantic field declaration order，把 `TrainingConfig` 放在独立的 `domain/` 让前端契约文档化
- `infrastructure/` 把"数据库 / 路径 / 配置"跟业务隔离，方便未来替换（SQLite → Postgres、本地 disk → S3）

**为什么不一步到位用 DDD（方案 C）**：

- DDD 词汇（aggregate root / value object / repository）对单人项目反而是负担
- 4 层已经覆盖 80% 的好处（关注点分离 + 依赖方向单向），剩下 20% 的代价不值得

**为什么 PR-3..7 各自单独 PR**：

- 单 PR 改 25k 行无法 review；每 PR 限定单一主题 + full sweep gate
- 实施期发现的偏差（每 PR 都有 1-3 个"偏离 planning"决策）写在 PR description，避免事后翻账

## 实施期 lessons（8 条）

逐 PR 累积。每条都跟 codify 进 ADR 而非散落在 commit message。

### 1. `sys.modules` 别名 shim 模式（PR-3 起）

包同名覆盖模块时（如 `services/presets.py` 被 `services/presets/` 子包覆盖），用 `_sys.modules[__name__] = _real` 让旧路径透明转发：

```python
import sys as _sys
from .infrastructure import paths as _real
_sys.modules[__name__] = _real
```

效果：旧 `from studio.paths import X` + tests 的 module-attr 访问 + monkeypatch 全部 work。这是物理搬迁时的 backward-compat 桥梁，不是"空文件"。

### 2. 包内跨子模块调用必须走 `module.func()`（PR-3.8）

`from .sub import func; func()` 把 `func` bind 成本模块名。tests `monkeypatch.setattr("pkg.sub.func", X)` 改子模块 attribute 对调用方无效。

解法：`from . import sub as _sub; _sub.func()` 让 lookup 走 module attribute lookup。

**规则**：任何被 test patch 的函数，调用方必须用 `module.func()` 而非 `func()`。

### 3. fixture monkeypatch path 必须跟 router/模块搬迁（PR-5/6 反复出现）

handler 搬走后，老 `monkeypatch.setattr(server, X)` 看不到新位置。fixture 必须同步加新位置的 patch（或更新到直接访问新模块）。

PR-6 发现 3 个子类：
1. 老 `setattr(server, X)` patch 不到新 router
2. 老 `server.X` 直接 import 现在不存在
3. 老 `del sys.modules["studio.server"]` + 重 import 在共享 app 实例下污染（见 lesson 6）

### 4. shim 一个 package（PR-7）

搬走的 package（如 `studio/migrations/` → `studio/infrastructure/migrations/`）在老路径放 `.py` 文件做 sys.modules 别名：

```python
# studio/migrations.py
import sys as _sys
from .infrastructure import migrations as _real
_sys.modules[__name__] = _real
```

这是 PR-3 同名包覆盖模式的**镜像**：那里是 package shim 一个 module；这里是 module shim 一个 package。子模块访问 `from studio.migrations._v2_projects import X` 经 sys.modules 转发也透明 work。

### 5. file-relative `Path(__file__).parent` 搬深一层要补 `.parent`（PR-7）

`studio/paths.py` 搬到 `studio/infrastructure/paths.py` 后：

```python
# 老
REPO_ROOT = Path(__file__).resolve().parent.parent

# 新（多一层）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
```

**规则**：搬动模块时 grep `Path(__file__).resolve().parent` 检查，搬到深一层就补 `.parent`。同样适用 `PRESETS_DIR` 这种 data resource path。

### 6. `del sys.modules + reimport` 在共享 app 实例下污染（PR-6）

PR-5 起 `server.py` 的 `@app` 装饰器跟 `api/app.py` 的 `app` 是同一 `FastAPI` 实例。tests `del sys.modules["studio.server"]` 删 server 但 `api.app` 仍 cached，重 import `server.py` 会让 `@app` 装饰器对**同一 app 重复注册** —— 每跑一次 routes 数翻倍（实测 160 → 320 → 480 ...）。

解法：放弃重 import 模式，改返轻量壳暴露 tests 直接用的 2 个名字。

### 7. 状态耦合高的类**不**拆（PR-2 TrainingConfig / PR-4 Supervisor）

行数痛点 < 状态耦合时，保单类不拆 mixin。具体：

- `TrainingConfig` 643 行 19 字段组：拆 mixin 会改变 pydantic field declaration order，影响 `model_dump()` / YAML 序列化 / `/api/schema` JSON 输出 —— 这是**行为变更**
- `Supervisor` 1162 行 37 method：全部 read/write 共享 self 字段（`_slots / _daemon_*`），拆 mixin 反而让未来扩展跨文件改 self 状态

**规则**：行数大 ≠ 必须拆。共享状态多的对象拆出去会让未来 PR 更难写。

### 8. shim 删除 ≠ shim 是 bug（PR-9）

shim 是物理搬迁的 backward-compat 桥梁，**承担了真实工作**（10 行 `sys.modules` 别名 + import system 路径转发）。删除 = 完成迁移闭环（callers 100% 跟随到 canonical path），不是 cleanup hygiene。

策略：
- **不用 `__getattr__` lazy 回退** — 隐藏 ImportError，fail late 难 debug
- **每 caller 显式改 canonical path** — fail loud + fail early
- **每 commit full sweep test gate** — baseline 不能降
- **大批量用 Python 脚本** — sed-style 替换 + 逐文件审查 + 测试

49 个 shim 删 45 个，保 4 个永久（`server.py` / `db.py` / `paths.py` / `secrets.py`，test fixture monkeypatch 大量使用，删除成本巨大）。

## 后果

### 好处

- **`server.py` 51 行**：编辑代码不再被它 dominate；新加 router 走 `api/routers/<name>.py` 模式
- **`services/` 11 子包**：找代码先按主题进 subdir（tagging / booru / inference / ...）
- **`domain/` 8 文件**：前端 schema 契约对应 1 个文件 = 1 个 BaseModel，定位精确
- **`infrastructure/` 隔离**：路径 / 数据库 / 配置不混在业务代码里
- **测试基线 1500+ → 1551**：重构期没掉测试覆盖率
- **`@app` 装饰器集中 27 router 文件**：route 总览有索引（`api/routers/__init__.py` docstring）

### 新增约束

- **不要在 `studio/` 顶层加新文件**：除非是 shim；新代码进 `api/` / `services/` / `domain/` / `infrastructure/` 之一
- **不要反向依赖**：`api/` 可以 import `services/`，反之不行；`services/` 可以 import `domain/` 和 `infrastructure/`，反之不行
- **router 文件命名 `api/routers/<domain>.py`**：跟 `api/schemas/<domain>.py` 配对
- **跨 router 共用 helper 进 `api/deps.py`**；某个域内的 helper 进 `api/routers/<domain>/_shared.py`（如 projects/_shared.py）
- **新增的 router 必须 `app.include_router(...)` 在 `api/app.py`**，且 path 顺序约束要遵守（如 queue/io 必须在 queue/lifecycle 之前 include，避免 `/api/queue/export` 被 `/api/queue/{task_id}` 截胡）

### 还的债（0.11.1+ follow-up）

| 项 | 原计划 PR | 推迟理由 |
|---|---|---|
| `cli.py` 841 → 7 文件拆 | PR-8 | launcher 入口单文件可读；独立 PR 隔离风险 |
| `secrets.py` 763 → models/store/migrations 3 文件 | PR-7 | Pydantic v2 跨文件循环风险；3-way 收益主要是视觉隔离 |
| `db.py` 188 → connection/tasks/settings 3 文件 | PR-7 | 同上 |
| 统一 exception handler 替代 4 套 `err_code` helper | PR-6 | 4 套 helper → `BusinessError` + 单 `exception_handler` 是行为变更 |
| `secrets.py` 170 行 legacy migration 加 deprecation log | PR-7 | 跟 3-way 拆一起做 |

### 永久保留的 4 个 shim

- `studio/server.py` (51 行) — FastAPI `app` + `main` + SPA mount + `HTTPException` + 6 path 常量 + `db` re-import
- `studio/db.py` / `studio/paths.py` / `studio/secrets.py` (各 10 行) — test fixture 大量 `monkeypatch.setattr(server.db, ...)` 使用，删除 = 改 30+ 测试文件 + 风险，ROI 不值

这 4 个 shim 是**架构的一部分**，不是删除候选。

## 参考

- 11 个 PR：#141 #142 #143 #144 #145 #147 #148 #149 #150 #151 #152 #153
- 实施日志 + 三方 agent 报告：`tmp/0.11.0_planning.md` + `tmp/0.11.0_agent_{a,b,c}_*.md`（gitignored，本地保留）
- 关联 ADR：[#0003 anima_train.py 模块化重构](0003-anima-train-refactor.md)（同性质的 runtime 端拆分，2026-05 已落）
