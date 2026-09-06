# 0017 — 分离 LLM 预设、用户设置与服务凭证的持久化边界

**状态**：Proposed（决策已批准；实现与验证完成后改为 Accepted）
**日期**：2026-09-06
**决策者**：@WalkingMeatAxolotl

## 背景

Studio 目前把服务凭证、普通用户设置、LLM Tagger 内置预设 override、自定义
LLM 预设和模型列表缓存共同保存在 `studio_data/secrets.json`。所有修改都经过
“读取整个文件 → deep merge → 重写整个文件”的路径。

这个模型已经导致确定性数据丢失：`POST /api/llm-tagger/models/refresh` 只提交当前
预设的单项 `presets` patch，而通用列表 merge 实际会用该单项替换整个列表；随后
validator 只补回内置预设，其他自定义预设永久消失。当前持久化还有以下系统性风险：

1. `Path.write_text()` 会先截断再写入，不是原子写；进程崩溃可留下空文件或半截 JSON。
2. 完整 read-modify-write 没有统一锁；并发请求会发生 lost update。
3. `load()` 对解析、I/O 和 schema 错误统一返回默认值；下一次保存可能用默认值覆盖
   原始损坏文件。
4. API Key 与普通配置共同出现在同一对象，API 依赖 `"***"` 同时表达“已配置”和
   “保持原值”，容易产生掩码回写和跨域泄漏。
5. 单一数组无法明确表达“修改一个 preset”“替换整个列表”和“删除一个 preset”
   三种不同语义。
6. 模型服务返回的 `model_ids` 是可重建缓存，却与用户 prompt 配方共命运。
7. 一个文件损坏会同时影响全部 preset、全部服务凭证和全部用户设置。

Studio 是本地、单用户、单 server 进程应用；训练 preset 已经使用
`studio_data/presets/*.yaml` 的一预设一文件模型，运行台账则使用 SQLite WAL。
用户要求 LLM preset 同样成为可直接备份和导入导出的独立文件，并明确不采用 OS
keyring。

## 候选方案

### A — 维持单一 `secrets.json`，只修 deep merge

- 优点：改动最小，兼容成本低。
- 缺点：只修复当前触发点；非原子写、并发覆盖、损坏静默默认和凭证混存仍存在；
  未来仍会出现列表 patch 语义歧义。
- **否决**。

### B — 所有数据迁入 SQLite

把 LLM preset、设置、凭证和缓存全部建表保存。

- 优点：事务、并发控制、约束和 migration 机制成熟。
- 缺点：preset 与普通设置变得不便于查看、备份和分享；凭证进入 `studio.db` 及其
  backup/WAL 后扩大意外泄露面；SQLite 默认明文，对 API Key 没有额外保密能力；
  文档型低频设置不需要关系查询。
- **否决作为统一方案**。SQLite 继续只承载需要查询、关联和状态事务的运行数据。

### C — LLM preset 使用文件，settings 和 credentials 进入 SQLite

- 优点：preset 具有便携性；settings 更新和 credentials 写入获得 DB 事务。
- 缺点：普通用户设置仍不需要关系模型；凭证随业务 DB 和备份传播；配置恢复必须依赖
  SQLite 工具；数据库没有提供静态加密收益。
- **否决**。

### D — 按职责拆分文件，运行台账继续使用 SQLite（采纳）

- LLM preset：`studio_data/llm_presets/{id}.json`
- 普通设置：`studio_data/settings.json`
- 服务凭证：`studio_data/credentials.json`
- 可重建模型缓存：`studio_data/cache/llm_models/`
- 运行台账：现有 `studio.db`
- builtin 模板：现有 `studio/llm_presets/*.json`

通过统一的严格读取、进程锁、原子替换、备份和 ETag 并发协议解决文件可靠性问题。

## 决策

### 1. 数据所有权

| 数据 | 唯一权威源 | 说明 |
|---|---|---|
| 内置 LLM 模板 | `studio/llm_presets/*.json` | 随代码发布，只读 |
| 用户 LLM preset | `studio_data/llm_presets/{id}.json` | 一 preset 一文件 |
| 普通应用偏好 | `studio_data/settings.json` | 非敏感、低频、强类型 |
| API Key/token | `studio_data/credentials.json` | 独立明文凭证边界 |
| 模型列表缓存 | `studio_data/cache/llm_models/` | 可删除、可重建、不备份 |
| 项目、版本、任务与运行记录 | `studio.db` | 保持现有关系型权威源 |
| 训练 preset | `studio_data/presets/*.yaml` | 保持现有权威源 |

现有 `Secrets` 字段必须按下表完整迁移，不能只处理 LLM：

| 现有数据 | 新位置 |
|---|---|
| `llm_tagger.presets` 非敏感字段 | `llm_presets/{id}.json` |
| `llm_tagger.presets.*.api_key` | `credentials.json`，preset 保存 `credential_ref` |
| `gelbooru.api_key` / `danbooru.api_key` | `credentials.json`，对应 settings 保存 `credential_ref` |
| `huggingface.token` / `modelscope.token` | `credentials.json`，对应 settings 保存 `credential_ref` |
| `wandb.presets.*.api_key` | `credentials.json`，WandB preset 保存 `credential_ref` |
| 用户名、endpoint、模型选择、下载/生成/训练/系统偏好 | `settings.json` |
| `model_sources` 等带身份集合 | 暂存 `settings.json`，只能经专用资源 API 修改 |
| `wandb.presets` 非敏感字段 | 暂存 `settings.json`，只能经 WandB preset CRUD 修改 |

`settings.json` 可以包含少量带稳定 ID 的集合，但通用 settings PATCH **禁止**修改或
整体替换这些集合；它们必须经按 ID 的领域 API 执行 create/update/delete。这样 WandB
preset 与 model source 不会在新文件中重演当前的整列表截断问题。未来需要把某类集合
拆成独立文件时可复用相同资源 API，而无需再次改变前端语义。

`secrets.json` 在迁移完成后不再是权威源。兼容期内可由新权威源生成只供旧版本
rollback 使用的派生快照；新代码禁止读取该派生快照。

### 2. LLM preset 身份与 builtin overlay

- preset ID 创建后不可修改；label 是可变的人类显示名称。
- 自定义 ID 由服务端生成并限制为跨平台安全 ASCII；迁移时尽量保留现有合法 ID。
- 文件名等于 ID，且在 Windows 大小写不敏感语义下必须唯一。
- builtin ID 保持稳定，例如 `style_json`、`general_json`。
- 用户目录中与 builtin 同 ID 的文件是该 builtin 的**完整 override**，不是 sparse patch。
- 未被用户修改的 builtin 始终读取程序模板；修改 builtin 时写入完整 override；reset
  归档并删除 override。
- 一个无效用户 preset 只使该项进入 invalid 状态，不能阻止其他 preset 或 Studio
  启动；无效文件不得被默认内容静默覆盖。

### 3. Preset 与凭证分离

LLM preset 不保存真实 API Key，只保存：

```json
{"credential_ref": "cred_openai_main"}
```

`credentials.json` 以独立 credential ID 保存 secret。credential ID 与 preset ID 不强
绑定，以允许多个 preset 显式共享同一凭证。规则如下：

- duplicate preset 默认复制 `credential_ref`，不复制 secret；
- portable export 清除 `credential_ref`；
- 删除 preset 不自动删除 credential；
- 仍被 preset 或 pending/running task 引用的 credential 默认不允许删除；调用方可显式
  `force` 撤销，届时引用它的未开始任务必须 fail-fast；
- API 永不返回真实 secret，也不再使用 `"***"` 作为可写哨兵；
- secret 通过 write-only endpoint 创建或替换；
- diagnostics、普通 export、preset export 和日志必须排除 credential 内容。

不用 OS keyring 意味着 `credentials.json` 是本机明文文件。项目不实现“解密 key 与
密文同机存放”的伪加密；文件分离降低意外泄漏和故障爆炸半径，但不防御已经获得本机
文件读取权限的主体。POSIX 尽量设置 `0600`，备份执行相同保护和排除规则。

### 4. 模型列表不是 preset 内容

服务端发现的 `model_ids` 属于按连接信息生成的可重建 cache，不再写入 preset。
`models/refresh` 只更新 cache 并返回候选，不能修改 preset、自动切换 model 或产生
任何 preset 列表替换。用户明确选择 model 后才通过 preset PATCH 保存。

### 5. API 使用资源语义

LLM preset 使用独立资源 API：

```text
GET    /api/llm-tagger/presets
POST   /api/llm-tagger/presets
GET    /api/llm-tagger/presets/{id}
PATCH  /api/llm-tagger/presets/{id}
DELETE /api/llm-tagger/presets/{id}
POST   /api/llm-tagger/presets/{id}/duplicate
POST   /api/llm-tagger/presets/{id}/reset
PUT    /api/llm-tagger/presets/default
POST   /api/llm-tagger/presets/import
GET    /api/llm-tagger/presets/{id}/export
POST   /api/llm-tagger/presets/{id}/models/refresh
POST   /api/llm-tagger/presets/{id}/connection/test
```

credentials 使用 write-only secret API；settings 只接受 partial PATCH，不接受浏览器
回传完整快照。通用 settings PATCH 拒绝 `wandb.presets`、`model_sources` 等集合字段，
这些字段只经专用资源 API 修改。旧 `/api/secrets` 在新布局启用后只保留 GET
compatibility projection；旧写端点返回 `410 Gone` 并要求旧页面重新加载，不能继续
调用通用 deep merge。

preset GET 返回内容 hash ETag。PATCH、DELETE 和 reset 必须携带 `If-Match`；过期
浏览器标签页收到 `412 Precondition Failed`，不得静默覆盖最新文件。导入 ID 冲突
返回 `409 Conflict`，由用户明确选择覆盖、另存或取消。

### 6. 统一原子文件协议

新增共享基础设施，所有被本 ADR 管理的 JSON 写入，以及后续现有训练 preset 写入，
遵守同一协议：

1. 获取对应 store 的进程内 `threading.RLock`；
2. 严格读取现有文件并检查 expected ETag；
3. 在内存构建并验证完整新对象；
4. 在目标同目录创建唯一临时文件；
5. 写入、`flush()`、`os.fsync()`；
6. 仅当现有文件验证有效时归档备份；
7. `os.replace(temp, target)`；
8. 在支持的平台尽量 fsync 目录；
9. 清理残留临时文件并返回新 ETag。

锁覆盖完整 read-modify-write，而不是只锁最终 `write()`。Studio 当前保证 server
single-instance 且所有写入由 server 进程负责，因此进程内锁足够；worker 只能读取
原子快照。如果未来出现独立进程写者，必须先升级为跨进程锁，不能假定 RLock 有效。

### 7. 备份与损坏恢复

- 每个 LLM preset 和 settings 默认保留最近 10 个有效版本；
- credentials 默认保留最近 3 个有效版本；
- 删除 custom preset 或 reset builtin override 时原子归档原文件；
- 损坏文件绝不能覆盖最后一份有效 backup；
- 恢复会产生一个新版本，不能删除历史；
- backup 不代表新的权威源，仅供显式或受控自动恢复。

“文件不存在”与“文件存在但损坏”是两种状态：前者可以创建默认值，后者禁止回落默认
并覆盖。损坏时先保存 `.corrupt-<timestamp>`；有有效 backup 可受控恢复并向 UI 暴露
告警，没有 backup 则该 store 进入 degraded/read-only，相关写入失败，其余 Studio
功能继续运行。

### 8. 已入队任务冻结非敏感快照

打标任务入队时解析 preset，并冻结不含 secret 的完整配置快照：preset ID、ETag、
endpoint、model、messages、生成参数及 `credential_ref`。worker 从任务快照读取配置，
执行时再解析当前 credential；不重新读取可变 preset 文件。这样 preset 编辑或删除
不会改变已入队任务，credential 被删除时任务必须 fail-fast，禁止回退到其他 key。

### 9. 文件布局迁移使用完成标记

文件迁移不能借助 SQLite 事务伪装成跨文件原子事务。迁移采用不可变源快照、staging
和 manifest：

1. 严格读取并 hash 首次看到的旧 `secrets.json`；
2. 原子复制为以 source hash 命名的不可变 migration source，并在 manifest 记录其路径、
   hash 和 `source_kind: legacy-user-source`；后续 resume **只能**读取该快照；
3. 在 staging 目录生成 settings、credentials 和全部 LLM preset；
4. 全量验证数量、ID、字段、引用和 schema；
5. 逐个原子提交目标文件并写 `prepared` manifest；
6. 从最终路径再次读取，逐项与 manifest 中的目标 hash 互证；
7. 最后原子写 `complete`。

只有 `complete` 且全部目标 hash 与 manifest 相符时，新布局才成为权威源。`complete`
丢失时根据 prepared manifest 和不可变源快照幂等续跑；如果 manifest 丢失但目标文件
已经存在，必须进入 degraded/recovery 状态，禁止从当前 `secrets.json` 猜测并重迁。
迁移失败不得删除或改写原始 `secrets.json`，也不得用默认值继续。

兼容窗口采用**持续派生 rollback 快照**：迁移完成后，新权威源每次成功 mutation 都
尽力原子重建旧 schema 的 `secrets.json`，并附带可识别的 projection marker；该文件
可能短暂落后，失败时必须暴露健康告警，但不能反向影响 canonical commit。迁移器永远
不把带 projection marker 的文件当作 legacy source。一个兼容版本后停止生成并归档。

旧 `Secrets` 类型作为 facade 从新 stores 合成；新实现不得双读、双写权威数据，或将
兼容 projection 重新提升为权威源。

## 理由

### 为什么 settings 不进入 SQLite

设置是低频、强类型、无查询需求的配置文档。移除 preset 和 credentials 后文件规模与
故障半径显著下降；统一原子协议即可提供所需可靠性。把它放进数据库会降低备份、人工
检查和恢复能力，却没有关系查询收益。未来出现多用户、设置 profile、远端同步或可
查询历史时，再用新 ADR 重新评估。

### 为什么 credentials 不进入 SQLite

SQLite 默认不加密；把 token 写入 DB 只会使其进入 DB copy、WAL、在线 backup 和诊断
路径，不能获得保密性。独立文件让敏感数据边界、GET API、日志 redaction、导出和备份
规则更容易审查。

### 为什么 preset 使用完整 override

Sparse override 无法稳定区分“字段被用户删除”“字段使用旧默认”和“builtin 新版本
增加默认字段”，跨版本 merge 会再次引入隐式语义。完整 override 让用户修改后的行为
稳定可复现；reset 通过删除 override 显式回到最新 builtin。

### 为什么必须有 ETag

进程锁只串行当前 server 内的请求，不能判断请求是否基于旧浏览器快照。ETag 把多标签
页的 stale write 从静默覆盖变成用户可见冲突，是文件 store 与数据库事务之外的独立
并发保障。

## 后果

### 好处

- 当前刷新模型截断自定义 preset 的 bug 从数据模型上消失；
- 单 preset 修改不会重写其他 preset、设置或 secret；
- 崩溃后目标文件始终是完整旧版本或完整新版本；
- 一个 preset 损坏不会拖垮所有设置；
- preset 可直接备份、导入、导出和恢复历史；
- secret 不再经过完整 settings API 或 preset export；
- 已入队任务获得可复现的非敏感配置快照；
- 训练 preset 与 LLM preset 在“一个实体一个文件”的用户心智上对齐。

### 代价与约束

- 需要新 repository、API、迁移器、兼容 facade 和前端切换，不能作为单点 hotfix 完成；
- 跨多个文件的业务操作不是天然原子事务，必须依赖 prepared/complete manifest 和幂等
  resume；
- 不使用 keyring 意味着 credential 仍是明文 at rest；
- 直接在 Studio 运行期间外部编辑文件不受支持，应通过 import/API 或重启后加载；
- backup 中也可能包含 secret，必须与 credential 本体执行相同保护；
- `secrets.json` compatibility shim 和派生快照只能保留有限版本，必须明确移除计划。

### 实施切片

本决策作为一个 PR 的多个可审查 commit 落地：

1. ADR + 原子文件基础设施 + 当前截断回归修复；
2. LLM preset/credential repository + 独立 API；
3. crash-safe 三域迁移、兼容 facade 和任务快照；
4. 前端切换到新 preset/credential/settings API；
5. 将旧写 API 改为只读/`410`、完成文档和全量验证，并把本 ADR 状态改为 Accepted。

第一步不是废弃补丁：迁移自身必须建立在同一个原子文件底座之上。

## 不变量

后续修改必须保留或通过新 ADR 显式推翻：

1. LLM preset 文件不得包含真实 secret。
2. `model_ids` 等服务端发现结果不得成为 preset 权威字段。
3. 模型刷新不得写 preset。
4. 文件存在但损坏时不得回落默认并覆盖。
5. 所有 mutation 必须锁住完整 read-modify-write 并使用同目录原子替换。
6. preset stale write 必须通过 ETag 拒绝。
7. 新布局只有在 migration manifest 为 `complete` 后才成为权威源。
8. 新代码不得读取 compatibility `secrets.json` 作为权威源。
9. task snapshot 不得包含 secret，worker 不得在运行时重新解析可变 preset。
10. 删除仍被 preset 或活动任务引用的 credential 默认必须失败；显式强制撤销后，活动任务必须 fail-fast。

## 参考

- 现有统一配置与凭证实现：`studio/infrastructure/secrets.py`
- 现有 builtin LLM preset loader：`studio/infrastructure/llm_presets.py`
- 现有训练 preset 文件模式：`studio/services/presets/io.py`
- 现有 LLM Tagger 路由：`studio/api/routers/taggers.py`
- 现有模型刷新路由：`studio/api/routers/installs.py`
- 现有 SQLite WAL 与 migration：`studio/infrastructure/db.py`、`studio/infrastructure/migrations/`
