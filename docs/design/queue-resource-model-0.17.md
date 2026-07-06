# 队列资源档位模型 —— 显存并发准入（0.17 内落地）

> 状态：设计定稿讨论中（2026-07-04）。前置阅读 `queue-overhaul-0.17.md`（P-A..P-I 已全部完成）。
> 本文回答：**「两条队列」的真实目的是显存并发准入控制——按这个目的，队列应该怎么设计。**
> 结论在本版（0.17.0 发布前）全部落地，不拖 0.18；后续优化仅限不改变用户感知的内部增强。

---

## 1. 第一性目标

队列分两条（tasks / project_jobs）的**核心目的从来不是任务分类，而是回答：训练运行时，什么可以同时上卡**。

- generate / reg_ai 进 tasks（与训练同一串行语义）：**大显存**，训练时启动会 OOM / WDDM 换页，把训练打崩 → 必须互斥。
- 超分 / 打标进 project_jobs：相对训练**显存占用很小** → 低风险，可以（受控地）并行。

即：分界线应该按**显存兼容性**画，而当前实现是按「归属表」近似的。错位的直接后果见 §2。

## 2. 现状核对：三个准入漏洞

（逐行核对 `studio/supervisor/core.py`，2026-07-04）

| # | 漏洞 | 位置 | 后果 |
|---|------|------|------|
| L1 | **训练运行时后端不挡 generate**。`_dispatch_generate` 不查 `_train_busy`，`POST /api/generate` 入队也不查；互斥纯靠前端按钮 disabled + 轮询（有竞态窗口，直接打 API 全绕过）。`allow_gpu_during_train` 对该路径不生效 | `core.py:436-456`、`generate.py:167` | 训练中 daemon 加载整套底模抢显存。P-I 允许连续提交 + P-B 定时训练后，「排队 generate + 定时训练到点启动」可真实触发 |
| L2 | **训练不躲数据车道（仲裁单向）**。数据车道 GPU job 会给训练让路，但 pending 训练看到 eval_samples（底模级，独立第二份拷贝）正在跑时照样 spawn | `core.py:417-434` 无 data-busy 检查 | 两个底模级进程同时上卡 |
| L3 | **`allow_gpu_during_train` 粒度过粗**。`GPU_BOUND_JOB_KINDS` 把 eval_samples（≈generate 显存）和 tag/CLIP（几百 MB）混在同一开关下 | `cmd_builder.py:18-27` | 用户为「训练时能打标」开开关 → 同时放行 eval_samples 与训练并行 → 必崩 |

**显存量级事实**（按任务实际加载的模型）：

| 量级 | 任务 |
|------|------|
| 底模级（数 GB） | train、reg_ai、generate、**eval_samples**（与 generate 同款底模栈，独立子进程） |
| 小模型（数十–数百 MB） | preprocess（spandrel 超分）、tag（ONNX）、eval_clip / eval_dino / eval_tag / eval_ccip |
| 不上卡 | download（纯 IO） |

## 3. 资源档位模型（目标设计）

**一张台账 + 每个工作项一个静态资源档位 + 集中准入函数。**「两条队列」在旧模型里同时承担存储隔离、并发准入、UI 分区三个职责；资源模型把三者解耦。

### 3.1 档位映射

| 档位 | 成员 | 准入规则 |
|------|------|----------|
| `gpu-exclusive` | train / reg_ai / generate / **eval_samples** | 全系统同时最多 1 个。daemon 常驻模型 = 持有一张可吊销的 exclusive 租约（要派其他 exclusive 项时先 `request_unload`，running 不强中断） |
| `gpu-light` | preprocess / tag / reg_build / eval_clip / eval_dino / eval_tag / eval_ccip | 无 exclusive 运行时恒放行；有 exclusive 运行时按开关放行（`allow_gpu_during_train` 语义收窄为**仅此档**） |
| `io` | download | 恒放行（仅受 queue hold 约束） |

已拍板（2026-07-04）：
- **D-R1 preprocess（超分）归 light**。
- **D-R2 eval_samples 归 exclusive**——eval「归属违和」就此消解：它本来就不是一个任务，出图部分是底模级、指标部分是小模型，按档位切开即可。

### 3.2 档位内排序：平级 FIFO（已拍板 D-R3）

**exclusive 档内各类型平级**，无「训练优先 / 评估优先」的类型间策略：按入队顺序（`priority DESC, created_at ASC`，用户可 reorder）依次执行，**一个结束才启动下一个，running 永不被中断**。

推论（回答 generate 防饿死问题，已拍板）：
- 连点 generate 合法：先点的 generate 排在后入队的 train 前面（自己点的，先跑理所当然）；train 入队后再点的 generate 排在 train 后面。纯队列顺序，无需预留锁。
- 「train 开始后锁定」是 FIFO 的自然结果：train running 期间任何 exclusive 新项（含 generate）只能排队。
- **行为变化注意**：今天「下一个训练插队在排队的 eval 出图前」（两表隐式训练优先）在新模型下消失——eval_samples 先入队就先跑。这是有意的语义简化。
- daemon 模型复用不受影响：连续 generate 之间无 train 插入时仍走热缓存；FIFO 中夹了 train 则自然卸载/重载（物理必然）。

light 档同理平级 FIFO；现 `dispatch_order` 的「指标插队在出图前」硬编码随 eval_samples 升档自动消解（指标与出图不再同档竞争），删除。

### 3.3 为什么不做动态显存预算（探测 free VRAM 按余量准入）

被本项目自己的 WDDM 血泪史否决（PR #281 / `runtime/training/models.py:42-67`）：Windows WDDM 下显存超限**不报干净 OOM**，而是在总显存 ~50% 处触发换页，单 op 从 <1s 退化到上百秒且无可捕获信号——「装得下」≠「能跑」，且训练显存随阶段波动，准入时点的余量骗人。主机制必须是 proactive 静态档位；free-VRAM 探测仅留作 light 档「自动放行」的远期增强（探测失真也只放进小模型，有档位边界兜底）。

### 3.4 串行槽数量

槽数 = 各档位物理瓶颈容量：exclusive=1（单卡）、light=1（保守，WDDM engine 争抢会拖慢共存训练）、io=1。三档各 1 恰好等于今天三车道的最大并行度——**并行度不变，边界画对**。多卡时代 exclusive=GPU 数，是配置不是架构。

## 4. 用户感知锚点（本版一次到位，此后不再变）

> 原则：不允许「这版让用户习惯一种形态、下版又改」。以下用户可见承诺在 0.17.0 发布时定稿：

1. **队列页 = 两个视图**：「GPU 任务」（= exclusive 档）/「数据任务」（= light + io 档），header 切换钮、过滤、分页、详情页交互均维持现有形态（P-G 已定稿）。
2. **eval 出图（评估出图）出现在「GPU 任务」视图**，eval 指标留在「数据任务」视图——本版随档位模型一步到位，用户第一次见到的就是最终归属。
3. **执行顺序 = 所见队列顺序**（平级 FIFO + 手动 reorder），无隐藏的类型间插队。
4. **设置项**：「训练时允许轻量任务并行」两态开关，**默认开启**（语义 = 仅 light 档），不再暗示可放行重任务。
5. 训练运行时提交 generate：照常入队排队（不再默认禁用按钮——后端已有正确准入，前端 activeBlockingTask 硬禁用改为提示性文案），到点自动执行。

## 5. 落地排期（全部在 0.17.0 发布前）

| PR | 内容 | 性质 |
|----|------|------|
| **R-1 集中准入**（最高优先，独立价值） | `resource_class_of(kind)` 映射 + 单一 `_admit(class)`；三个 dispatch 全改走它；修 L1/L2/L3（generate 补 train-busy 互斥、训练等 exclusive data job、eval_samples 无视开关）；开关语义收窄 + Settings 文案 | 零 schema 改动，纯行为修复 |
| **R-2 台账合并 schema** | `_v17`：tasks 加 `params` JSON、`task_type` 扩容 job kinds、日志路径统一 `tasks/<id>/run.log`；新作业写 tasks；旧 `project_jobs` 转只读历史（不迁移旧行，避免 ID 重映射污染 eval run 引用） | schema + 写路径切换，可回滚 |
| **R-3 调度收编 + FIFO** | DATA 槽从 tasks 按档位取活；删 `dispatch_order` 硬编码；原 job 类白得 reorder / scheduled / 类型过滤 | 后端调度 |
| **R-4 事件/API 统一** | `job_state_changed` → `task_state_changed`（带 kind）；`/api/jobs*` 留兼容 shim 读旧表历史 | 契约 |
| **R-5 前端数据源合流（视图不变）** | 两个视图（GPU 任务 / 数据任务）**像素级维持现状**，只把数据源合一：同读 tasks 表按档位过滤（GPU 视图 = exclusive，数据视图 = light+io）；`QueueJobDetail` 并入 `QueueDetail`（本就同款）；事件/ID 空间归一。用户可见变化仅两处（均为 §4 锚点）：eval_samples 归位 GPU 视图、generate 页训练时提交解禁 | 前端 |
| ~~R-6~~（并入 R-1） | light 放行开关定为**两态 enable/disable，默认 enable**（D-R4 已决，2026-07-04：NVML 自动档砍掉——WDDM 下探测不可靠，两态可预期）。开关仅辖 light 档；配置项随语义收窄重命名 + 迁移（显式 false 保留为 disable，缺省 → enable） | 并入 R-1 |

**后续优化（0.17.0 后，明确不改变用户感知）**：
- light 档并发上限做成配置（>1）；多 GPU exclusive>1；
- 旧 project_jobs 只读历史表的最终淘汰。

## 6. 迁移细节备忘

- tasks 表 `project_id`/`version_id` 自 `_v2` 就有；缺口仅 `params` JSON、`task_type` 值域、job 事件/日志约定。
- 旧 project_jobs 行不迁移（ID 重映射会污染 eval run 引用与 log 路径）；历史区双源只读聚合，一版后淘汰。
- ID 撞号问题随「新作业统一进 tasks」自然终结（增量方向单一 ID 空间）。
- daemon（D2）不动：模型复用是出图体验命根子；资源模型中它只是 exclusive 租约的持有形式。

## 7. 开放问题

全部已决（2026-07-04）：

- ~~**Q-R1** 最小可发布集~~ 发布窗口充足，质量优先，R-1~R-5 整包发布，不做最小集/过渡态。
- ~~**Q-R2** 旧 project_jobs 历史展示~~ **不展示**——数据任务队列视图是 0.17 新增，正式版用户从未见过旧 job 的队列呈现，无怀旧包袱；旧表留档不读，步骤页 latest-job 回放换数据源。
- ~~**Q-R3** `/api/jobs*` 兼容 shim~~ **不需要**——前后端同包发布、无第三方消费者、无历史展示需求，同版本直接切换删除。
- ~~**Q-R4** light 放行模式~~ **两态 enable/disable，默认 enable**；NVML 自动档砍掉（WDDM 下不好判断，两态可预期）。

## 8. 澄清备忘：「前端合流」≠ 合并视图

用户感知的「两个车道」（GPU 任务 / 数据任务双视图）**就是最终形态**（§4-1 锚点）。R-5 的"合流"仅指管道层：今天两个视图是两套数据源（/api/queue vs /api/jobs）、两个 ID 空间（撞号）、两个详情页、两套事件；合流后同读 tasks 表按档位过滤，视觉与交互零变化。底层物理执行位仍是三个（exclusive / light / io），但 io（下载）不值得独立视图，挂在数据视图内——用户感知两车道与底层三执行位并存，互不矛盾。
