# 0011 — LoRA 评估指标体系与可审查 PR 拆分

**状态**：Proposed
**日期**：2026-05-27
**决策者**：@WalkingMeatAxolotl

> **维护约定**：本 ADR 定义 LoRA checkpoint validation metrics 的目标、边界和可审查实现步骤。
> 如果后续指标实测与本文假设冲突，在末尾追加 Addendum，不改写初版决策。
>
> ⚠️ **2026-06-25 重大修订（见末尾 Addendum: 验证集 held-out 重设计）**：初版的
> reference/prompt **manifest** 协议已被**取代**。参考图不再取自训练集（那会奖励记忆），
> 改为训练前从数据集划出的 **held-out 验证集 `validation/`**；manifest 文件整层删除，
> 每个 `run.json` 自包含；出图参数取训练配置 `sample_*`；eval 输出统一 task-scoped。
> 下文「Eval manifest」「reference 取自 train 图」等描述以该 Addendum 为准。

## 背景

当前 Studio 对 LoRA 质量的判断主要来自训练 loss、采样图和人工观察：

- loss 能发现训练是否爆炸，但不能稳定代表角色拟合、prompt 服从或过拟合程度；
- 采样图能直观看质量，但不同 checkpoint / seed / prompt 之间难以比较；
- 训练后 LoRA picker 和 Test 页能评估结果，但没有固定验证集、固定 seed、固定指标；
- 过拟合常表现为主体相似度继续上升，同时 prompt 服从、多样性或泛化下降；
- “学得像”与“复制训练图”需要分开看，否则高相似度会被误读成好结果。

调研 `sorryhyun/anima_lora` 时发现其使用 paired CMMD² 作为验证信号，同时社区个性化生成论文常用
DINO-I / CLIP-I / CLIP-T / diversity / copy-risk 等组合指标。它们适合作为 Studio 的可选评估层，
但不应进入训练 step 主路径。

## 目标

1. 给每个 version / checkpoint 建立可重复的评估协议：同一批 held-out 图、caption、prompt、seed。
2. 在 checkpoint 后异步生成 eval samples，保留 grid、单图和 metadata，方便人工对比。
3. 逐步接入自动指标，分别回答：
   - 是否还听 prompt；
   - 是否学到主体 / 风格；
   - 是否出现模式塌缩；
   - 是否疑似复制训练图；
   - 生成集合是否贴近 held-out 分布。
4. 输出可解释的趋势和诊断，不用单一总分掩盖指标冲突。
5. 保持训练主路径零侵入：默认不加载额外评估模型，不抢训练显存，不降低训练 step 速度。

## 不在范围

- 不把指标同步塞进训练 loop。
- 不默认启用高成本评估。
- 不在第一阶段做自动早停。
- 不把 CLIP / DINO / SSCD / CMMD 等模型常驻 GPU。
- 不承诺任何单个指标等价于人类审美。
- 不把 Hydra / ReFT / IP-Adapter 等 adapter 变体纳入本 ADR；它们属于训练算法扩展。

## 决策

### 评估运行方式

LoRA eval 是 checkpoint 后的独立任务，不是训练 step 的一部分：

1. 训练任务保存 checkpoint 或用户手动触发后，提交 eval job；
2. eval job 读取固定 manifest，调用既有推理能力生成样图；
3. 指标模型在 eval job 内按需加载，用完卸载；
4. real image embeddings 可缓存，checkpoint 之间只重算 generated embeddings；
5. eval job 可取消，可低频运行，可在 CPU / low-vram 模式下降级；
6. UI 明确标注评估会额外占用时间 / 显存。

默认策略：

- 训练时不自动抢 GPU 跑指标；
- 初期仅手动触发或在训练结束后触发；
- 后续可加“每 N epoch / N checkpoint 评估一次”选项，默认关闭；
- 若训练仍在占用 GPU，eval job 应排队等待或走 CPU fallback，而不是与训练争抢显存。

### 数据协议

新增 eval manifest 作为后续所有指标的共同输入。建议路径：

```
studio_data/projects/{id}-{slug}/versions/{label}/eval/
├── manifest.json
├── samples/{run_id}/
│   ├── run.json
│   ├── metrics.json
│   └── images/
└── cache/
    └── embeddings/
```

manifest 记录：

- version / project 标识；
- held-out image path + caption path；
- eval prompts；
- seeds；
- generation params（尺寸、steps、cfg / guidance、sampler、LoRA scale 等）；
- manifest schema version；
- 创建时间和来源（自动抽样 / 用户选择）。

`metrics.json` 跟随 sample run 存放，而不是另建并行 checkpoint 目录，原因是后续 CLIP / DINO / SSCD / CMMD runner 都需要引用同一批 generated images、manifest digest、checkpoint 身份和 generation metadata。run 级目录能让样图、指标结果与缓存索引共享一个稳定的 `run_id`。

### 指标分层

指标按问题分层，不合成单一魔法分：

| 层 | 指标候选 | 回答的问题 | 备注 |
|---|---|---|---|
| Prompt following | CLIP-T | 出图还听不听 prompt | 第一批接入，成本低 |
| Image similarity | CLIP-I | 粗略图像相似 | 与 CLIP-T 同模型族，便于共用缓存 |
| Subject / style fidelity | DINO-I | 主体 / 风格是否学到 | 个性化生成论文常用 |
| Diversity | LPIPS / DreamSim | 同 prompt 多 seed 是否塌缩 | 检测“训僵” |
| Copy-risk | SSCD nearest neighbor | 是否疑似复制训练图 | 必须与 subject similarity 分开 |
| Distribution fit | paired CMMD² | generated 集合是否贴近 held-out 集合 | 综合趋势指标，后置接入 |
| Quality preference | HPSv2 / ImageReward / PickScore | 人类偏好式质量 | 可选增强，不作为拟合核心 |

诊断逻辑只能在单项指标稳定后再引入。例如：

- DINO-I / CLIP-I 上升且 CLIP-T 不降：拟合改善；
- DINO-I 高但 SSCD copy-risk 高：可能复制训练图；
- DINO-I 高但 diversity 降、CLIP-T 降：疑似过拟合或触发词绑死；
- paired CMMD² 到低点后回升：可能进入过拟合区间，需要人工看样图确认。

### 可审查实现步骤

吸取 PR #18 的经验，禁止把评估体系做成一个无法审查的大块。PR #138 固定为 foundation PR；后续指标、UI 和 checkpoint ranking 均拆成 stacked PR。每个 PR 只回答一个问题，尽量只引入一个新概念、依赖或指标。

| Step | 范围 | 明确不做 | 阻塞关系 |
|---|---|---|---|
| 0 + 1 | 本 ADR + 文档索引 + Eval manifest：固定 eval reference 与 sample preset | 不生成图、不算指标、不做 UI | 无 |
| 2 | Eval sample runner：按 manifest 对 checkpoint 出图，保存单图与 `run.json` metadata | 不接 CLIP / DINO / CMMD | 0 + 1 |
| 3 | Metric result schema：定义 `metrics.json`、embedding cache 目录、API 返回格式、空状态 | 不实现具体指标 | 2 |
| 4（PR2） | CLIP-T / CLIP-I：新增 `eval_clip` job，读取 sample run 与 manifest reference，写入 `metrics.json` 的 `clip_t` / `clip_i` | 不做 diversity / copy-risk / paired CMMD²，不做 UI，不做 checkpoint ranking | 3 |
| 5（PR2） | DINO-I：新增 `eval_dino` job，读取 generated/reference image pairs，写入 `metrics.json` 的 `dino_i` | 不做 diversity / copy-risk / paired CMMD²、不做诊断 UI | 3, 4 |
| 6（PR2） | Eval metric model settings：为 CLIP / DINO 指标保存默认模型名或本地路径，API 请求省略 `model_name` 时读取这些默认值 | 不新增下载 UI、不改指标算法、不接 diversity / copy-risk / paired CMMD²、不做诊断 UI | 4, 5 |
| 7（PR3） | Training-integrated eval：训练保存 LoRA checkpoint 后自动排 eval sample，sample 完成后自动排 CLIP / DINO | 不做 polished monitor UI、不自动给训练参数建议、不新增 diversity / copy-risk / paired CMMD²、不做 checkpoint ranking | 4, 5, 6 |
| 8（当前 PR4） | Formal UI embedding：Settings 明确三指标模型 / 自动评估配置，Monitor 按 checkpoint 展示 CLIP-T / CLIP-I / DINO-I 值和趋势 | 不新增指标算法、不做综合诊断、不做 checkpoint ranking | 7 |
| 9 | Diversity（LPIPS 或 DreamSim） | 不判断训练图复制 | 8 |
| 10 | SSCD copy-risk：nearest-neighbor 相似度、高风险比例、对照图 | 不做综合评分 | 8 |
| 11 | paired CMMD² | 不做 checkpoint 自动推荐 | 8, 9 |
| 12 | Diagnosis UI：汇总指标并提示拟合不足 / 过拟合 / 复制风险 | 不新增指标模型 | 8, 9, 10, 11 |
| 13 | Checkpoint ranking：基于已有指标给可追溯推荐理由 | 不改变训练默认流程 | 12 |

Step 2 的 sample runner 应先落地为一个可持久化、可重跑、可被后续指标读取的 eval sample run：

- 输入：version eval manifest、`output/` 下的一个 LoRA checkpoint，以及 version 训练配置里的模型 / runtime 默认值；
- 调度：通过现有 `project_jobs` 增加 `eval_samples` job kind，按 GPU-bound job 处理，避免在训练 step 内同步抢资源；
- 输出：`versions/{label}/eval/samples/{run_id}/run.json` 与 `images/*.png`；
- metadata：记录 manifest digest / snapshot、checkpoint 身份、prompt、seed、生成参数、每张图状态和 summary counts；
- 明确不做：不计算 CLIP / DINO / SSCD / CMMD，不做 checkpoint ranking，不输出质量推荐。

这样 step 2 能证明 manifest 已经成为真实跨任务契约，同时仍把指标依赖和诊断 UI 留给后续 step。

Step 4 的 CLIP runner 应先落地为一个最小可复用指标 job：

- 输入：已完成的 eval sample run、run 内冻结的 manifest snapshot、generated images、对应 prompt 与 reference image；
- 调度：通过 `project_jobs` 新增 `eval_clip` job kind，并按 GPU-bound job 处理，避免默认与训练并行抢显存；
- 输出：写回同一个 `metrics.json`，只更新 `clip_t` / `clip_i` 与对应 `metric_states`，不得覆盖 DINO / diversity / SSCD / CMMD 等其他指标结果；
- 缓存：在 `eval/cache/embeddings/clip/` 写 generated / text / reference embeddings 与 metadata，便于后续同 checkpoint 或同 reference 复用；
- 降级：缺模型、缺依赖、缺 reference 或缺 prompt 时，把对应 metric 标为 `failed` 或 `unavailable`，不影响 sample run 与其他 metric；
- 明确不做：不改 UI、不做 DINO、不做 copy-risk、不自动推荐 checkpoint。

Step 5 的 DINO runner 沿用同一套 metric job contract，但只回答 subject/style fidelity：

- 输入：已完成的 eval sample run、run 内冻结的 manifest snapshot、generated images 与 reference image；
- 调度：通过 `project_jobs` 新增 `eval_dino` job kind，并按 GPU-bound job 处理；
- 输出：写回同一个 `metrics.json`，只更新 `dino_i` 与对应 `metric_states`，不得覆盖 CLIP / diversity / SSCD / CMMD 等其他指标结果；
- 缓存：在 `eval/cache/embeddings/dino/` 写 generated / reference embeddings 与 metadata；
- 降级：缺模型、缺依赖或缺 paired reference 时，把 `dino_i` 标为 `failed` 或 `unavailable`；
- 明确不做：不改 CLIP runner、不做 diversity、不做 copy-risk、不自动推荐 checkpoint。

Step 6 的 eval metric model settings 先把服务器实测过的 ModelScope / 本地路径流程产品化，但不引入新的指标依赖：

- 输入：Settings 中保存的 CLIP / DINO 默认模型名或本地目录；
- 调度：不新建 job，只影响已有 `eval_clip` / `eval_dino` API 在请求体省略 `model_name` 时的默认值解析；
- 输出：保存到 secrets 配置，metric job params 与 `metric_states.*.model_name` 记录实际使用的模型值；
- 兼容：手动 API 请求仍可传 `model_name` 覆盖默认值，便于单次实验；
- 明确不做：不新增模型下载 UI、不改变 CLIP / DINO 指标算法、不做 diversity / copy-risk / paired CMMD²、不做 checkpoint ranking。

Step 7 的 training-integrated eval POC 用已有 CLIP / DINO 指标回答“同一训练过程里的不同 checkpoint 能否被区分”：

- 触发：训练保存 LoRA checkpoint（step / epoch / final）时 emit `eval_checkpoint_saved` 结构化事件；
- 调度：Studio supervisor 在 POC 开关开启时，把该 checkpoint 排入 `eval_samples` job；GPU-bound job 仍按现有队列策略排队，默认不与训练并行抢显存；
- 串联：自动排队的 sample run 成功后，继续排 `eval_clip` 与 `eval_dino`，模型路径读取 Step 6 的 Settings 默认值；
- 输出：每个 checkpoint 仍产生普通 `eval/samples/{run_id}/run.json` 与同目录 `metrics.json`，不引入单独的实验结果格式；
- 明确不做：不根据指标自动修改训练参数，不新增正式 monitor UI，不新增新指标，不做 ranking。

Step 8 的 UI embedding 把前三个指标从“脚本可查”推进到“训练监控可见”：

- Settings：保留 CLIP / DINO 默认模型路径和自动评估开关，文案明确本地 ModelScope 路径与 Monitor 输出位置；
- State context：`/api/state` 为绑定 project/version 的训练任务附带轻量上下文，前端可据此读取该 version 的 eval metrics；
- Monitor：新增 LoRA eval 面板，按 checkpoint 展示 CLIP-T / CLIP-I / DINO-I 最新值、pending/running/failed 状态和简单趋势；
- 刷新：训练连接仍在或 metric job 尚未结束时定时刷新 eval results，训练结束后保留手动刷新；
- 明确不做：不新增 Diversity / SSCD / CMMD，不解释“最佳 checkpoint”，不根据指标自动修改训练参数。

每个后续 review step / commit 说明固定包含：

```md
## Hypothesis
本 PR 验证什么假设。

## Scope
本 PR 做什么。

## Out of Scope
明确不做什么。

## Validation
命令、样例项目、mock、截图或指标 JSON。

## Follow-ups
后续 step 接什么。
```

如果某个 step 超过约 500 行核心逻辑变化，说明里需要解释为什么不能继续拆。

## 候选方案

### A：训练 loop 内同步跑指标

否决。会直接影响训练速度和显存，且指标模型失败会污染训练任务。

### B：训练结束后手动跑一次完整评估

部分采纳。第一阶段以手动 / 结束后触发为默认，后续再做低频自动触发。

### C：checkpoint 后异步 eval job（采纳）

优点：

- 不影响训练 step 速度；
- 可排队、可取消、可 CPU fallback；
- 指标失败不影响训练产物；
- 与现有 Queue / Test / daemon 心智接近；
- 便于按 PR 分层接入指标。

缺点：

- 需要新增 eval manifest 和结果缓存协议；
- 指标趋势有延迟；
- 如果用户选择训练中共享 GPU，会额外占用显存。

### D：只做 CMMD² 一个综合指标

否决。CMMD² 是集合分布距离，不能单独区分 prompt 服从下降、主体没学到、多样性塌缩或训练图复制。

### E：直接做综合评分 / 自动早停

否决。单项指标尚未校准前，综合评分会给用户虚假的确定性。

## 后果

**好处**：

- 后续实现有清晰边界，不会重演大 diff 难审；
- 每个指标都能独立验证是否对 Anima LoRA 有用；
- 训练主路径保持稳定；
- 用户可以同时看样图、趋势和风险提示；
- copy-risk 与 subject fidelity 分离，避免把背图误判为好拟合。

**约束 / 新债**：

- 需要管理额外模型依赖和缓存体积；
- 指标阈值必须在二次元 / Anima 数据上重新校准；
- 评估结果只能辅助决策，不能替代人工看图；
- 若未来自动触发 eval，需要与 Queue GPU 调度策略对齐。

## 验收策略

当前 PR #138 到 step 3 为止的验收：

- 新增 ADR / docs 索引；
- 新增 version-scoped `eval/manifest.json` 服务和 API；
- 新增 version-scoped eval sample run 服务、API 和 `eval_samples` job kind；
- 新增 metric result contract、`metrics.json` 位置、embedding cache layout 和 API 空状态；
- 不新增指标模型依赖；
- 不计算 CLIP / DINO / SSCD / CMMD，不改 UI；
- GET manifest 不隐式写文件，POST/PUT 才落盘；
- manifest 明确区分 eval reference 与 sample preset，并说明默认 reference 来源；
- 实施计划明确到每个后续 step / PR 的范围和不在范围。

后续 step 的共同验收：

- 每个 PR 独立可运行 / 可展示；
- 新指标必须有 mock 或小样例测试；
- 指标模型失败时不影响训练任务；
- GPU / CPU / no-model 三种状态都要有明确 UI 或 API 行为；
- 指标输出必须保留原始分项，不只给综合结论。

## 参考

- AnimaLoraStudio PR #18：维护者反馈“2000 多行更改 1 个 commit”难审，后续大功能按审查主题拆 PR。
- ADR 0003：训练栈 plugin 化，adapter / optimizer / scheduler / loss / timestep_sampler 分层扩展。
- ADR 0006：Queue pause/resume，训练任务与后续 GPU 任务调度的产品边界。
- ADR 0007：Project / Version / Task 生命周期，为 version 级 eval 结果挂载提供语义边界。
- ADR 0008：studio/ 4 层重构，当前 PR 的 API 接入应走 `studio/api/routers/`，业务逻辑放 `studio/services/`。
- DreamBooth / DreamBench 系列：DINO-I / CLIP-I / CLIP-T 常用于个性化生成评估。
- Rethinking FID：CMMD 使用 CLIP embedding + MMD 替代 FID 的高斯假设。
- SSCD：图像 copy detection，可作为训练图复制风险的 nearest-neighbor 特征模型。

## Addendum: Step 7 队列调度修正（2026-06-11）

真实长训练测试发现，自动评估可能一次性排出较多 `eval_samples` job。若队列纯按
job id FIFO 调度，已完成 sample 后新排出的 `eval_clip` / `eval_dino` 可能被后续大量
`eval_samples` 堵在后面，用户需要等待更多采样完成后才能看到已经可计算的指标。

对应修复是让 eval metric job 优先于后续 eval sample job 调度，同一 run 已有
pending/running metric job 时复用已有 job，并把 `auto_metrics` / `auto_source`
写入 run metadata，方便后续 UI 和排障直接从 `run.json` 判断自动评估来源。

## Addendum: 验证集 held-out 重设计（2026-06-25）

初版的 reference 取自**训练集**（默认 manifest 用 train 图 + 其 caption 当 prompt + 同图当参考）。
真实使用发现这等于**奖励记忆/复制**：LoRA 越照抄训练数据，CLIP-I / DINO-I 越高，与
ADR 背景里「学得像 ≠ 复制训练图」的目标相悖。本次重构（仍在 PR #138 内）改掉数据模型。

### 决策

1. **真 held-out 验证集**。训练前从数据集物理移动一部分图（含 caption）到与 `train/`
   同级的 `versions/{label}/validation/`，这些图**不参与训练**，作为 eval 的 reference +
   prompt 来源。划分在 studio 侧训练启动前的钩子执行，`anima_train` 不改。
2. **按比例补足、不移回**。`target = round(ratio × |train+validation|)`；`|validation|`
   不足才从 `train/` 随机（固定 seed）补差额，达到目标即不再划分，且永不自动移回 —— 用户
   手动放进 `validation/` 的图也计入目标。
3. **训练配置开关（per-version）**，取代初版的全局 Settings 自动评估总开关
   `auto_eval_on_checkpoint`（已删）：`eval_validation_enabled` /
   `eval_validation_split_ratio`（0–1，0=不划分）/ `eval_validation_split_seed`。
   **是否**评估由该 per-version 开关驱动（同时门控训练前的 held-out 划分）；**何时**评估
   由保留在全局 Settings 的 `auto_eval_trigger`（`after_training` 训练全部结束后批量 /
   `checkpoint` 训练中每存一个 LoRA inline 评）决定。全局 Settings 还保留指标模型名。
   **评估范围 = 整个 validation set**（大小由 `eval_validation_split_ratio` 决定），不再有
   `auto_eval_max_items` 截断：该旋钮是 #138 为「从大 train 池采样子集」而设，validation set
   本身就是受控评估契约，再截断在概念上矛盾（默认 1 更让验证集形同虚设），已删。
4. **删除 manifest 整层**。`eval_manifest` service + 3 个 manifest 路由 + 默认 manifest
   生成逻辑全部删除。`validation/` 文件夹本身就是「评估契约」；create_run 直接扫该目录建
   items，每个 item 自带 `reference_image`（相对 version dir）。`run.json` 自包含本次快照，
   不再有 `manifest_snapshot` / `manifest_digest`。
5. **出图参数取训练配置 `sample_*`**（width/height/infer_steps/cfg_scale/sampler/
   scheduler/negative + sample seed），修掉初版「出图参数与训练脱节」的问题。出图 seed 与
   分隔 seed 是两件事。
6. **无验证集图 → 只跑 CLIP-T**（不需参考图），用训练配置的 sample prompts 出图，**不回退到
   train 图**（回退会重新引入记忆奖励）。
7. **eval 输出统一 task-scoped**（`tasks/{id}/eval/`），删除 version-scoped 输出分支与不可达
   的裸 `POST /eval/samples`。改由 `POST /eval/run`（带 task_id + 显式 checkpoint 集）触发
   手动评估，结果落在该 task 的 eval 页（指标页）。
8. **评估时机沿用 #138 的 inline 设计**。`auto_eval_trigger=checkpoint` 时，训练进程在每次
   保存 LoRA 后**复用已加载的训练模型**串行出图 + 算指标（`ctx.model.eval()` → 评 →
   `train()`），评完再继续训练，因此不另起进程、不重载模型、不与训练抢显存；只有 inline
   拿不到 `LORA_TASK_ID`（罕见）时才回退到 supervisor 的 `eval_checkpoint_saved` 事件 +
   `queue_checkpoint_eval` 异步 job。两条路径都受 per-version `eval_validation_enabled` 门控。

### 仍未实现 / 后续

- ~~验证集文件夹在训练配置页目前以字段描述呈现路径；专门的「手动加图」UI 是后续增强。~~
  已实现，见 Addendum: 验证集手动 curation（2026-06-26）。
- Diversity / SSCD copy-risk / paired CMMD² 仍未实现（与初版边界一致）。

## Addendum: 验证集手动 curation（2026-06-26）

上一个 Addendum 把验证集做成 train 前按 `eval_validation_split_ratio` 自动划分；本次补上**手动维护**入口，因为「就用这几张图当 held-out」的诉求无法用比例表达。

1. **Curation 页（训练集筛选）加「训练集 / 验证集」切换**。默认训练集（现行为）；切到验证集后右栏变成 `validation/`，从 download 选图加入 / 移除，与训练集对称。
2. **held-out 不重叠在 curation 层强制**：左栏候选 = `download − train − validation`，两个 bucket 共用同一候选池；copy 到 validation 时再次跳过已在 train 的名字。这同时修了一个潜在泄漏 —— 训练后 auto-split 把图移进 `validation/`，旧 `curation_view`（只减 train）会让这些图重新冒回 train 左栏候选、可被重新加进 train。
3. **验证集扁平、无文件夹概念**。`iter_images` 要求图在子目录下，故手动加入的图统一落固定 `validation/1_data/`（复用 `DEFAULT_TRAIN_FOLDER`）；UI 不暴露 repeat 文件夹管理，右栏把 `validation/` 下所有子文件夹（手动 + auto-split）拍平成一个列表展示。caption sidecar 跟随复制（eval 拿它当生成 prompt）。
4. **与自动划分叠加**：手动选的图计入 `ensure_validation_split` 的目标数（split 逻辑不动），比例设 0 即纯手动。手动加图**不**自动改 `eval_validation_enabled`，面板提示去训练配置开启。
5. validation 无 manifest（held-out 集只读、靠目录位置区分身份），故新增的 `list/copy/remove_validation` service 比 train 路径更简单：纯物理复制 / 删除，不写 manifest、不做 origin 去重。新增 `GET/POST .../curation/validation[/copy|/remove]` 三个 endpoint，`version_thumb` 增 `validation` bucket（按 `folder + name` 寻址，同 train）。
