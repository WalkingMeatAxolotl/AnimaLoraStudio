# 0011 — LoRA 评估指标体系与可审查 PR 拆分

**状态**：Proposed
**日期**：2026-05-27
**决策者**：@WalkingMeatAxolotl

> **维护约定**：本 ADR 定义 LoRA 训练后评估体系的目标、边界和 PR 拆分。
> 当前 review unit 合并 PR 0 + PR 1，只落地 eval manifest 协议/API；模型依赖、UI、采样 runner 和具体指标实现都在后续 PR 分别落地。
> 如果后续指标实测与本文假设冲突，在末尾追加 Addendum，不改写初版决策。

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

- 不在当前 review unit 引入采样 runner、指标模型或 UI。
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
├── samples/{checkpoint_id}/
│   ├── images/
│   ├── grid.jpg
│   └── generation_metadata.json
├── metrics/{checkpoint_id}/
│   └── metrics.json
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

### 可审查 PR 拆分

吸取 PR #18 的经验，禁止把评估体系做成一个大 PR。每个 PR 只回答一个问题，尽量只引入一个新概念、依赖或指标。

| PR | 范围 | 明确不做 | 阻塞关系 |
|---|---|---|---|
| 0 + 1（当前 PR） | 本 ADR + 文档索引 + Eval manifest：固定 held-out images / captions / prompts / seeds / metadata | 不生成图、不算指标、不做 UI | 无 |
| 2 | Eval sample runner：按 manifest 对 checkpoint 出图，保存 grid / 单图 / metadata | 不接 CLIP / DINO / CMMD | 0 + 1 |
| 3 | Metric result schema：定义 metrics.json、embedding cache 目录、API 返回格式、空状态 UI | 不实现具体指标 | 2 |
| 4 | CLIP-T / CLIP-I | 不做 DINO / diversity / copy-risk | 3 |
| 5 | DINO-I | 不改 CLIP 逻辑、不做诊断 | 3, 4 |
| 6 | Diversity（LPIPS 或 DreamSim） | 不判断训练图复制 | 3 |
| 7 | SSCD copy-risk：nearest-neighbor 相似度、高风险比例、对照图 | 不做综合评分 | 3 |
| 8 | paired CMMD² | 不做 checkpoint 自动推荐 | 3, 4, 5 |
| 9 | Diagnosis UI：汇总指标并提示拟合不足 / 过拟合 / 复制风险 | 不新增指标模型 | 4, 5, 6, 7, 8 |
| 10 | Checkpoint ranking：基于已有指标给可追溯推荐理由 | 不改变训练默认流程 | 9 |

PR2 的 sample runner 应先落地为一个可持久化、可重跑、可被后续指标读取的 eval sample run：

- 输入：version eval manifest、`output/` 下的一个 LoRA checkpoint，以及 version 训练配置里的模型 / runtime 默认值；
- 调度：通过现有 `project_jobs` 增加 `eval_samples` job kind，按 GPU-bound job 处理，避免在训练 step 内同步抢资源；
- 输出：`versions/{label}/eval/samples/{run_id}/run.json` 与 `images/*.png`；
- metadata：记录 manifest digest / snapshot、checkpoint 身份、prompt、seed、生成参数、每张图状态和 summary counts；
- 明确不做：不计算 CLIP / DINO / SSCD / CMMD，不做 checkpoint ranking，不输出质量推荐。

这样 PR2 能证明 manifest 已经成为真实跨任务契约，同时仍把指标依赖和诊断 UI 留给后续 PR。

每个后续 PR body 固定包含：

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
下一 PR 接什么。
```

如果某个 PR 超过约 500 行核心逻辑变化，PR body 需要解释为什么不能继续拆。

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

当前 PR（PR 0 + PR 1）的验收：

- 新增 ADR / docs 索引；
- 新增 version-scoped `eval/manifest.json` 服务和 API；
- 不新增依赖；
- 不生成样图、不计算指标、不改 UI；
- GET manifest 不隐式写文件，POST/PUT 才落盘；
- 实施计划明确到每个后续 PR 的范围和不在范围。

后续 PR 的共同验收：

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
