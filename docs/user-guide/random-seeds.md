# 随机种子与可复现性

Studio 的 seed 数字输入统一使用：

- `0`：随机。每次**新建对应任务**时生成实际的正整数 seed。
- 非零值：固定。再次使用相同参数时可以复现相同噪声。

“随机”不等于每个步骤都换 seed。Studio 会按用途确定随机化的边界，既让新任务有新结果，
又保留训练和评估所需的公平对比。

## 训练任务

训练配置中有三个独立的 seed：

| 字段 | 用途 | 填 `0` 后的行为 |
|---|---|---|
| `seed` | 训练数据顺序、初始化及训练随机过程 | 新训练任务随机一次，全程固定 |
| `sample_seed` | 训练期间的 baseline、step、epoch 采样图 | 新训练任务随机一次；同一 prompt 跨轮次固定 |
| `eval_validation_split_seed` | 自动划分 held-out 验证集 | 新训练任务随机一次；该任务内固定 |

实际值会写入任务自己的 `snapshot/config.yaml`。源预设中的 `0` 不会被覆盖，因此下次新建训练
任务会得到新的值。

- **Retry、暂停后恢复**：继续使用原任务快照中的实际 seed，不重新随机。
- **重新点击开始训练，创建新 task**：所有仍为 `0` 的字段重新随机。
- **显式填写非零值**：新建、重试和恢复都使用该值。

多 prompt 采样会为每个 prompt 使用稳定的 index offset。因此 prompt A 的 baseline 与后续
采样始终共享噪声，prompt B 也共享自己的噪声；不同 prompt 不必共用同一张噪声图。

验证集目录若已经达到目标比例，新的划分任务可能不再移动图片。此时即使新任务生成了新的
划分 seed，现有 held-out 集也不会被无故重排。

## 测试页普通出图

- 单张图 seed 为 `0`：daemon 为该图生成实际 seed。
- 普通批量 seed 为 `0`：每张图都独立随机，不会变成固定的 `0, 1, 2...`。
- 普通批量 seed 为非零值 `N`：各图使用 `N, N+1, N+2...`，可重复生成整批结果。

生成完成后，历史记录和 PNG metadata 保存的是**实际 seed**，不是请求中的 `0`。从历史恢复
参数时可以直接复现该图。

## XY 矩阵

XY 的目标是只比较轴参数差异，所以一次矩阵内所有 cell 共享同一个 seed：

- seed 为 `0`：矩阵启动时随机一次，全部 cell 使用该实际值；
- seed 为非零值：全部 cell 使用该显式值。

不要将普通 batch 的“每张独立随机”套用到 XY，否则不同 cell 的变化会同时混入噪声差异。

## 训练后评估

同一个 Eval Session 的 baseline 和所有 checkpoint 候选共享生成 seed，验证图片再按稳定 index
offset 派生。这样指标差异主要来自 LoRA/checkpoint，而不是随机噪声。

训练完成后自动创建的 Eval Session 会优先沿用该训练 task 已冻结的 `sample_seed`。手动创建
且没有父训练 task 的 Eval Session 会在 session 创建时解析 `0`，并把实际值冻结在 EvalPlan。

## AI 正则图

AI 正则图的 seed 为 `0` 时，会在新任务创建时生成一个实际 base seed；该任务中的图片按
`base, base+1, base+2...` 派生。任务配置会保存实际 base seed，所以重试同一任务不会改变
尚待生成图片的 seed。显式非零值仍直接作为 base seed。

## 排查与复现

需要复现或报告问题时，请优先提供：

1. task ID；
2. 任务目录下的 `snapshot/config.yaml`（训练）；
3. 生成历史或 PNG metadata 中的实际 seed（测试页）；
4. Eval Session 的 `plan.json`（评估）。

这些持久化值才是运行时使用的权威 seed。
