# 0020 — 在任务边界解析随机种子并冻结比较组

**状态**：Accepted
**日期**：2026-09-12
**决策者**：@WalkingMeatAxolotl

## 背景

Studio 的数字输入用 `0` 表示“随机种子”，但旧实现没有统一随机化的时机和作用域：

- 训练 bootstrap 先用固定训练 seed 初始化 Python RNG，再从它生成 `sample_seed`，因此默认配置每个新任务会得到同一个“随机”值；
- 验证集划分直接把 `0` 交给 `random.Random(0)`，跨任务永远选择同一批图片；
- 测试页批量生成把 `0` 展开为 `0, 1, 2...`，只有第一张随机；
- 推理进程解析出的实际 seed 没有回写历史和 PNG metadata，随机结果无法复现；
- Eval 把 `0` 替换成常量 `12345`，而且候选 run 没有完整消费 EvalPlan 冻结的生成参数；
- 多 prompt 的 baseline 使用 prompt offset，后续 epoch/step 却丢失 offset，破坏同 prompt 跨轮次比较。

这些场景不能简单统一成“每张都随机”。训练 sample、XY 矩阵和 Eval
baseline/checkpoint 候选都依赖共享噪声，才能把图像变化归因于模型或参数差异。

## 候选方案

1. 在每次消费 seed 时把 `0` 临时随机化：实现局部简单，但 retry/resume、历史复现和比较组共享都会失效。
2. 保留各模块现状，仅修已报告的固定值：改动小，但继续存在互相矛盾的 `0` 语义和新的遗漏风险。
3. 在业务任务/比较组边界解析一次，持久化实际值，下游只消费已解析 seed：采用。

## 决策

### 公共规则

- 用户可配置 seed 的 `0` 统一表示随机哨兵；显式非零值保持原样。
- 随机值使用独立的系统随机源生成正整数，不消费训练用的 Python/Torch RNG。
- 一旦进入任务或比较组，实际 seed 必须冻结并随持久化快照传递。
- retry、pause/resume 复制原任务快照，不重新解析；新建任务重新解析。

### 各场景作用域

| 场景 | `0` 的解析边界 | 组内规则 |
|---|---|---|
| 训练主 `seed` | 新训练 task 入队 | 整个训练过程固定 |
| `sample_seed` | 新训练 task 入队 | 同一 prompt 跨 baseline/epoch/step 固定；不同 prompt 用稳定 index offset |
| `eval_validation_split_seed` | 新训练 task 入队 | 该 task 的验证集划分固定 |
| 普通测试出图 | 每张 task 在 daemon 中 | 每张独立随机；显式 seed 的 batch 继续按 index 递增 |
| XY 矩阵 | 每个 XY task 在 daemon 中 | 所有 cell 共享同一 seed |
| Eval Session | 创建 session / EvalPlan 时 | baseline 与所有 checkpoint 候选共享；验证图片按稳定 index offset |

训练预设和 version 源配置仍保留用户输入的 `0`；只有 task 专属
`snapshot/config.yaml` 写入实际值。这保证下一次新建任务会重新随机，而 retry/resume
仍严格复现。

推理 daemon 的 `image_done` 事件携带实际 seed。服务端用它更新 task 参数快照、历史记录、
缓存 sidecar 与 PNG metadata；不得继续保存提交时的 `0`。

Eval Session 优先从父训练 task 的冻结快照读取 `sample_seed`，以便训练后评估延续该次训练
的比较基线；没有父 task 的手动评估在创建 session 时独立解析并冻结。

## 理由

方案 3 同时满足两类看似冲突的需求：新任务获得真正随机的初始条件，同一比较组又具有固定
噪声。把解析放在任务边界而不是算法内部，也让重试、恢复、历史展示和元数据共享同一个
权威值，避免依赖隐式全局 RNG 状态。

不采用“所有图片都独立随机”，因为 XY、训练采样与 Eval 的主要用途是可控对比；不采用
“整个 batch 只随机一次”，因为普通批量测试的每张图是独立结果，不是比较矩阵。

## 后果

- `0` 生成的任务快照会与源预设不同，这是有意的物化行为。
- 任务快照会直接记录三个实际 seed，问题报告可以用该快照复现。
- 随机普通 batch 的多个 task 仍提交 `0`，由 daemon 为每张独立解析；前端不得自行生成伪随机序列。
- 老任务快照若仍包含 `sample_seed=0`，bootstrap 保留兼容兜底并用系统随机源解析；它无法补回历史上从未持久化的实际值。
- PNG metadata/历史的 seed 从“用户请求值”改为“实际生成值”，属于数据语义修正。

## 参考

- [随机种子用户说明](../user-guide/random-seeds.md)
- [ADR 0006：Queue 暂停 / 恢复](0006-queue-pause-resume.md)
- [ADR 0011：LoRA 评估指标](0011-lora-eval-metrics.md)
