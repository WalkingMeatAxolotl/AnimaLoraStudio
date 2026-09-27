---
date: 2026-09-27
tag: migration
title: 随机种子 0 现在按任务或比较组冻结
pin: true
version: "0.28.0"
---
从 v0.28.0 起，所有 seed 字段里的 `0` 都表示“创建任务时随机解析”，但随机边界会按用途保持一致。

**自动生效，无需操作。**

- 新训练任务分别冻结训练、采样和验证集 seed；Retry、暂停和恢复继续使用任务快照里的实际值。
- 普通批量出图每张图独立随机；XY 矩阵和同一评估会话共享一个比较 seed，避免噪声差异干扰对比。
- 生成历史、PNG metadata 和任务快照保存实际 seed，不再只留下请求时的 `0`。

需要跨任务严格复现时，请填写非零 seed；详细规则见[随机种子与可复现性](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/blob/master/docs/user-guide/random-seeds.md)。
