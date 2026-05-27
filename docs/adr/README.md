# Architecture Decision Records

记录架构层面的「**我们选 X 而不选 Y**」的决策与理由。代码会说"是什么"，但说不清楚"为什么不那么做"——ADR 填这个空。

## 索引

| # | 标题 | 状态 | 日期 |
|---|---|---|---|
| 0001 | [LoKr 适配器走 lycoris-lora 而不切 sd-scripts](0001-lokr-via-lycoris-lora.md) | Accepted | 2025 |
| 0002 | [Webui 内自更新（flag + shell wrapper loop）](0002-webui-self-update.md) | Proposed | 2026-05-12 |
| 0003 | [anima_train.py 模块化重构（plugin 边界 + adapter hook protocol）](0003-anima-train-refactor.md) | Proposed | 2026-05-14 |
| 0004 | [预处理状态用单 manifest 替代「双 bucket + per-image sidecar」](0004-preprocess-manifest.md) | Superseded by #0010 | 2026-05-15 |
| 0005 | [更新通道作为用户视图偏好，与 git 工作树状态解耦](0005-update-channel-as-preference.md) | Accepted | 2026-05-16 |
| 0006 | [Queue 任务暂停 / 恢复 + 队列挂起 / 恢复调度](0006-queue-pause-resume.md) | Accepted | 2026-05-18 |
| 0007 | [Project / Version / Task 生命周期重构](0007-project-version-lifecycle-refactor.md) | Proposed | 2026-05-23 |
| 0008 | [studio/ 4 层重构（0.11.0）](0008-studio-restructure-0.11.0.md) | Accepted | 2026-05-28 |
| 0010 | [preprocess scope 从项目级 download 下沉到 version 级 train](0010-preprocess-train-scope.md) | Accepted | 2026-06-03 |
| 0011 | [LoRA 评估指标体系与可审查 PR 拆分](0011-lora-eval-metrics.md) | Proposed | 2026-05-27 |

## 状态值

- **Proposed** — 草拟中，未决
- **Accepted** — 已采纳并落地
- **Superseded by #N** — 被新 ADR 取代（保留原文，不删）
- **Deprecated** — 不再适用（标记原因，保留原文）

## 写一份新 ADR

```markdown
# NNNN — 简短标题（动词起头）

**状态**：Proposed | Accepted | Superseded by #N | Deprecated
**日期**：YYYY-MM-DD
**决策者**：@handle / 团队

## 背景

当时面对的问题、约束、外部因素。让未来读者不需要回到当时的语境也能看懂。

## 候选方案

简要列出讨论过的所有方案，包括最后没选的。每个方案给出优缺点。

## 决策

选了哪个、做了什么。

## 理由

为什么这么选。重点写**否决其他方案的具体理由**——这才是 ADR 的核心价值。

## 后果

落地后带来的好处、新增的约束、未来可能要还的债。

## 参考

链接到相关 PR、commit、外部资料。
```

文件名规则：`NNNN-kebab-case-title.md`，编号四位数字递增。
