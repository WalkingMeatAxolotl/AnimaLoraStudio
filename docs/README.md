# 文档

文档分六类，对应不同使用场景：

| 目录 | 给谁看 | 维护节奏 |
|---|---|---|
| [`user-guide/`](user-guide/) | 用户、社区贡献者 | 跟着行为变更随时更 |
| [`architecture/`](architecture/) | 开发者，要改代码或排查 bug | 架构调整时更 |
| [`adr/`](adr/) | 想知道「为什么是这样」的人 | 新决策时新增；老决策不改写，只追加状态 |
| [`design/`](design/) | 想了解某个 ADR 怎么讨论出来的人 | 设计阶段持续更新；ADR 落地后冻结作为参考 |
| [`todo/`](todo/) | 维护者；记"现在做不了、未来要检查"的事 | 触发条件到了再回来处理或归档 |
| [`announcements/`](announcements/) | 用户；app 内公告栏 + 派生 `CHANGELOG.md` | 每个 user-facing PR / 发版时写一篇 |

> **不在这里**：版本变更见根目录 [`CHANGELOG.md`](../CHANGELOG.md)；Studio 内部模块结构见 [`studio/README.md`](../studio/README.md)。

---

## User guide

| 文档 | 内容 |
|---|---|
| [tagging-guide.md](user-guide/tagging-guide.md) | Anima 标签格式、最佳实践、tag 顺序 |
| [training-tips.md](user-guide/training-tips.md) | 训练参数、显存配置矩阵、过拟合/欠拟合排查、ComfyUI 用法 |
| [optimizers.md](user-guide/optimizers.md) | 各优化器（Lion / Prodigy / PPSF / SOAP 等）起步参数与换算 |
| [regularization.md](user-guide/regularization.md) | 正则集生成原理（tag 分布贪心搜索 + AR 聚类） |
| [caption-format.md](user-guide/caption-format.md) | JSON caption 格式 + 分类 shuffle |

## Architecture

| 文档 | 内容 |
|---|---|
| [studio-pipeline.md](architecture/studio-pipeline.md) | 跨步骤架构总览：数据模型、目录布局、SQLite schema、secrets、SSE 事件、Tagger 抽象、Preset 池 |

## Architecture Decision Records (ADR)

历史决策记录。记录「我们为什么选 X 而不选 Y」，已落地的就是历史，**不删**——保留是为了未来想反悔时知道当初的取舍。

| ADR | 状态 | 内容 |
|---|---|---|
| [0001-lokr-via-lycoris-lora.md](adr/0001-lokr-via-lycoris-lora.md) | Accepted（2025） | LoKr 改走官方 lycoris-lora 库，而不是切到 sd-scripts |
| [0011-lora-eval-metrics.md](adr/0011-lora-eval-metrics.md) | Accepted | LoRA 评估指标体系、异步 eval job 边界与可审查 PR 拆分 |

详见 [adr/README.md](adr/README.md)。

---

## 本地草稿

`docs/_local/` 已加入 `.gitignore`。在仓库内随手记笔记、写未定稿设计、临时 TODO，放这个目录就不会污染提交。
