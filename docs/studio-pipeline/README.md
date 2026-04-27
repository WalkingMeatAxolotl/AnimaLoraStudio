# Studio Pipeline 改造文档

把 Studio 从「散件工具集」升级为「项目流水线」：每次 LoRA 训练对应一个 `Project`，从建项 → 下载数据 → 筛选 → 打标 → 正则集 → 配置 → 入队，按 step 推进，所有产物按 project + version 组织。

## 文档结构

| 文档 | 范围 |
|---|---|
| [overview.md](overview.md) | 架构总览：数据模型、目录布局、SQLite schema、secrets.json、Sidebar、SSE、Tagger 抽象、Preset 关系、复用现有组件清单 |
| [pp0-restructure.md](pp0-restructure.md) | ✅ 已完成 — Sidebar 重构 / `configs/` → `presets/` 迁移 / Settings 页 / 旧 Datasets 删除 |
| [pp1-project-version.md](pp1-project-version.md) | ✅ 已完成 — Project + Version 数据模型、CRUD、Stepper、VersionTabs |
| [pp2-download.md](pp2-download.md) | Gelbooru/Danbooru 下载集成、project_jobs、Job 守护扩展 |
| [pp3-curation.md](pp3-curation.md) | download / train 双面板、缩略图、复制/移除、子文件夹管理 |
| [pp4-tagging.md](pp4-tagging.md) | Tagger 协议、WD14 (本地+HF 自动下载)、JoyCaption (vLLM)、单图标签编辑、批量增/删/去重/统计 |
| [pp5-regularization.md](pp5-regularization.md) | 基于 train tag 分布生成正则集，version 级 |
| [pp6-train.md](pp6-train.md) | Preset fork/save、Train 页、入队带 project_id+version_id、monitor 按 version 拆分 |

## 实施原则

- **每阶段一个 PR**：PP0 独立走，PP1-PP6 顺序推进。
- **每 PR 完成停下来手测**：每个 step 文档末尾给出「手测剧本」。
- **不破坏现有功能**：训练、配置编辑、监控、队列都继续可用。Configs 路由保留兼容直到 PP0 完成迁移。
- **Studio 必须能在 Windows 下跑**：因此外部依赖（vLLM）保持「用户自己起」，Studio 只通过 URL 调用。
- **复用 P1-P5 已有积木**：`schema.TrainingConfig`、`db.tasks` (+ supervisor)、`event_bus`、`Toast`、`PathPicker`、`SchemaForm`、SSE pattern。新增一层 `projects` / `versions` / `project_jobs` 串起来。

## 关键设计决定（讨论确认）

| # | 决定 | 备注 |
|---|---|---|
| 1 | `download/` 在 project 级共享；`train/`、`reg/`、`output/`、`samples/`、`monitor_state.json` 全部进 `versions/{label}/` | reg 跟 train 走，因为 train 变 → tag 分布变 |
| 2 | version label 由用户填（不自动 v1/v2） | 鼓励用 `baseline` / `high-lr` 这类语义名 |
| 3 | `configs/` → `presets/` 重命名；预设是全局池 | Version 与池双向流：fork / save_as_preset |
| 4 | Sidebar：项目 / 队列 同顶级；预设、监控、设置在工具组 | 旧 Datasets 页删除 |
| 5 | WD14 + JoyCaption 同时支持，下拉切换 | WD14 优先（本地无服务依赖） |
| 6 | WD14 模型解析：本地 → HuggingFace 自动下载 | 用户可指定 `secrets.wd14.local_dir` 强制本地 |
| 7 | JoyCaption 用户自己起 vLLM，UI 只填 `base_url` | Studio 不在 Win 下管 vLLM 进程 |
| 8 | 所有 token / API key 集中到 `studio_data/secrets.json` | `studio_data/` 已 git-ignore |
| 9 | 项目软删（移到 `_trash/`） | 显式按钮清空 trash |
| 10 | Curation 双面板用文件名做差集；磁盘 `download/` 永远全量 | 前端展示去交集，后端不删 |

## 时间表（建议）

PP0 → PP1 → PP2 → PP3 → PP4 → PP5 → PP6，**每一步都等你手测过再继续**。预计每 PP 1-2 个工作日，整体 2-3 周可上线。
