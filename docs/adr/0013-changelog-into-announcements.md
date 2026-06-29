# 0013 — Changelog 迁入公告系统（移除 release_notes.yaml + 旧端点）

**状态**：Accepted
**日期**：2026-06-28
**决策者**：@WalkingMeatAxolotl
**相关**：公告栏设计 [`docs/todo/announcement-center.md`](../todo/announcement-center.md)（Phase 2）

## 背景

`release_notes.yaml` 是 changelog 的单一权威源（单文件、结构化 `kind/summary/detail`、纯中文），经：

- `tools/bump_version.py` → 校验 + 派生 `CHANGELOG.md`（按 kind 分组）；
- `studio/services/release_notes.py` + `/api/system/release_notes{,_all}` → 前端 Settings 版本段展示 + 历史 modal。

公告栏（announcement-center Phase 1）上线后，更新公告 / 迁移提示走 `docs/announcements/` 一篇一文件、双语。用户决定把 **changelog 也统一进公告栏**：一篇一文件、双语、`tag: release`、可直接在 GitHub 浏览；单文件 yaml 随版本增多（已 24 个）难维护、难翻历史。

## 决策

1. **数据迁移**：`release_notes.yaml` 的 24 个版本 → 每版一篇 `docs/announcements/<date>-v<version>.md`（`tag: release`，`version` 填该版本）。正文是从原结构化 entries 转出的 markdown（保留 kind 分组为 `**新增**` 等小标题 + bullet + detail）。**先迁中文**，en 回填见 Phase 3。迁完**删除 `release_notes.yaml`**。
2. **`bump_version.py` 重构**：
   - `render-changelog`：改从 `docs/announcements/` 的 `release` post（按 version 降序）生成 `CHANGELOG.md`（各版正文拼接 + 版本头）。
   - `validate`：改为校验 `release` post 的 frontmatter（version/date/tag/title 齐全、文件名与 version 自洽、version 顺序）。
   - `bump` / `verify-versions`：版本号同步逻辑不变。
3. **删旧后端**：移除 `studio/services/release_notes.py` + `/api/system/release_notes` + `/api/system/release_notes_all` 端点 + serializer。公告栏（`/api/announcements`）成为唯一来源。
4. **前端**：`SystemSection` 去掉 release-notes 展示 + 历史 modal（保留更新检查 / 更新 / 回滚）；版本段加「查看更新公告」入口 → 打开公告栏（默认过滤 `release`）。删 `client.ts` 的 `getReleaseNotes/getAllReleaseNotes` + `ReleaseNotes*` 类型 + 相关 i18n。
5. **文档**：`docs/release-notes-spec.md` 退役（作者指南统一到 `docs/announcements/README.md`）；改写 `CONTRIBUTING.md` 发版流程（发版 = 新增/编辑一篇 `docs/announcements/<version>.md`，不再改 yaml）。

## 理由 / 取舍

- **一篇一文件 + 双语 + GitHub 可读** 是用户明确诉求；单文件 yaml 难维护。
- **放弃**：CHANGELOG.md 按 kind 分组渲染 + `summary≤80 / kind 白名单 / pr_refs` 强校验（Q1 已定用自由 markdown 正文换 GitHub 可读 + 一篇一文件）。迁移时保留 kind 小标题，所以历史 changelog 的分组信息不丢，只是不再是工具强约束。
- **删端点**：公告栏已覆盖「看更新内容」，两套并存是重复。旧端点只读本地 yaml，删后无外部已知消费方（仅本仓前端）。

## 后果

- 发版流程变化：维护者发版时写一篇 markdown（双文件双语），而非编辑 yaml 顶部 block。CONTRIBUTING 同步。
- `CHANGELOG.md` 内容形态变化（分组 → 各版正文拼接）；仍自动生成、仍可被 GitHub Release body 复制。
- 一次性迁移脚本产出 24 篇 `release` post（中文）；Phase 3 补英文。
- 前端版本段简化；「更新内容」改由公告栏统一承载。

## 已决（2026-06-28）

- **Q-A｜CHANGELOG.md**：**留**，由 `bump_version` 从 release post 自动生成（各版正文拼接）。
- **Q-B｜bump_version 校验**：**保留轻校验** —— `validate` 校 release post frontmatter（version/date/tag/title 齐全、文件名与 version 自洽、版本顺序唯一）。
- **Q-C｜拆 PR**：**一个原子 PR**（迁数据 + 工具 + 删端点 + 改前端必须一起，否则中间态崩）。

## 参考

- [`docs/todo/announcement-center.md`](../todo/announcement-center.md)、Phase 1（公告栏基建）已合入 dev
- 迁移涉及：`release_notes.yaml`、`tools/bump_version.py`、`studio/services/release_notes.py`、`studio/api/routers/system.py`、`studio/web/src/pages/tools/settings/SystemSection.tsx`、`studio/web/src/api/client.ts`、`tests/test_bump_version.py`、`tests/test_release_notes.py`、`CONTRIBUTING.md`、`docs/release-notes-spec.md`
