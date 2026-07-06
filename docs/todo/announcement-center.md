# 设计草案 — 统一公告栏（Announcement Center）

**状态**：✅ 已实现（Phase 1 #336 / Phase 2 #337 / Phase 3 #339）— 保留作设计记录
**日期**：2026-06-28
**相关**：升级现有 `release_notes` 展示；首篇公告承载 ADR 0012（`/studio→/`）迁移提示

> 本文是 plan，不是最终决策记录。已确认项进「决策」；未定项进「待定」。实现拆 PR 见「分阶段」。

## 背景 / 动机

- 版本更新后用户不知道变了什么；撞到行为变化（如 #334 把入口从 `/studio/` 改到 `/`）会发懵，即便做了 307 兼容跳转仍困惑。
- 现状缺一个**主动 push** 的更新公告：现有 changelog 是 **pull 式**（用户得自己去 Settings → 系统翻），而且：
  - **纯中文**（`release_notes.yaml` 的 summary/detail 只有中文），英文用户无更新信息。
  - **单文件**（`release_notes.yaml` 随版本增多越来越长，翻历史版本困难）。
  - 缺「需要注意 / 需要迁移」这类**操作型提示**（factual changelog 按规范故意不写指令）。

## 目标

1. **统一公告栏弹窗**（modal，非独立页面）：左 list（tag 过滤 + 每篇红点）/ 右正文，仿游戏公告。
2. **一篇一文件**：所有 post（更新日志 + 公告 + 迁移）迁入 `docs/announcements/`，一个 post 一个文件，便于翻历史、git diff 友好。
3. **双语**：post 双语（zh/en）；现有 changelog **回填全部历史到英文**。
4. **主动 + 常驻双入口**：更新后有未读自动弹一次；Topbar 常驻入口 + 未读红点随时点开。
5. **复用既有模式**：触发/持久化仿 `FirstRunOnboardingModal`；文案走 i18n；markdown 渲染复用现有 release notes detail 的渲染器。

## 不在范围

- 不替代「完整 factual changelog」的存在；它成为公告栏里 `更新日志` tag 的 post。
- 不做 in-app 自动迁移（只告知，不替用户改）。
- 不做服务器远程推送公告（本地工具，公告随 git 更新到达——正好是需要它的时机）。

## 用户故事（主流程）

1. app 加载 → 读 `/api/health` 当前版本 + 拉公告列表 → 比对 localStorage 已读集合。
2. **全新安装**（无已读记录）→ 把当前所有 post 标记已读，**不弹**（新用户不被历史公告/迁移提示打扰）。
3. 更新后存在**未读** post → **自动弹**公告栏一次；Topbar 图标显示未读红点 + 数。
4. 公告栏：左 list（按 tag 过滤，未读带红点）/ 右正文（按当前 i18n 语言渲染 markdown）。
5. 读过的 post 红点消失（写入 localStorage 已读集合）；全读完 Topbar 红点消失。

## 决策（已确认）

### D1 — 形态：弹窗，升级现有 modal
公告栏是 **modal**，本质是把现有 `release_notes_all`（已有的「历史版本切换 modal」）**重构升级**成统一公告栏：加 tag 过滤 + 红点 + 双语 + 公告类 post + Topbar 入口 + 更新后自动弹。Settings → 系统里现有的 release-notes 展示迁进来 / 改为「打开公告栏」入口。

### D2 — 源：一篇一文件，全部迁入 `docs/announcements/`
- 废弃单文件 `release_notes.yaml`；**每个 post 一个文件**。
- 双语用**双文件**约定（对齐仓库现有 `README.md` / `README.en.md`）：
  - `docs/announcements/<id>.md`（中文）
  - `docs/announcements/<id>.en.md`（英文）
  - en 缺失 → fallback 显示 zh（保证不空）。
- `<id>` 稳定唯一，用作 read 状态 key。建议 `YYYY-MM-DD-<slug>`，如 `2026-06-28-v0.16.0`、`2026-06-28-url-root`。

### D3 — 每篇 frontmatter（草案）
```yaml
---
id: 2026-06-28-url-root        # 稳定唯一，read 状态靠它
date: 2026-06-28               # ISO，列表排序 + 显示
tag: migration                 # 见 D4
title: 入口地址改为根路径       # 该语言标题（en 文件里是英文）
pin: false                     # 可选：置顶（重要迁移）
version: 0.16.0                # 可选：关联版本；更新日志类必填
---
（正文 markdown，该语言）
```

### D4 — tag 分类（初始集，可扩）
- `release`（更新日志）— 由历史 changelog 迁移而来，每版一篇。
- `notice`（公告）— 一般通知。
- `migration`（迁移）— 行为变化 / 需要用户注意或操作（如 `/studio→/`）。

公告栏顶部按 tag 过滤。颜色/图标各 tag 区分。**首篇 migration post = `/studio→/`**。

### D5 — 双语 changelog + 回填
- 现有 `release_notes.yaml` 每个**版本**迁成一篇 `release` post（一版一文件）。
- **正文是自由双语 markdown**（不再保留结构化 `kind`/`summary`/`pr_refs`）——`docs/announcements/` 这堆 per-version markdown 本身就是**可直接在 GitHub 浏览**的 changelog。
- **回填全部历史版本到英文**（agent 翻译，PR1 的大头）。
- **取舍（Q1 已定）**：放弃 CHANGELOG.md 按 kind 分组渲染 + `bump_version` 的 schema 校验（summary≤80 / kind 白名单 / pr_refs int）。换来一篇一文件 + GitHub 可读 + 双语。
- **CHANGELOG.md**：退化为「各版 release post zh 正文的拼接」（保留文件给现有消费方 / GitHub Release body 复制用），由 `bump_version` 从 per-file 重新生成；不再手维护。GitHub Release / CHANGELOG **先保持中文**（待定 Q3）。

### D6 — read 状态 / 触发
- localStorage 存**已读 id 集合**（`studio.announcements.read`）。
- 未读 = 全部 post − 已读集合。Topbar 红点 = 未读数。
- 自动弹：当前版本 > 上次记录版本 且存在未读 → 弹一次。
- 首次安装：把当前全部 id 写入已读集合，不弹。
- 纯前端 + localStorage，read 状态不进后端。

### D7 — 后端 / 前端职责
- **后端**：新 service 读 `docs/announcements/*.md`，解析 frontmatter + 配对 zh/en，返回结构化双语 post 列表。新端点（取代 `/api/system/release_notes` + `release_notes_all`）。
- **前端**：公告栏 modal 组件（list + tag filter + 红点 + 正文 markdown）；Topbar 入口；自动弹逻辑（仿 onboarding 的 event + localStorage）；i18n 文案（标题/按钮，非 post 内容）。

### D8 — 合并现有「版本更新铃铛」
Topbar 现有的「有新版本」pill（`Topbar.tsx`：`checkSystemUpdate('master')` → 显示 `latest_tag` → 点击 `settingsDrawer.open({section:'version'})`）**并入公告栏**：
- **该位置改成公告栏入口（铃铛图标）**，不再单独放更新 pill。
- 「有新版本可更新」放公告栏 **header 角落一个小按钮**（不是大 banner）；保留 `checkSystemUpdate` 逻辑，点击 → 跳 `Settings → version` section。
- 铃铛红点 = **有未读公告 _或_ 有可用更新**。
- 语义区别（两者都进同一个铃铛但表现不同）：
  - **公告** = 你**已装**版本「更新了啥 / 要注意啥」——读后即已读、红点消失。
  - **可用更新** = 有**更新的版本你还没装**——actionable banner，装完（版本号变了）才消失，不是「已读」。

## 数据流（API 草案）

`GET /api/announcements` → `{ posts: [{ id, date, tag, version?, pin?, title: {zh,en}, body: {zh,en} }] }`（按 date desc + pin 优先）。前端按当前语言取 title/body，缺 en → zh。

## 分阶段（先基建 → 再增量迁移 → 翻译最后）

顺序原则：基建是有价值、可独立 review 的单元；内容迁移机械、可一步步来；英文回填最枯燥、架构零风险，放最后。

- **Phase 1 — 公告栏基建（infra）**【已合入 dev：#336】
  - 定义 `docs/announcements/` 格式（frontmatter + zh/en 双文件）。
  - 后端：announcements service（读 `docs/announcements/*.md` + 配对 zh/en + 解析 frontmatter）+ `GET /api/announcements`。
  - 前端：公告栏 modal（list + tag 过滤 + 红点 + 正文 markdown）+ 自动弹 + read 状态（localStorage）。
  - **Topbar：移除现有「有新版本」pill，该位置换成公告栏铃铛入口（D8）**；可用更新检查并入公告栏顶部 banner；铃铛红点 = 未读公告 _或_ 可用更新。
  - 种 1–2 篇**手写**双语 post 验证（首篇 `migration` = `/studio→/`，新写所以直接双语；可加一篇 welcome `notice`）。
  - **不动** `release_notes.yaml` / 现有 Settings changelog —— 暂共存，下个 phase 再迁。
  - 产出：一个自包含可 review 的 PR；公告栏能跑、能显示新 post。

- **Phase 2 — 迁移现有 changelog 进公告栏**【本 PR · 见 ADR 0013】
  - `release_notes.yaml` → 每版一篇 `docs/announcements/<version>.md`（先迁现有**中文**内容，en 暂缺）。
  - 重构 `tools/bump_version.py`（新格式 + 从 per-file 生成 CHANGELOG.md）+ 改 CONTRIBUTING 发版流程。
  - 现有 Settings release-notes 展示 → 收进公告栏 / 废旧端点（`/api/system/release_notes*`）。
  - 此时 `release` post 仅中文（en 缺失 → fallback zh）。可分多次小步迁。
  - **大概率配 ADR**（发版格式/流程变更 + 移除 `release_notes.yaml` 单一权威源）。

- **Phase 3 — 英文回填（最后）**
  - 给全部历史 `release` post 补 `<version>.en.md`。可分批（近版优先）。纯内容，零架构风险。

依赖：Phase 1 不依赖 2/3；Phase 2 依赖 1 的格式；Phase 3 依赖 2 的文件。每个 phase 可独立成 1 个（或多个）PR。

## 待定（open questions）

- ~~Q1 — changelog 迁移后是否保留结构化 kind 分组~~ **已定：用自由 markdown 正文**（见 D5），放弃 kind 分组 + schema 校验，换 GitHub 可读 + 一篇一文件。
- **Q2 — tag 最终集**：`release`/`notice`/`migration` 够不够？要不要 `event`(活动) / `fix`(重要修复) 等。
- **Q3 — CHANGELOG.md / GitHub Release 是否也双语**：本草案先保持中文。
- **Q4 — 自动弹的「上次版本」记录**：用 localStorage 版本号 vs 已读集合是否为空——两者交互细节实现时定。
- **Q5 — 是否给本特性单独 ADR**（见 PR1）：发版流程 + 单一权威源变更达到 ADR 触发条件。

## 影响文件（预估）

- 新增：`docs/announcements/*.md`(+`.en.md`)、`studio/services/announcements.py`、`studio/api/routers/` 公告端点、`studio/web/src/components/AnnouncementCenter*.tsx`、Topbar 入口、i18n keys。
- 改：`studio/web/src/components/Topbar.tsx`（Phase 1：移除「有新版本」pill → 公告栏铃铛入口，update 检查并入 banner）、`tools/bump_version.py`、`CONTRIBUTING.md`（release 流程）、`studio/web/src/pages/tools/settings/SystemSection.tsx`（迁移旧展示）、`studio/api/routers/system.py`（废旧端点）。
- 删/迁：`release_notes.yaml`（迁成 per-file）。
