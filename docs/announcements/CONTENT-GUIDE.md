# 公告内容编写规范

**适用**：`docs/announcements/` 下所有公告 post 的**写什么 / 怎么写**——`release`（更新日志）、
`notice`（公告）、`migration`（行为变化 / 需注意）及未来类型。
**文件格式 / frontmatter / tag / 工具**见 [`README.md`](README.md)，本文只管内容与文风。

> 本规范综合了 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)、
> [Apple HIG · Writing](https://developer.apple.com/design/human-interface-guidelines/writing)
> 与业界 release notes 实践（[ProductPlan](https://www.productplan.com/learn/release-notes-best-practices)、
> [Appcues](https://www.appcues.com/blog/release-notes-examples)）。

---

## 第一原则（所有公告通用）

1. **写给用户，不写给工程师**。描述用户体验的变化，不是工程实现。用户不读 release notes
   多半不是不在乎，而是它是按工程视角写的。
2. **可扫读：每条第一句就答「变了什么 + 跟我有没有关」**。用户扫公告只想知道这两件事，
   让答案落在第一句。
3. **平实语言，少术语**。避免未经解释的专业词 / 内部代号 / 实现细节；像跟一个非程序员朋友解释。
4. **只写 vs 上个版本用户能感知的差异**。开发周期内引入又修掉的 bug 不写；防回归不是 fix；
   纯内部重构 / 个例 workaround 不写。一条 entry = 用户能感知的一个 unit of change。
5. **简洁**。每个词都检查是否必要，能少则少。主动语态、现在时。

## 各类公告写什么

### `release` —— 更新日志（一版一篇）
- 标题 `title`：`zh「v0.16.0 更新」/ en「v0.16.0 release」`；`date` 用 ISO `YYYY-MM-DD`；最新版在公告栏最前。
- 正文按变更类型分组（Keep a Changelog 的类别）：**新增 / 变更 / 改进 / 修复 / 弃用 / 删除 / 安全**。
  只列本版实际有的组。
- 每组下每条一行用户视角的话，结尾带 PR 号 `（#NN）`；需要时下面补一段细节 / 子要点。
- **弃用 / 破坏性变更必须显眼列出**，让用户能规划升级——它们最该升级为一篇 `migration`（见下）。

### `notice` —— 一般公告
- 通知 / 提示类（如某功能上线、维护说明、社区信息）。一句话讲清是什么 + 对用户意味着什么。
- 不必按 kind 分组；按一条信息一篇。

### `migration` —— 行为变化 / 需要注意或操作（最重要）
触发：用户能感知的行为变了、默认值变了、入口 / 地址 / 路径变了、需要用户做点什么。

- **开头一句讲清「变了什么」**，别埋在细节里。
- **必给行动指引**——这是 migration 区别于普通公告的核心：
  - 不需要用户做事 → 明说「**自动生效，无需操作**」。
  - 需要 → 给可执行的具体步骤，如「去 **设置 → X → Y** 改一下」「旧书签会自动跳转，建议更新为…」。
- 重要的设 `pin: true` 置顶。

## 语气 / 人称 / 时态 / 长度
- 语气：平实、友好、有调性但不牺牲信息量；不口语化堆梗。
- 人称：对用户用「你 / you」；讲变化用主动语态、现在时（"现在可以…"，不是"我们重构了…"）。
- 长度：一条一行能说清就一行；细节进下面的子段，别写成大段墙。

## 不要写
- 内部重构 / 模块挪动 / 依赖升级等用户无感的工程改动（除非带来可感知行为变化）。
- 「修复了一个回归」——防回归不是用户视角的 fix（那个 bug 用户从没见过）。
- 个例 workaround / 调试细节 / commit log 式罗列（noise）。
- 夸大（"彻底解决""完美支持")、未经解释的缩写和代号。
- 技术对照（"与业界对齐""参考了 X 的实现"）——这些放 PR description，不进公告。

## 结构 / markdown 约定
- 用真 markdown：`### 分组标题`、`- ` 列表、必要时**一层**嵌套子要点（缩进对齐）；少用更深嵌套。
- 链接给完整 URL 或 markdown 链接；代码 / 路径 / 字段名用 `` ` `` 包。
- 日期 ISO `YYYY-MM-DD`；避免地区歧义。
- 别整段加粗 / 全大写制造"重点"；醒目靠结构（置顶、第一句、分组），不靠排版噪音。

## 双语
- `zh` 和 `en` 各自写地道，不是互相机翻：英文也按英文母语者的 release notes 习惯写（Apple 那种简洁友好），别中式英语。
- 标题约定见各类型；正文两边信息等价即可，不必逐字对应。

## 参考
- [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
- [Apple HIG · Writing](https://developer.apple.com/design/human-interface-guidelines/writing)
- [ProductPlan · Release notes best practices](https://www.productplan.com/learn/release-notes-best-practices)
- [Appcues · 13 release notes examples](https://www.appcues.com/blog/release-notes-examples)
