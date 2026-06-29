# 公告 post 编写指南

本目录的 markdown 会被后端解析、在 app 内「公告栏」（右上角铃铛）展示。
整体设计见 [`docs/todo/announcement-center.md`](../todo/announcement-center.md)。

**本文管「文件格式 / frontmatter / tag / 工具」；「写什么 / 怎么写」（各类公告的内容与文风）见
[`CONTENT-GUIDE.md`](CONTENT-GUIDE.md)。**

## 一篇 = 两个文件（双语）

- `<id>.md` —— 中文（必有）
- `<id>.en.md` —— 英文（可选；缺失时该篇英文 fallback 用中文）
- `<id>` = 文件名去掉 `.md` / `.en.md`，也是前端「已读」状态的 key
  —— **改名会被当成新公告、让用户重新看到**。
- 命名建议 `YYYY-MM-DD-slug`，如 `2026-06-28-url-root`。

## frontmatter 字段

```yaml
---
date: 2026-06-28         # 必填，ISO 日期；列表排序 + 显示
tag: migration           # 必填，见下方枚举
title: 访问地址改为根路径   # 必填，该语言标题（.en.md 里写英文）
pin: true                # 可选，默认 false；置顶（重要迁移常用）
version: "0.16.0"        # 可选，关联版本号
---
正文（markdown）
```

> 正文目前前端按纯文本 `whitespace-pre-wrap` 显示（依赖最少，真 markdown 渲染待后续）。
> 所以请把正文写得**适合纯文本阅读**：少用 `#` 标题（会显示成原文），`-` 列表、链接 URL 没问题。

## tag 枚举（白名单）

| tag | 含义 | 何时用 |
|---|---|---|
| `release` | 更新日志 | 一个版本的发布说明（Phase 2 起从 release_notes 迁入，一版一篇） |
| `notice` | 公告 | 一般通知 / 提示 |
| `migration` | 迁移 | 行为变化、需要用户注意或操作（如入口地址变更） |

不在白名单的 tag 会被后端**直接跳过**（不显示）。运行时权威是
`studio/services/announcements.py` 的 `VALID_TAGS`，本表与它保持一致。

## 加一个新 tag 要同步这 5 处

1. `studio/services/announcements.py` → `VALID_TAGS`
2. `studio/web/src/api/client.ts` → `AnnouncementPost['tag']` 联合类型
3. `studio/web/src/components/AnnouncementCenter.tsx` → `TAG_ORDER` + `tagChipClass`（配色）
4. `studio/web/src/i18n/locales/zh.json` + `en.json` → `announcements.tags.<new>`
5. 本文件上面的 tag 表

## 模板

直接抄现成的：`2026-06-28-welcome.md` / `.en.md`（`notice`）或
`2026-06-28-url-root.md` / `.en.md`（`migration`，置顶）。
