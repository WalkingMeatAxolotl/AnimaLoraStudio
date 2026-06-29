"""公告栏数据源 —— 把 `docs/announcements/` 下的 markdown 解析成结构化双语 post。

设计见 [`docs/todo/announcement-center.md`](../../docs/todo/announcement-center.md)。

约定（Phase 1）：
- **一篇一文件**：每个 post 一个 markdown 文件，`---` frontmatter + 正文。
- **双语双文件**（对齐仓库 `README.md` / `README.en.md`）：
  - `<id>.md`     —— 中文（必有）
  - `<id>.en.md`  —— 英文（可选；缺失 → fallback 用中文）
  - `id` = 文件名 stem（去掉 `.en`），同时是前端 read 状态的 key。
- frontmatter 字段：`date`(必) / `tag`(必，白名单见 VALID_TAGS) / `title`(必) /
  `pin`(可选，默认 false) / `version`(可选，关联版本)。

只读、无副作用；read 状态在前端 localStorage，不进后端。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from ..infrastructure.paths import ANNOUNCEMENTS_DIR

logger = logging.getLogger(__name__)

# tag 白名单。新增 tag 在这里加一行（前端的颜色/文案也要同步）。
VALID_TAGS = frozenset({"release", "notice", "migration"})


@dataclass
class AnnouncementPost:
    id: str
    date: str
    tag: str
    title: dict[str, str]          # {"zh": ..., "en": ...}
    body: dict[str, str]           # {"zh": ..., "en": ...}
    pin: bool = False
    version: Optional[str] = None


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """解析 `---\\n<yaml>\\n---\\n<body>`。无合法 frontmatter → ({}, 全文)。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text
    try:
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    body = "\n".join(lines[end + 1:]).strip("\n")
    return meta, body


def _read(path: Path) -> Optional[tuple[dict[str, Any], str]]:
    try:
        return _split_frontmatter(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def list_posts(directory: Optional[Path] = None) -> list[AnnouncementPost]:
    """读目录下全部 post，配对 zh/en，返回排序后的列表（pin 优先 → date 降序）。

    `directory` 默认 `ANNOUNCEMENTS_DIR`；测试传 tmp 目录。目录不存在 → []。
    缺字段 / tag 非法的文件跳过（log warning），不让一个坏文件拖垮整个公告栏。
    """
    base = directory if directory is not None else ANNOUNCEMENTS_DIR
    if not base.is_dir():
        return []

    posts: list[AnnouncementPost] = []
    for zh_path in sorted(base.glob("*.md")):
        if zh_path.name.endswith(".en.md"):
            continue
        post_id = zh_path.stem
        zh = _read(zh_path)
        if zh is None:
            continue
        meta, zh_body = zh

        date = str(meta.get("date", "")).strip()
        tag = str(meta.get("tag", "")).strip()
        title_zh = str(meta.get("title", "")).strip()
        if not (date and tag and title_zh):
            logger.warning("announcement %s 缺 date/tag/title，跳过", zh_path.name)
            continue
        if tag not in VALID_TAGS:
            logger.warning("announcement %s tag=%r 不在白名单，跳过", zh_path.name, tag)
            continue

        # en 配对：缺文件 / 缺字段时 fallback 中文
        title_en, body_en = title_zh, zh_body
        en = _read(base / f"{post_id}.en.md")
        if en is not None:
            en_meta, en_body = en
            if (t := str(en_meta.get("title", "")).strip()):
                title_en = t
            if en_body.strip():
                body_en = en_body

        version = meta.get("version")
        posts.append(AnnouncementPost(
            id=post_id,
            date=date,
            tag=tag,
            title={"zh": title_zh, "en": title_en},
            body={"zh": zh_body, "en": body_en},
            pin=bool(meta.get("pin", False)),
            version=str(version) if version is not None else None,
        ))

    # 排序：pin 优先，再按 (date, id) 降序。stable sort 多趟。
    posts.sort(key=lambda p: (p.date, p.id))
    posts.reverse()
    posts.sort(key=lambda p: not p.pin)
    return posts
