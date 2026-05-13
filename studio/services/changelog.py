"""解析 CHANGELOG.md，按 tag 返回该版本的 release notes（ADR 0002 / chunk 2）。

格式假定 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)：

    ## [0.6.0] — 2026-05-12
    ### 新增
    - **<title>**（#NN）
      - 子细节（忽略）
    ### 变更
    - ...

VersionSection 拉来填进 master 卡的 change-block。CHANGELOG 没有这条 tag
时返回 found=False，UI 给链接占位即可。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..paths import REPO_ROOT

CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"

# `## [0.6.0] — 2026-05-12` / `## [0.6.0]` — em-dash / en-dash / hyphen 都允许
_VERSION_HEADING_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(?:[—–-]\s*(\d{4}-\d{2}-\d{2}))?")
_SECTION_HEADING_RE = re.compile(r"^###\s*(.+?)\s*$")
_TOP_BULLET_RE = re.compile(r"^- (.+)$")


@dataclass
class ReleaseNotesSection:
    """单个 H3 分组（新增 / 变更 / 修复 / 改进 / 等）+ 顶层 bullet 列表。"""
    title: str
    items: list[str] = field(default_factory=list)


@dataclass
class ReleaseNotesResult:
    tag: str                                  # caller 传入的原始 tag（v 前缀保留）
    found: bool
    date: Optional[str] = None                # ISO YYYY-MM-DD
    sections: list[ReleaseNotesSection] = field(default_factory=list)


def _normalize_tag(tag: str) -> str:
    """`v0.6.0` → `0.6.0`。CHANGELOG header 用 `[0.6.0]` 不带 v 前缀。"""
    return tag.lstrip("vV").strip()


def _strip_top_bullet(line: str) -> str:
    """`- **<title>**（#NN）` → `<title>（#NN）`；其他 `- xxx` → `xxx`。

    脱掉 markdown **bold** 包装让前端简单渲染（保留 PR 号方便跳转）；非 bold
    的整体直接返回。
    """
    m = _TOP_BULLET_RE.match(line)
    if not m:
        return line
    text = m.group(1).strip()
    if text.startswith("**") and "**" in text[2:]:
        end = text.index("**", 2)
        bold = text[2:end]
        rest = text[end + 2:].strip()
        return f"{bold}{rest}" if rest else bold
    return text


def parse(tag: str) -> ReleaseNotesResult:
    """读 CHANGELOG.md，返回指定 tag 的 release notes。

    - tag 带不带 'v' 前缀都吃
    - CHANGELOG 不存在 / 读失败 / 找不到该 tag → found=False
    - 仅返回**顶层** bullet（嵌套的子细节忽略，保持 release notes 简洁）
    - 没有任何 bullet 的 H3 section 自动丢弃（避免空 "### 测试" 等）
    """
    norm = _normalize_tag(tag)
    if not CHANGELOG_PATH.exists():
        return ReleaseNotesResult(tag=tag, found=False)
    try:
        # utf-8-sig 容忍可能的 BOM（与 updater.py 一致）
        lines = CHANGELOG_PATH.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return ReleaseNotesResult(tag=tag, found=False)

    start_idx: Optional[int] = None
    date: Optional[str] = None
    for i, line in enumerate(lines):
        m = _VERSION_HEADING_RE.match(line)
        if m and _normalize_tag(m.group(1)) == norm:
            start_idx = i + 1
            date = m.group(2)
            break

    if start_idx is None:
        return ReleaseNotesResult(tag=tag, found=False)

    sections: list[ReleaseNotesSection] = []
    current: Optional[ReleaseNotesSection] = None
    for line in lines[start_idx:]:
        if line.startswith("## ") or line.strip() == "---":
            break
        sec_m = _SECTION_HEADING_RE.match(line)
        if sec_m:
            current = ReleaseNotesSection(title=sec_m.group(1).strip(), items=[])
            sections.append(current)
            continue
        if current is None:
            continue
        if line.startswith("- "):
            current.items.append(_strip_top_bullet(line))

    sections = [s for s in sections if s.items]
    return ReleaseNotesResult(tag=tag, found=bool(sections), date=date, sections=sections)
