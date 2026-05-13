"""读 release_notes.yaml 返回结构化数据（ADR 0002 / chunk 2 重做）。

yaml 是 source of truth；CHANGELOG.md 由 tools/bump_version.py render-changelog
派生。编写规范见 docs/release-notes-spec.md。

模块仅做读 + 索引，不做校验（校验在 bump_version.py，agent 写 yaml 时跑）。
yaml 不存在 / 损坏 → 静默回退到空数据，UI fallback 到链接占位。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml

from ..paths import REPO_ROOT

YAML_PATH = REPO_ROOT / "release_notes.yaml"


@dataclass
class ReleaseNotesEntry:
    """单条变更：一个 (kind, summary, pr_refs, detail) 元组。"""
    kind: str                                 # added/changed/improved/fixed/removed/deprecated/security
    summary: str                              # one-line user-facing
    pr_refs: list[int] = field(default_factory=list)
    detail: Optional[str] = None              # optional markdown


@dataclass
class ReleaseNotesResult:
    """单个版本的 release notes。found=False 时其它字段空，UI 退化到 CHANGELOG 链接。"""
    tag: str                                  # caller 传入的 tag（v 前缀保留）
    found: bool
    date: Optional[str] = None                # ISO YYYY-MM-DD
    summary: Optional[str] = None             # block-level 一句话总览
    entries: list[ReleaseNotesEntry] = field(default_factory=list)


def _normalize_tag(tag: str) -> str:
    """`v0.6.0` → `0.6.0`。yaml 里 version 字段不带 v 前缀。"""
    return tag.lstrip("vV").strip()


def _load_all() -> list[dict]:
    """读整个 yaml 返回 versions list；不存在 / 损坏 → []。"""
    if not YAML_PATH.exists():
        return []
    try:
        data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(data, list):
        return []
    return data


def parse(tag: str) -> ReleaseNotesResult:
    """按 tag 查 release notes。yaml / tag 缺失 / 损坏 → found=False。"""
    norm = _normalize_tag(tag)
    for block in _load_all():
        if not isinstance(block, dict):
            continue
        version = block.get("version")
        if not isinstance(version, str) or _normalize_tag(version) != norm:
            continue
        entries_raw = block.get("entries") or []
        entries: list[ReleaseNotesEntry] = []
        for e in entries_raw:
            if not isinstance(e, dict):
                continue
            kind = e.get("kind")
            summary = e.get("summary")
            if not isinstance(kind, str) or not isinstance(summary, str) or not summary.strip():
                continue
            pr_refs = e.get("pr_refs") or []
            if not isinstance(pr_refs, list):
                pr_refs = []
            pr_refs = [p for p in pr_refs if isinstance(p, int) and p > 0]
            detail = e.get("detail")
            entries.append(ReleaseNotesEntry(
                kind=kind,
                summary=summary.strip(),
                pr_refs=pr_refs,
                detail=detail if isinstance(detail, str) else None,
            ))
        date = block.get("date") if isinstance(block.get("date"), str) else None
        block_summary = block.get("summary") if isinstance(block.get("summary"), str) else None
        return ReleaseNotesResult(
            tag=tag,
            found=bool(entries),
            date=date,
            summary=block_summary,
            entries=entries,
        )
    return ReleaseNotesResult(tag=tag, found=False)
