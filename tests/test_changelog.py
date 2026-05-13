"""CHANGELOG.md 解析（ADR 0002 / chunk 2）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.services import changelog


@pytest.fixture
def fake_changelog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """所有 parse 调用读 tmp_path/CHANGELOG.md。"""
    p = tmp_path / "CHANGELOG.md"
    monkeypatch.setattr(changelog, "CHANGELOG_PATH", p)
    return p


def test_parse_normal_release(fake_changelog: Path) -> None:
    """常见 Keep a Changelog 格式：H2 tag + 日期 + H3 sections + 顶层 bullet。"""
    fake_changelog.write_text(
        "# Changelog\n\n"
        "## [0.6.0] — 2026-05-12\n\n"
        "intro paragraph\n\n"
        "### 新增\n\n"
        "- **LLM tagger**（#18）\n"
        "  - sub-bullet that should be ignored\n"
        "- 其它无 bold 的 top-level\n"
        "### 修复\n\n"
        "- **Danbooru 403**\n"
        "\n---\n\n"
        "## [0.5.2] — 2026-05-11\n",
        encoding="utf-8",
    )
    r = changelog.parse("v0.6.0")
    assert r.found is True
    assert r.tag == "v0.6.0"
    assert r.date == "2026-05-12"
    titles = [s.title for s in r.sections]
    assert titles == ["新增", "修复"]
    new_items = r.sections[0].items
    assert new_items == ["LLM tagger（#18）", "其它无 bold 的 top-level"]
    fix_items = r.sections[1].items
    assert fix_items == ["Danbooru 403"]


def test_parse_tag_without_v_prefix(fake_changelog: Path) -> None:
    """`0.6.0` 和 `v0.6.0` 都能匹配到 `## [0.6.0]`。"""
    fake_changelog.write_text(
        "## [0.6.0] — 2026-05-12\n### 新增\n- A\n",
        encoding="utf-8",
    )
    assert changelog.parse("0.6.0").found is True
    assert changelog.parse("v0.6.0").found is True
    assert changelog.parse("V0.6.0").found is True


def test_parse_missing_tag(fake_changelog: Path) -> None:
    """CHANGELOG 里没这条 tag → found=False，不抛错。"""
    fake_changelog.write_text(
        "## [0.5.0] — 2026-01-01\n### 新增\n- A\n",
        encoding="utf-8",
    )
    r = changelog.parse("v9.9.9")
    assert r.found is False
    assert r.sections == []


def test_parse_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CHANGELOG.md 不存在 → found=False。"""
    monkeypatch.setattr(changelog, "CHANGELOG_PATH", tmp_path / "absent.md")
    r = changelog.parse("v0.6.0")
    assert r.found is False
    assert r.sections == []


def test_parse_section_without_bullets_dropped(fake_changelog: Path) -> None:
    """`### 测试` 这种只有段落、没 bullet 的 section 自动剔除。"""
    fake_changelog.write_text(
        "## [0.6.0]\n"
        "### 新增\n"
        "- A\n"
        "### 测试\n"
        "段落描述，没有 bullet\n",
        encoding="utf-8",
    )
    r = changelog.parse("0.6.0")
    assert [s.title for s in r.sections] == ["新增"]


def test_parse_date_optional(fake_changelog: Path) -> None:
    """`## [0.6.0]`（没日期）也能解析。"""
    fake_changelog.write_text(
        "## [0.6.0]\n### 新增\n- A\n",
        encoding="utf-8",
    )
    r = changelog.parse("0.6.0")
    assert r.found is True
    assert r.date is None


def test_parse_stops_at_next_version(fake_changelog: Path) -> None:
    """下一个 `## [` 截断，不会越界把 0.5.x 的内容算进 0.6.x。"""
    fake_changelog.write_text(
        "## [0.6.0]\n### 新增\n- A60\n"
        "## [0.5.0]\n### 新增\n- A50\n",
        encoding="utf-8",
    )
    r = changelog.parse("0.6.0")
    assert r.sections[0].items == ["A60"]
    r2 = changelog.parse("0.5.0")
    assert r2.sections[0].items == ["A50"]


def test_parse_stops_at_separator(fake_changelog: Path) -> None:
    """`---` 分隔线也截断（CHANGELOG 模板里 preamble 和第一个版本之间用 ---）。"""
    fake_changelog.write_text(
        "## [0.6.0]\n### 新增\n- A\n\n---\n# 其它内容\n- B\n",
        encoding="utf-8",
    )
    r = changelog.parse("0.6.0")
    assert r.sections[0].items == ["A"]


def test_parse_real_changelog_smoke() -> None:
    """跑真实 repo CHANGELOG.md 一遍 — 应能拿到 0.6.0 数据，至少几个 section。"""
    r = changelog.parse("v0.6.0")
    # 不强断言 section 数量（CHANGELOG 会变），只验证基本契约
    if r.found:
        assert r.date == "2026-05-12"
        assert all(s.items for s in r.sections), "空 section 应被剔除"
        all_items = [item for s in r.sections for item in s.items]
        assert any("LLM tagger" in s for s in all_items), "应能匹配到 'LLM tagger' 顶层 bullet"
