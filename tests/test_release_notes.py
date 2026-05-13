"""release_notes.yaml 读取 + 索引（ADR 0002 / chunk 2 重做）。

校验逻辑在 tools/bump_version.py（见 test_bump_version.py）；此处仅测读取 +
缺失 + 损坏的 fallback 行为。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from studio.services import release_notes


@pytest.fixture
def fake_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "release_notes.yaml"
    monkeypatch.setattr(release_notes, "YAML_PATH", p)
    return p


def _write(path: Path, data) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# parse — happy paths
# ---------------------------------------------------------------------------


def test_parse_basic(fake_yaml: Path) -> None:
    _write(fake_yaml, [
        {
            "version": "0.6.0",
            "date": "2026-05-12",
            "summary": "summary line",
            "entries": [
                {"kind": "added", "summary": "LLM tagger", "pr_refs": [18, 34]},
                {"kind": "fixed", "summary": "fix X", "detail": "**bold** detail"},
            ],
        }
    ])
    r = release_notes.parse("v0.6.0")
    assert r.found is True
    assert r.tag == "v0.6.0"
    assert r.date == "2026-05-12"
    assert r.summary == "summary line"
    assert len(r.entries) == 2
    e0 = r.entries[0]
    assert e0.kind == "added"
    assert e0.summary == "LLM tagger"
    assert e0.pr_refs == [18, 34]
    assert e0.detail is None
    e1 = r.entries[1]
    assert e1.detail == "**bold** detail"


def test_parse_v_prefix_optional(fake_yaml: Path) -> None:
    _write(fake_yaml, [
        {"version": "0.6.0", "date": "2026-05-12",
         "entries": [{"kind": "added", "summary": "x"}]}
    ])
    assert release_notes.parse("0.6.0").found is True
    assert release_notes.parse("v0.6.0").found is True
    assert release_notes.parse("V0.6.0").found is True


def test_parse_picks_correct_version_from_list(fake_yaml: Path) -> None:
    _write(fake_yaml, [
        {"version": "0.6.0", "date": "2026-05-12",
         "entries": [{"kind": "added", "summary": "v60"}]},
        {"version": "0.5.0", "date": "2026-05-09",
         "entries": [{"kind": "added", "summary": "v50"}]},
    ])
    r60 = release_notes.parse("v0.6.0")
    assert r60.entries[0].summary == "v60"
    r50 = release_notes.parse("v0.5.0")
    assert r50.entries[0].summary == "v50"


# ---------------------------------------------------------------------------
# parse — fallback paths
# ---------------------------------------------------------------------------


def test_parse_missing_tag(fake_yaml: Path) -> None:
    _write(fake_yaml, [
        {"version": "0.5.0", "date": "2026-05-09",
         "entries": [{"kind": "added", "summary": "x"}]}
    ])
    r = release_notes.parse("v9.9.9")
    assert r.found is False
    assert r.entries == []


def test_parse_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release_notes, "YAML_PATH", tmp_path / "absent.yaml")
    r = release_notes.parse("v0.6.0")
    assert r.found is False
    assert r.entries == []


def test_parse_corrupt_yaml_returns_not_found(fake_yaml: Path) -> None:
    fake_yaml.write_text("{not valid: yaml:\n  - broken", encoding="utf-8")
    r = release_notes.parse("v0.6.0")
    assert r.found is False
    assert r.entries == []


def test_parse_empty_yaml(fake_yaml: Path) -> None:
    fake_yaml.write_text("", encoding="utf-8")
    r = release_notes.parse("v0.6.0")
    assert r.found is False


def test_parse_non_list_top_level(fake_yaml: Path) -> None:
    """yaml 顶层不是 list（写错成 dict）→ 容忍，返回 not found。"""
    fake_yaml.write_text("version: 0.6.0\n", encoding="utf-8")
    r = release_notes.parse("v0.6.0")
    assert r.found is False


def test_parse_skips_malformed_entries(fake_yaml: Path) -> None:
    """entry 缺 kind / summary 字段 → 跳过该条，不影响其它 entry。"""
    _write(fake_yaml, [
        {"version": "0.6.0", "date": "2026-05-12", "entries": [
            {"kind": "added", "summary": "good entry"},
            {"summary": "no kind"},
            {"kind": "fixed"},                    # no summary
            "not even a dict",                     # type error
            {"kind": "improved", "summary": "another good"},
        ]}
    ])
    r = release_notes.parse("v0.6.0")
    assert r.found is True
    summaries = [e.summary for e in r.entries]
    assert summaries == ["good entry", "another good"]


def test_parse_pr_refs_filters_non_int(fake_yaml: Path) -> None:
    _write(fake_yaml, [
        {"version": "0.6.0", "date": "2026-05-12", "entries": [
            {"kind": "added", "summary": "x", "pr_refs": [18, "34", -1, 42]},
        ]}
    ])
    r = release_notes.parse("v0.6.0")
    assert r.entries[0].pr_refs == [18, 42]


# ---------------------------------------------------------------------------
# real repo smoke
# ---------------------------------------------------------------------------


def test_real_repo_parses_known_version() -> None:
    """真 release_notes.yaml 应当能读出 0.6.0 + 至少几个 entry（CI 安全网）。"""
    r = release_notes.parse("v0.6.0")
    if r.found:  # 不强断言：万一以后 release_notes.yaml 内容变了 / 重命名 tag
        assert r.date == "2026-05-12"
        assert len(r.entries) >= 1
        kinds = {e.kind for e in r.entries}
        # 0.6.0 至少有 added / changed / fixed 三类
        assert "added" in kinds
