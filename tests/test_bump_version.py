"""tools/bump_version.py：yaml schema 校验 + 版本号同步 + CHANGELOG 派生。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import bump_version as bv  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_block(version: str, date: str, kind: str = "added",
                summary: str = "summary text", pr_refs=None) -> dict:
    entry = {"kind": kind, "summary": summary}
    if pr_refs is not None:
        entry["pr_refs"] = pr_refs
    return {"version": version, "date": date, "entries": [entry]}


# ---------------------------------------------------------------------------
# validate — happy / error paths
# ---------------------------------------------------------------------------


def test_validate_minimal_ok() -> None:
    r = bv.validate([_make_block("0.1.0", "2026-01-01")])
    assert not r.has_errors


def test_validate_empty_list_is_error() -> None:
    r = bv.validate([])
    assert r.has_errors
    assert any("至少要有一个" in i.message for i in r.issues)


def test_validate_bad_semver_rejected() -> None:
    r = bv.validate([_make_block("not-a-semver", "2026-01-01")])
    assert r.has_errors
    assert any("无效 semver" in i.message for i in r.issues)


def test_validate_duplicate_version_rejected() -> None:
    r = bv.validate([
        _make_block("0.2.0", "2026-02-01"),
        _make_block("0.2.0", "2026-01-01"),
    ])
    assert r.has_errors
    assert any("重复版本" in i.message for i in r.issues)


def test_validate_wrong_order_rejected() -> None:
    """yaml 顺序应当 latest 在 top；倒序应当被拒。"""
    r = bv.validate([
        _make_block("0.1.0", "2026-01-01"),   # 老
        _make_block("0.2.0", "2026-02-01"),   # 新
    ])
    assert r.has_errors
    assert any("顺序错误" in i.message for i in r.issues)


def test_validate_bad_date_rejected() -> None:
    r = bv.validate([_make_block("0.1.0", "2026/01/01")])
    assert r.has_errors


def test_validate_unknown_kind_rejected() -> None:
    r = bv.validate([_make_block("0.1.0", "2026-01-01", kind="refactor")])
    assert r.has_errors


def test_validate_summary_too_long_rejected() -> None:
    r = bv.validate([_make_block("0.1.0", "2026-01-01", summary="x" * 81)])
    assert r.has_errors


def test_validate_summary_markdown_rejected() -> None:
    r = bv.validate([_make_block("0.1.0", "2026-01-01", summary="**bold** stuff")])
    assert r.has_errors


def test_validate_summary_underscore_allowed() -> None:
    """`__init__.py` 之类 `_` 太常见，不再被当 markdown bold 拒（曾经是 bug）。"""
    r = bv.validate([_make_block("0.1.0", "2026-01-01", summary="改 studio/__init__.py 的 __version__")])
    assert not r.has_errors


def test_validate_summary_too_short_warns() -> None:
    r = bv.validate([_make_block("0.1.0", "2026-01-01", summary="短")])
    assert not r.has_errors  # warn 不阻塞
    assert any(i.level == "warn" and "summary 太短" in i.message for i in r.issues)


def test_validate_pr_refs_non_int_rejected() -> None:
    r = bv.validate([_make_block("0.1.0", "2026-01-01", pr_refs=["18"])])
    assert r.has_errors


def test_validate_pr_refs_negative_rejected() -> None:
    r = bv.validate([_make_block("0.1.0", "2026-01-01", pr_refs=[-1])])
    assert r.has_errors


def test_validate_entries_empty_rejected() -> None:
    r = bv.validate([{"version": "0.1.0", "date": "2026-01-01", "entries": []}])
    assert r.has_errors


def test_validate_repeated_pr_ref_warns() -> None:
    """同 PR 在 ≥ 4 条 entry 出现 → warn（可能拆得太细）。"""
    versions = [{
        "version": "0.1.0", "date": "2026-01-01",
        "entries": [{"kind": "fixed", "summary": f"fix item {i} description text", "pr_refs": [42]}
                    for i in range(4)],
    }]
    r = bv.validate(versions)
    assert not r.has_errors
    assert any(i.level == "warn" and "PR #42" in i.message for i in r.issues)


# ---------------------------------------------------------------------------
# render_changelog
# ---------------------------------------------------------------------------


def test_render_changelog_basic_structure() -> None:
    versions = [_make_block("0.2.0", "2026-02-01"), _make_block("0.1.0", "2026-01-01")]
    out = bv.render_changelog(versions)
    assert out.startswith("# Changelog")
    assert "## [0.2.0] — 2026-02-01" in out
    assert "## [0.1.0] — 2026-01-01" in out
    # block summary 缺省时不应渲染额外空段
    # 顶部派生注释
    assert "tools/bump_version.py" in out


def test_render_changelog_kind_section_translation() -> None:
    versions = [{
        "version": "0.2.0", "date": "2026-02-01",
        "entries": [
            {"kind": "added", "summary": "feature A added long enough"},
            {"kind": "fixed", "summary": "bug X fixed with detail"},
            {"kind": "improved", "summary": "X 更快 with enough chars"},
        ],
    }]
    out = bv.render_changelog(versions)
    assert "### 新增" in out
    assert "### 修复" in out
    assert "### 改进" in out


def test_render_changelog_kind_ordering() -> None:
    """添加 / 变更 / 改进 / 修复 应按固定顺序排，无论 yaml 里 entries 顺序。"""
    versions = [{
        "version": "0.2.0", "date": "2026-02-01",
        "entries": [
            {"kind": "fixed",    "summary": "f1 fix description text"},
            {"kind": "added",    "summary": "a1 add description text"},
            {"kind": "improved", "summary": "i1 improvement description"},
        ],
    }]
    out = bv.render_changelog(versions)
    # 新增 在 改进 之前；改进 在 修复 之前
    pos_add = out.index("### 新增")
    pos_imp = out.index("### 改进")
    pos_fix = out.index("### 修复")
    assert pos_add < pos_imp < pos_fix


def test_render_changelog_detail_indented() -> None:
    versions = [{
        "version": "0.2.0", "date": "2026-02-01",
        "entries": [{
            "kind": "added",
            "summary": "feature X added description",
            "detail": "Line 1\nLine 2\n  with leading spaces",
        }],
    }]
    out = bv.render_changelog(versions)
    assert "- **feature X added description**" in out
    # detail 应该 indent 2 空格
    assert "  Line 1" in out
    assert "  Line 2" in out


# ---------------------------------------------------------------------------
# bump --version + studio/__init__.py + package.json 同步
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """所有 bump_version 路径 monkeypatch 到 tmp_path 隔离副本。"""
    yaml_path = tmp_path / "release_notes.yaml"
    init_path = tmp_path / "studio" / "__init__.py"
    pkg_path = tmp_path / "studio" / "web" / "package.json"
    changelog_path = tmp_path / "CHANGELOG.md"
    init_path.parent.mkdir(parents=True)
    pkg_path.parent.mkdir(parents=True)

    yaml_path.write_text(
        yaml.safe_dump(
            [_make_block("0.7.0", "2026-06-01"),
             _make_block("0.6.0", "2026-05-12")],
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    init_path.write_text(
        '"""docstring."""\n\n__version__ = "0.6.0"\n',
        encoding="utf-8",
    )
    pkg_path.write_text(
        json.dumps({"name": "studio-web", "version": "0.6.0", "private": True}, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(bv, "YAML_PATH", yaml_path)
    monkeypatch.setattr(bv, "STUDIO_INIT_PATH", init_path)
    monkeypatch.setattr(bv, "PACKAGE_JSON_PATH", pkg_path)
    monkeypatch.setattr(bv, "CHANGELOG_PATH", changelog_path)
    monkeypatch.setattr(bv, "REPO_ROOT", tmp_path)

    return {
        "yaml": yaml_path, "init": init_path, "pkg": pkg_path, "changelog": changelog_path,
    }


def test_bump_syncs_init_and_package_json(
    isolated_repo: dict[str, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    rc = bv.main(["bump"])
    assert rc == 0
    assert '__version__ = "0.7.0"' in isolated_repo["init"].read_text(encoding="utf-8")
    pkg_data = json.loads(isolated_repo["pkg"].read_text(encoding="utf-8"))
    assert pkg_data["version"] == "0.7.0"
    # name 字段应保留
    assert pkg_data["name"] == "studio-web"
    # CHANGELOG.md 派生出来
    assert isolated_repo["changelog"].exists()
    assert "## [0.7.0]" in isolated_repo["changelog"].read_text(encoding="utf-8")


def test_bump_version_mismatch_rejected(
    isolated_repo: dict[str, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """--version 0.8.0 但 yaml top 是 0.7.0 → 拒绝（yaml 是 source of truth）。"""
    rc = bv.main(["bump", "--version", "0.8.0"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "yaml top version=0.7.0" in err
    # 版本号文件不动
    assert '"0.6.0"' in isolated_repo["init"].read_text(encoding="utf-8")


def test_bump_blocked_by_validate_error(
    isolated_repo: dict[str, Path],
) -> None:
    """yaml 含 schema error → bump 拒绝执行，不写任何文件。"""
    bad_yaml = yaml.safe_dump(
        [_make_block("not-semver", "2026-06-01")],
        allow_unicode=True,
    )
    isolated_repo["yaml"].write_text(bad_yaml, encoding="utf-8")
    rc = bv.main(["bump"])
    assert rc == 1
    assert '"0.6.0"' in isolated_repo["init"].read_text(encoding="utf-8")
    assert not isolated_repo["changelog"].exists()


def test_render_changelog_subcommand_only_writes_changelog(
    isolated_repo: dict[str, Path],
) -> None:
    """render-changelog 不动 __init__.py / package.json，仅重写 CHANGELOG.md。"""
    rc = bv.main(["render-changelog"])
    assert rc == 0
    # 版本号文件不动
    assert '"0.6.0"' in isolated_repo["init"].read_text(encoding="utf-8")
    assert isolated_repo["changelog"].exists()


# ---------------------------------------------------------------------------
# 真 repo CHANGELOG smoke
# ---------------------------------------------------------------------------


def test_real_repo_release_notes_validates() -> None:
    """真 release_notes.yaml 应当 validate ok（CI 安全网）。"""
    versions = bv.load_yaml()
    r = bv.validate(versions)
    assert not r.has_errors, f"release_notes.yaml validate failed: {[i.message for i in r.issues if i.level=='error']}"
