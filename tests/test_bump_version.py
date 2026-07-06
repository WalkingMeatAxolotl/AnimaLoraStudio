"""tools/bump_version.py（ADR 0013）：release post 校验 + CHANGELOG 派生 + 版本号同步。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import bump_version as bv  # noqa: E402


def _post(version: str, date: str, *, title: str | None = None, body: str = "正文",
          filename: str | None = None) -> "bv.ReleasePost":
    return bv.ReleasePost(
        version=version, date=date, title=title if title is not None else f"v{version}",
        body=body, filename=filename or f"{date}-v{version}.md",
    )


def _write_release(d: Path, version: str, date: str, body: str = "正文") -> None:
    (d / f"{date}-v{version}.md").write_text(
        f'---\ndate: {date}\ntag: release\ntitle: v{version}\nversion: "{version}"\n---\n{body}\n',
        encoding="utf-8",
    )


# ---- validate -------------------------------------------------------------

def test_validate_ok() -> None:
    assert not bv.validate([_post("0.2.0", "2026-02-01")]).has_errors


def test_validate_empty_is_error() -> None:
    assert bv.validate([]).has_errors


def test_validate_bad_semver() -> None:
    r = bv.validate([_post("not-semver", "2026-01-01")])
    assert r.has_errors and any("semver" in i.message for i in r.issues)


def test_validate_bad_date() -> None:
    r = bv.validate([_post("0.1.0", "2026/01/01")])
    assert r.has_errors and any("date" in i.location for i in r.issues)


def test_validate_duplicate_version() -> None:
    r = bv.validate([_post("0.2.0", "2026-02-01"), _post("0.2.0", "2026-01-01", filename="b.md")])
    assert r.has_errors and any("重复版本" in i.message for i in r.issues)


def test_validate_missing_title_is_error() -> None:
    r = bv.validate([_post("0.2.0", "2026-02-01", title="")])
    assert r.has_errors and any("title" in i.location for i in r.issues)


def test_validate_filename_mismatch_is_warn_not_error() -> None:
    r = bv.validate([_post("0.2.0", "2026-02-01", filename="wrong-name.md")])
    assert not r.has_errors
    assert any(i.level == "warn" for i in r.issues)


# ---- load_release_posts ---------------------------------------------------

def test_load_release_posts_filters_and_sorts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bv, "ANNOUNCEMENTS_DIR", tmp_path)
    _write_release(tmp_path, "0.1.0", "2026-01-01")
    _write_release(tmp_path, "0.2.0", "2026-02-01")
    # 非 release tag + README 都应被忽略
    (tmp_path / "2026-03-01-note.md").write_text(
        "---\ndate: 2026-03-01\ntag: notice\ntitle: n\n---\nb\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# 指南\n", encoding="utf-8")
    posts = bv.load_release_posts()
    assert [p.version for p in posts] == ["0.2.0", "0.1.0"]  # semver 降序


# ---- render_changelog -----------------------------------------------------

def test_render_changelog_has_header_and_versions() -> None:
    out = bv.render_changelog([_post("0.2.0", "2026-02-01", body="新版正文")])
    assert out.startswith("# Changelog")
    assert "## v0.2.0 — 2026-02-01" in out
    assert "新版正文" in out


# ---- verify-versions ------------------------------------------------------

def _setup_version_files(tmp: Path, init_v: str, pkg_v: str, lock_v: str, lock_pkg_v: str,
                         monkeypatch: pytest.MonkeyPatch) -> None:
    init = tmp / "__init__.py"; init.write_text(f'__version__ = "{init_v}"\n', encoding="utf-8")
    pkg = tmp / "package.json"; pkg.write_text(json.dumps({"version": pkg_v}), encoding="utf-8")
    lock = tmp / "package-lock.json"
    lock.write_text(json.dumps({"version": lock_v, "packages": {"": {"version": lock_pkg_v}}}), encoding="utf-8")
    monkeypatch.setattr(bv, "STUDIO_INIT_PATH", init)
    monkeypatch.setattr(bv, "PACKAGE_JSON_PATH", pkg)
    monkeypatch.setattr(bv, "PACKAGE_LOCK_PATH", lock)


def test_verify_versions_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_version_files(tmp_path, "0.5.0", "0.5.0", "0.5.0", "0.5.0", monkeypatch)
    assert bv.cmd_verify_versions(argparse.Namespace()) == 0


def test_verify_versions_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_version_files(tmp_path, "0.5.0", "0.5.1", "0.5.0", "0.5.0", monkeypatch)
    assert bv.cmd_verify_versions(argparse.Namespace()) == 1
