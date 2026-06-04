"""preprocess_manifest: 项目级 read-only fallback（写路径见 train_* / 见 test_train_manifest_mutations.py）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.services.preprocess import manifest as pm


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    (tmp_path / "download").mkdir()
    (tmp_path / "preprocess").mkdir()
    return tmp_path


def _write_manifest(project_dir: Path, images: dict) -> None:
    pm.manifest_path(project_dir).write_text(
        json.dumps({"images": images}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def test_load_missing_returns_empty(project_dir: Path) -> None:
    assert pm.load(project_dir) == {"images": {}}


def test_load_corrupted_returns_empty(project_dir: Path) -> None:
    pm.manifest_path(project_dir).write_text("not json")
    assert pm.load(project_dir) == {"images": {}}


def test_load_invalid_shape_returns_empty(project_dir: Path) -> None:
    pm.manifest_path(project_dir).write_text(json.dumps(["array"]))
    assert pm.load(project_dir) == {"images": {}}


# ---------------------------------------------------------------------------
# resolve / resolve_origin / entry_origin / get_entry
# 仅保留 thumb endpoint 仍要用的 read 路径。
# ---------------------------------------------------------------------------


def test_resolve_unknown_name_returns_download(project_dir: Path) -> None:
    p = pm.resolve(project_dir, "foo.jpg")
    assert p == project_dir / "download" / "foo.jpg"


def test_resolve_existing_entry_returns_preprocess(project_dir: Path) -> None:
    _write_manifest(project_dir, {"foo.png": {"origin": "foo.png"}})
    assert pm.resolve(project_dir, "foo.png") == project_dir / "preprocess" / "foo.png"


def test_entry_origin_prefers_origin_then_fallback() -> None:
    assert pm.entry_origin({"origin": "a.jpg"}, "x") == "a.jpg"
    assert pm.entry_origin({}, "fallback.png") == "fallback.png"


def test_get_entry_returns_none_when_missing(project_dir: Path) -> None:
    assert pm.get_entry(project_dir, "nope.png") is None


def test_resolve_origin_returns_derivatives(project_dir: Path) -> None:
    _write_manifest(project_dir, {
        "X_c0.png": {"origin": "X.jpg"},
        "X_c1.png": {"origin": "X.jpg"},
    })
    paths = pm.resolve_origin(project_dir, "X.jpg")
    assert sorted(p.name for p in paths) == ["X_c0.png", "X_c1.png"]
    # 不匹配 → fallback download
    assert pm.resolve_origin(project_dir, "ghost.jpg") == [
        project_dir / "download" / "ghost.jpg"
    ]


def test_resolve_origin_returns_empty_when_only_duplicate_removed(project_dir: Path) -> None:
    _write_manifest(project_dir, {
        "X.png": {"origin": "X.jpg", "kind": "duplicate_removed"},
    })
    assert pm.resolve_origin(project_dir, "X.jpg") == []


def test_is_duplicate_removed_entry() -> None:
    assert pm.is_duplicate_removed_entry({"kind": "duplicate_removed"}) is True
    assert pm.is_duplicate_removed_entry({"kind": "processed"}) is False
    assert pm.is_duplicate_removed_entry({}) is False
    assert pm.is_duplicate_removed_entry(None) is False
