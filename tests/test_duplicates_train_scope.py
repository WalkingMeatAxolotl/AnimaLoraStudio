"""ADR 0010 — duplicates train-scope API（PR-2 step E）。

覆盖 `_resolve_train_sources`、`apply_train_duplicate_removals` 行为。
`scan_train_duplicates` 主体跟 `scan_project_duplicates` 共享，只是 sources
来源不同 — sources 解析正确即可信任主流程（无需重测 hash/compare/group）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from studio import db
from studio.domain.errors import InvalidPathError, NotFoundError
from studio.services.preprocess import duplicates as duplicate_finder
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.projects import projects, versions


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    pdir = projects.project_dir(p["id"], p["slug"])
    sub = pdir / "versions" / "v1" / "train" / "1_data"
    sub.mkdir(parents=True, exist_ok=True)
    return {"db": dbfile, "p": p, "v": v, "pdir": pdir, "sub": sub}


def _png(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path, "PNG")


# ---------------------------------------------------------------------------
# _resolve_train_sources
# ---------------------------------------------------------------------------


def test_resolve_train_sources_lists_subfolder_images(env) -> None:
    _png(env["sub"] / "X.png")
    _png(env["sub"] / "Y.png")
    with db.connection_for(env["db"]) as conn:
        out = duplicate_finder._resolve_train_sources(
            conn, env["p"]["id"], env["v"]["id"], env["pdir"],
        )
    names = [n for n, _ in out]
    assert names == ["1_data/X.png", "1_data/Y.png"]


def test_resolve_train_sources_skips_duplicate_removed(env) -> None:
    _png(env["sub"] / "X.png")
    _png(env["sub"] / "Y.png")
    preprocess_manifest.train_mark_duplicate_removed(
        env["pdir"], env["v"]["label"], ["1_data/Y.png"],
    )
    with db.connection_for(env["db"]) as conn:
        out = duplicate_finder._resolve_train_sources(
            conn, env["p"]["id"], env["v"]["id"], env["pdir"],
        )
    names = [n for n, _ in out]
    assert names == ["1_data/X.png"]


def test_resolve_train_sources_skips_non_image(env) -> None:
    _png(env["sub"] / "X.png")
    (env["sub"] / "X.txt").write_text("caption")
    with db.connection_for(env["db"]) as conn:
        out = duplicate_finder._resolve_train_sources(
            conn, env["p"]["id"], env["v"]["id"], env["pdir"],
        )
    names = [n for n, _ in out]
    assert names == ["1_data/X.png"]


def test_resolve_train_sources_empty_when_no_train(env) -> None:
    import shutil
    shutil.rmtree(env["pdir"] / "versions")
    with db.connection_for(env["db"]) as conn:
        out = duplicate_finder._resolve_train_sources(
            conn, env["p"]["id"], env["v"]["id"], env["pdir"],
        )
    assert out == []


def test_resolve_train_sources_mismatched_version_raises(env) -> None:
    """version_id 跟 project_id 不一致 → NotFoundError(version.not_found)。"""
    # 另开一个 project + version
    with db.connection_for(env["db"]) as conn:
        other_p = projects.create_project(conn, title="Other")
        other_v = versions.create_version(
            conn, project_id=other_p["id"], label="v1"
        )
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(NotFoundError) as exc:
            duplicate_finder._resolve_train_sources(
                conn, env["p"]["id"], other_v["id"], env["pdir"],
            )
        assert exc.value.code == "version.not_found"


# ---------------------------------------------------------------------------
# apply_train_duplicate_removals
# ---------------------------------------------------------------------------


def test_apply_train_duplicate_marks_manifest(env) -> None:
    _png(env["sub"] / "A.png")
    _png(env["sub"] / "B.png")
    with db.connection_for(env["db"]) as conn:
        result = duplicate_finder.apply_train_duplicate_removals(
            conn, env["p"]["id"], env["v"]["id"],
            names=["1_data/A.png", "1_data/B.png"],
        )
    assert sorted(result["removed"]) == ["1_data/A.png", "1_data/B.png"]
    # 物理文件已删（tombstone 只在 manifest）
    assert not (env["sub"] / "A.png").exists()
    assert not (env["sub"] / "B.png").exists()
    # manifest 标记
    entry = preprocess_manifest.train_get_entry(
        env["pdir"], env["v"]["label"], "1_data/A.png"
    )
    assert entry["kind"] == preprocess_manifest.DUPLICATE_REMOVED_KIND


def test_apply_train_duplicate_rejects_invalid_rel_name(env) -> None:
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(InvalidPathError) as exc:
            duplicate_finder.apply_train_duplicate_removals(
                conn, env["p"]["id"], env["v"]["id"],
                names=["../etc/passwd"],
            )
        assert exc.value.code == "path.invalid"


def test_apply_train_duplicate_rejects_flat_name(env) -> None:
    """rel path 必须含 folder 前缀（两段格式）。"""
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(InvalidPathError) as exc:
            duplicate_finder.apply_train_duplicate_removals(
                conn, env["p"]["id"], env["v"]["id"],
                names=["X.png"],  # 平铺，缺 folder
            )
        assert exc.value.code == "path.invalid"


def test_apply_train_duplicate_mismatched_version_raises(env) -> None:
    with db.connection_for(env["db"]) as conn:
        other_p = projects.create_project(conn, title="Other")
        other_v = versions.create_version(
            conn, project_id=other_p["id"], label="v1"
        )
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(NotFoundError) as exc:
            duplicate_finder.apply_train_duplicate_removals(
                conn, env["p"]["id"], other_v["id"],
                names=["1_data/A.png"],
            )
        assert exc.value.code == "version.not_found"


