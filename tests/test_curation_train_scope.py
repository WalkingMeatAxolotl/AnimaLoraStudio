"""ADR 0010 — curation `copy_download_to_train` (PR-2 step C)。

跟老 `copy_to_train` 的差异（独立单测在 test_curation.py 不重叠）：
- 取消 preprocess 派生分支（bytes 始终从 download/{name}）
- 写 train manifest entry，key = "{folder}/{name}"，origin = name
- caption 仍从 download/{stem}.* 复制
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db
from studio.services.dataset import curation
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
    return {"db": dbfile, "p": p, "v": v}


def _pdir(env) -> Path:
    return projects.project_dir(env["p"]["id"], env["p"]["slug"])


def _dl(env, name: str, blob: bytes = b"img") -> Path:
    f = _pdir(env) / "download" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(blob)
    return f


def _train(env, folder: str = "1_data") -> Path:
    return _pdir(env) / "versions" / env["v"]["label"] / "train" / folder


# ---------------------------------------------------------------------------
# 基础：复制 + manifest entry
# ---------------------------------------------------------------------------


def test_copy_download_to_train_writes_manifest_entry(env) -> None:
    _dl(env, "X.jpg", blob=b"orig" * 10)
    with db.connection_for(env["db"]) as conn:
        result = curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["X.jpg"], dest_folder="1_data",
        )
    assert result == {"copied": ["X.jpg"], "skipped": [], "missing": []}
    # bytes 复制
    assert (_train(env) / "X.jpg").read_bytes() == b"orig" * 10
    # manifest entry key = "1_data/X.jpg"，origin = "X.jpg"
    entry = preprocess_manifest.train_get_entry(
        _pdir(env), env["v"]["label"], "1_data/X.jpg"
    )
    assert entry is not None
    assert entry["origin"] == "X.jpg"
    assert entry["size"] == 40


def test_copy_download_to_train_copies_caption(env) -> None:
    _dl(env, "Y.jpg")
    (_pdir(env) / "download" / "Y.txt").write_text("a tag, b tag")
    (_pdir(env) / "download" / "Y.json").write_text('{"k":1}')
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["Y.jpg"], dest_folder="1_data",
        )
    assert (_train(env) / "Y.txt").read_text() == "a tag, b tag"
    assert (_train(env) / "Y.json").read_text() == '{"k":1}'


def test_copy_download_to_train_skips_existing(env) -> None:
    _dl(env, "Z.jpg")
    dst = _train(env)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "Z.jpg").write_bytes(b"existing")

    with db.connection_for(env["db"]) as conn:
        result = curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["Z.jpg"], dest_folder="1_data",
        )
    assert result == {"copied": [], "skipped": ["Z.jpg"], "missing": []}
    # 内容没被覆盖
    assert (dst / "Z.jpg").read_bytes() == b"existing"


def test_copy_download_to_train_missing(env) -> None:
    with db.connection_for(env["db"]) as conn:
        result = curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["ghost.jpg"], dest_folder="1_data",
        )
    assert result == {"copied": [], "skipped": [], "missing": ["ghost.jpg"]}


def test_copy_download_to_train_multiple_folders_indep_entries(env) -> None:
    """同一张图复制到两个 folder → manifest 两个独立 entry。"""
    _dl(env, "shared.jpg", blob=b"s")
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["shared.jpg"], dest_folder="1_data",
        )
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["shared.jpg"], dest_folder="5_extra",
        )
    m = preprocess_manifest.train_load(_pdir(env), env["v"]["label"])
    assert "1_data/shared.jpg" in m["images"]
    assert "5_extra/shared.jpg" in m["images"]
    # 两个 entry origin 都指向同一 download
    assert m["images"]["1_data/shared.jpg"]["origin"] == "shared.jpg"
    assert m["images"]["5_extra/shared.jpg"]["origin"] == "shared.jpg"


def test_copy_download_to_train_no_preprocess_branch(env) -> None:
    """ADR 0010：copy_download_to_train 不消费 preprocess 派生，即使老 manifest
    标 X.jpg 已经处理过（preprocess/X.png），新函数仍只从 download 复制原图。"""
    import json
    _dl(env, "X.jpg", blob=b"raw")
    # 模拟老项目残留 preprocess/X.png + 项目级 manifest entry
    pre = _pdir(env) / "preprocess"
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "X.png").write_bytes(b"upscaled-residue")
    preprocess_manifest.manifest_path(_pdir(env)).write_text(
        json.dumps({"images": {"X.png": {"origin": "X.jpg"}}}),
        encoding="utf-8",
    )

    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["X.jpg"], dest_folder="1_data",
        )

    # train 收到的是 download bytes（"raw"），不是 preprocess 派生
    assert (_train(env) / "X.jpg").read_bytes() == b"raw"
    # 新 manifest entry key 是 "1_data/X.jpg"，origin = name 本身
    entry = preprocess_manifest.train_get_entry(
        _pdir(env), env["v"]["label"], "1_data/X.jpg"
    )
    assert entry["origin"] == "X.jpg"


def test_copy_download_to_train_invalid_folder_rejected(env) -> None:
    _dl(env, "X.jpg")
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(curation.CurationError):
            curation.copy_download_to_train(
                conn, env["p"]["id"], env["v"]["id"],
                files=["X.jpg"], dest_folder="../escape",
            )


def test_copy_download_to_train_invalid_filename_rejected(env) -> None:
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(curation.CurationError):
            curation.copy_download_to_train(
                conn, env["p"]["id"], env["v"]["id"],
                files=["../../etc/passwd"], dest_folder="1_data",
            )


# ---------------------------------------------------------------------------
# ADR 0010 fixup（2026-06-04）：list_train 按 origin 去重 + 仍含
# duplicate_removed（筛选时间 < 预处理，跟预处理状态解耦）
# ---------------------------------------------------------------------------


def test_list_train_dedupes_by_origin_on_fan_out(env) -> None:
    """multi-crop 派生 X_c0 / X_c1 同 origin=X.jpg → list_train 只一条。"""
    train_sub = _train(env, "1_data")
    train_sub.mkdir(parents=True, exist_ok=True)
    # 模拟 fan-out 后物理状态 + manifest
    (train_sub / "X_c0.png").write_bytes(b"c0")
    (train_sub / "X_c1.png").write_bytes(b"c1")
    preprocess_manifest.train_replace_with_crops(
        _pdir(env), env["v"]["label"],
        source_name="1_data/X.jpg",
        outputs=[
            {"name": "1_data/X_c0.png", "origin": "X.jpg", "mtime": 1, "size": 10},
            {"name": "1_data/X_c1.png", "origin": "X.jpg", "mtime": 1, "size": 10},
        ],
    )

    with db.connection_for(env["db"]) as conn:
        view = curation.curation_view(conn, env["p"]["id"], env["v"]["id"])
    assert [e["name"] for e in view["right"]["1_data"]] == ["X.jpg"]
    # name = origin（统一到 download scope）
    assert view["right"]["1_data"][0]["origin"] == "X.jpg"


def test_list_train_shows_duplicate_removed_in_curation(env) -> None:
    """duplicate_removed 标记的图物理还在 → Curation 右侧仍显示（筛选时间
    在预处理之前，时间语义上独立于 dedupe 决定）。"""
    train_sub = _train(env, "1_data")
    train_sub.mkdir(parents=True, exist_ok=True)
    (train_sub / "Y.jpg").write_bytes(b"y")
    preprocess_manifest.train_add_processed(
        _pdir(env), env["v"]["label"], "1_data/Y.jpg", {"origin": "Y.jpg"},
    )
    preprocess_manifest.train_mark_duplicate_removed(
        _pdir(env), env["v"]["label"], ["1_data/Y.jpg"],
    )

    with db.connection_for(env["db"]) as conn:
        view = curation.curation_view(conn, env["p"]["id"], env["v"]["id"])
    # duplicate_removed entry 物理仍在 → Curation 右侧仍含 Y.jpg
    assert [e["name"] for e in view["right"]["1_data"]] == ["Y.jpg"]


def test_remove_from_train_deletes_all_fan_out_derivatives(env) -> None:
    """按 origin 删 → 所有派生物理文件 + manifest entries 一起清。"""
    train_sub = _train(env, "1_data")
    train_sub.mkdir(parents=True, exist_ok=True)
    (train_sub / "X_c0.png").write_bytes(b"c0")
    (train_sub / "X_c1.png").write_bytes(b"c1")
    preprocess_manifest.train_replace_with_crops(
        _pdir(env), env["v"]["label"],
        source_name="1_data/X.jpg",
        outputs=[
            {"name": "1_data/X_c0.png", "origin": "X.jpg", "mtime": 1, "size": 10},
            {"name": "1_data/X_c1.png", "origin": "X.jpg", "mtime": 1, "size": 10},
        ],
    )

    with db.connection_for(env["db"]) as conn:
        res = curation.remove_from_train(
            conn, env["p"]["id"], env["v"]["id"], "1_data", ["X.jpg"],
        )
    assert res["removed"] == ["X.jpg"]
    # 两个派生物理文件都没了
    assert not (train_sub / "X_c0.png").exists()
    assert not (train_sub / "X_c1.png").exists()
    # manifest entry 都清掉
    m = preprocess_manifest.train_load(_pdir(env), env["v"]["label"])
    assert "1_data/X_c0.png" not in m["images"]
    assert "1_data/X_c1.png" not in m["images"]
