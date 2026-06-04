"""ADR 0010 — train-scope core API（PR-2 step B）。

覆盖 `list_train_images / summary_train / resolve_targets_train / start_job_train
/ start_crop_job_train / list_crop_workspace_train /
list_duplicate_removed_workspace_train / restore_products_train`。

train/ 是 LoRA repeat folder 结构（`train/{N_label}/{image}`）；manifest entry
key 和这些函数的 `name` 参数都用 POSIX 相对路径（`"1_data/X.png"`）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from studio import db
from studio.services.preprocess import core as preprocess
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.projects import jobs as project_jobs, projects, versions


DEFAULT_FOLDER = "1_data"


def _rel(name: str, folder: str = DEFAULT_FOLDER) -> str:
    return f"{folder}/{name}"


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """isolated 项目 + 自动创建 v1 version + `versions/v1/train/1_data/` sub-folder。"""
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="PP")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    sub = preprocess.version_train_dir(p, "v1") / DEFAULT_FOLDER
    sub.mkdir(parents=True, exist_ok=True)
    return {"db": dbfile, "project": p, "version": v, "sub": sub}


def _write_png(path: Path, size: tuple[int, int] = (10, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="red").save(path, "PNG")


def _download_dir(p: dict) -> Path:
    d, _ = preprocess.project_paths(p)
    return d


# ---------------------------------------------------------------------------
# list_train_images
# ---------------------------------------------------------------------------


def test_list_train_images_empty_when_no_train(isolated) -> None:
    # 删 fixture 预建的 sub-folder
    import shutil
    shutil.rmtree(isolated["sub"])
    items = preprocess.list_train_images(isolated["project"], "v1")
    assert items == []


def test_list_train_images_basic_entry(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "X.png", (100, 80))
    download = _download_dir(p)
    download.mkdir(parents=True, exist_ok=True)
    (download / "X.jpg").write_bytes(b"orig")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("X.png"), {"origin": "X.jpg"}
    )

    items = preprocess.list_train_images(p, "v1")
    assert len(items) == 1
    item = items[0]
    assert item["name"] == _rel("X.png")
    assert item["origin"] == "X.jpg"
    assert item["source"] == "X.jpg"
    assert item["w"] == 100 and item["h"] == 80
    assert item["orphan"] is False
    assert item["duplicate_removed"] is False
    assert item["model"] is None
    assert item["scale"] is None


def test_list_train_images_orphan_when_download_missing(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "Y.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("Y.png"), {"origin": "Y.jpg"}
    )

    items = preprocess.list_train_images(p, "v1")
    assert len(items) == 1
    assert items[0]["orphan"] is True


def test_list_train_images_marks_duplicate_removed(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "A.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", [_rel("A.png")]
    )

    items = preprocess.list_train_images(p, "v1")
    assert len(items) == 1
    assert items[0]["duplicate_removed"] is True


def test_list_train_images_includes_stale_duplicate_removed(isolated) -> None:
    """manifest 有 duplicate_removed entry 但 train/ 物理已删 → 仍报告 stale。"""
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "S.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", [_rel("S.png")]
    )
    (sub / "S.png").unlink()

    items = preprocess.list_train_images(p, "v1")
    assert len(items) == 1
    assert items[0]["name"] == _rel("S.png")
    assert items[0]["duplicate_removed"] is True
    assert items[0]["w"] is None


def test_list_train_images_returns_processed_flag(isolated) -> None:
    """list_train_images 返 `processed` 字段（前端不自己算）。"""
    p = isolated["project"]
    sub = isolated["sub"]
    download = _download_dir(p)
    download.mkdir(parents=True, exist_ok=True)

    # 扩展名变 → processed=True
    _write_png(sub / "ext.png")
    (download / "ext.jpg").write_bytes(b"j")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("ext.png"),
        {"origin": "ext.jpg"},
    )
    # PNG→PNG size diff → processed=True
    _write_png(sub / "big.png", (200, 200))
    Image.new("RGB", (40, 40), "blue").save(download / "big.png", "PNG")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("big.png"),
        {"origin": "big.png"},
    )
    # 原样副本 → processed=False
    _write_png(sub / "raw.png", (30, 30))
    import shutil
    shutil.copy2(sub / "raw.png", download / "raw.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("raw.png"),
        {"origin": "raw.png"},
    )

    items = preprocess.list_train_images(p, "v1")
    by_name = {it["name"]: it for it in items}
    assert by_name[_rel("ext.png")]["processed"] is True
    assert by_name[_rel("big.png")]["processed"] is True
    assert by_name[_rel("raw.png")]["processed"] is False


# ---------------------------------------------------------------------------
# summary_train
# ---------------------------------------------------------------------------


def test_summary_train_counts_physical_plus_stale(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "A.png")
    _write_png(sub / "B.png")
    # stale duplicate_removed
    _write_png(sub / "C.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", [_rel("C.png")]
    )
    (sub / "C.png").unlink()

    s = preprocess.summary_train(p, "v1")
    assert s["image_count"] == 3


# ---------------------------------------------------------------------------
# resolve_targets_train
# ---------------------------------------------------------------------------


def test_resolve_targets_all_lists_train(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "z.png")
    _write_png(sub / "a.png")

    out = preprocess.resolve_targets_train(p, "v1", mode="all")
    assert out == [_rel("a.png"), _rel("z.png")]


def test_resolve_targets_selected_intersects_train(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "X.png")

    out = preprocess.resolve_targets_train(
        p, "v1", mode="selected", names=[_rel("X.png"), _rel("ghost.png")]
    )
    assert out == [_rel("X.png")]


def test_resolve_targets_selected_empty_names_raises(isolated) -> None:
    p = isolated["project"]
    with pytest.raises(preprocess.PreprocessError):
        preprocess.resolve_targets_train(p, "v1", mode="selected", names=[])


def test_resolve_targets_unknown_mode_raises(isolated) -> None:
    with pytest.raises(preprocess.PreprocessError):
        preprocess.resolve_targets_train(
            isolated["project"], "v1", mode="bogus"
        )


def test_resolve_targets_name_with_traversal_rejected(isolated) -> None:
    """`..` 在 name 里被 _validate_name 拒。folder/file POSIX 形式 OK。"""
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "X.png")
    with pytest.raises(preprocess.PreprocessError):
        preprocess.resolve_targets_train(
            p, "v1", mode="selected", names=["../etc/passwd"]
        )


# ---------------------------------------------------------------------------
# start_job_train
# ---------------------------------------------------------------------------


def test_start_job_train_creates_job_with_version_id(isolated) -> None:
    p = isolated["project"]
    v = isolated["version"]
    with db.connection_for(isolated["db"]) as conn:
        job = preprocess.start_job_train(
            conn,
            project_id=p["id"],
            version_id=v["id"],
            mode="all",
        )
    assert job["version_id"] == v["id"]
    assert job["kind"] == preprocess.PREPROCESS_KIND
    import json
    params = json.loads(job["params"]) if isinstance(job["params"], str) else job["params"]
    assert params["stage"] == preprocess.STAGE_UPSCALE
    assert params["mode"] == "all"


def test_start_job_train_selected_requires_names(isolated) -> None:
    p = isolated["project"]
    v = isolated["version"]
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(preprocess.PreprocessError):
            preprocess.start_job_train(
                conn, project_id=p["id"], version_id=v["id"],
                mode="selected",
            )


def test_start_crop_job_train_validates_rects(isolated) -> None:
    p = isolated["project"]
    v = isolated["version"]
    with db.connection_for(isolated["db"]) as conn:
        job = preprocess.start_crop_job_train(
            conn,
            project_id=p["id"],
            version_id=v["id"],
            crops={_rel("X.png"): [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}]},
        )
        assert job["version_id"] == v["id"]
        with pytest.raises(preprocess.PreprocessError):
            preprocess.start_crop_job_train(
                conn, project_id=p["id"], version_id=v["id"],
                crops={_rel("X.png"): [{"x": 0, "y": 0, "w": 0.001, "h": 0.5}]},
            )


# ---------------------------------------------------------------------------
# list_crop_workspace_train
# ---------------------------------------------------------------------------


def test_list_crop_workspace_excludes_duplicate_removed(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "ok.png", (50, 40))
    _write_png(sub / "dup.png", (60, 50))
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", [_rel("dup.png")]
    )

    out = preprocess.list_crop_workspace_train(p, "v1")
    names = [it["name"] for it in out]
    assert names == [_rel("ok.png")]
    assert out[0]["w"] == 50 and out[0]["h"] == 40


def test_list_crop_workspace_processed_flag(isolated) -> None:
    """`_is_processed` 推断（详 ADR 0010）：扩展名变 / size diff vs download。

    覆盖 3 个 case：
    - 扩展名变（X.jpg → X.png upscale）→ processed=True
    - 扩展名同 + size 跟 download 不同（PNG → PNG upscale 后 train 大于
      原图）→ processed=True
    - 扩展名同 + size 跟 download 相同（curate 时复制的原样副本）
      → processed=False
    """
    p = isolated["project"]
    sub = isolated["sub"]
    download = _download_dir(p)
    download.mkdir(parents=True, exist_ok=True)

    # Case 1: 扩展名变（jpg→png upscale）
    _write_png(sub / "ext.png")
    (download / "ext.jpg").write_bytes(b"small-jpg")  # 不需要真图
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("ext.png"),
        {"origin": "ext.jpg"},
    )

    # Case 2: 同扩展名 + size 跟 download 不同（PNG → PNG upscale 后变大）
    _write_png(sub / "big.png", (200, 200))  # upscale 后产物
    Image.new("RGB", (40, 40), "blue").save(
        download / "big.png", "PNG",
    )  # 原图，size 小
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("big.png"),
        {"origin": "big.png"},
    )

    # Case 3: 同扩展名 + size 一致（curate 复制原样）
    _write_png(sub / "raw.png", (30, 30))
    import shutil
    shutil.copy2(sub / "raw.png", download / "raw.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("raw.png"),
        {"origin": "raw.png"},
    )

    out = preprocess.list_crop_workspace_train(p, "v1")
    by_name = {it["name"]: it for it in out}
    assert by_name[_rel("ext.png")]["processed"] is True
    assert by_name[_rel("big.png")]["processed"] is True
    assert by_name[_rel("raw.png")]["processed"] is False


# ---------------------------------------------------------------------------
# list_duplicate_removed_workspace_train
# ---------------------------------------------------------------------------


def test_list_duplicate_removed_workspace_returns_marked(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "Q.png", (40, 30))
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", [_rel("Q.png")]
    )

    out = preprocess.list_duplicate_removed_workspace_train(p, "v1")
    assert len(out) == 1
    assert out[0]["name"] == _rel("Q.png")
    assert out[0]["w"] == 40 and out[0]["h"] == 30


def test_list_duplicate_removed_workspace_stale_entry(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "R.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", [_rel("R.png")]
    )
    (sub / "R.png").unlink()

    out = preprocess.list_duplicate_removed_workspace_train(p, "v1")
    assert len(out) == 1
    assert out[0]["w"] is None


# ---------------------------------------------------------------------------
# restore_products_train
# ---------------------------------------------------------------------------


def test_restore_products_train_copies_download_to_train(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    download = _download_dir(p)
    download.mkdir(parents=True, exist_ok=True)
    (download / "X.jpg").write_bytes(b"original" * 5)
    _write_png(sub / "X.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("X.png"), {"origin": "X.jpg"}
    )

    result = preprocess.restore_products_train(p, "v1", [_rel("X.png")])
    assert result == {"restored": [_rel("X.png")], "missing": [], "no_origin": []}
    assert (sub / "X.png").read_bytes() == b"original" * 5


def test_restore_products_train_no_origin_when_download_missing(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "Y.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("Y.png"), {"origin": "Y.jpg"}
    )

    result = preprocess.restore_products_train(p, "v1", [_rel("Y.png")])
    assert result == {"restored": [], "missing": [], "no_origin": [_rel("Y.png")]}


def test_restore_products_train_validates_name(isolated) -> None:
    p = isolated["project"]
    with pytest.raises(preprocess.PreprocessError):
        preprocess.restore_products_train(p, "v1", ["../etc"])
