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
from studio.domain.errors import InvalidPathError, ValidationError
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


def test_list_train_images_includes_duplicate_removed_tombstone(isolated) -> None:
    """train_mark_duplicate_removed 物理删图 + 留 manifest tombstone → list 仍报告该 entry。"""
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "S.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", [_rel("S.png")]
    )
    assert not (sub / "S.png").exists()  # mark 已删

    items = preprocess.list_train_images(p, "v1")
    assert len(items) == 1
    assert items[0]["name"] == _rel("S.png")
    assert items[0]["duplicate_removed"] is True
    assert items[0]["w"] is None


def test_list_train_images_returns_processed_flag(isolated) -> None:
    """list_train_images 返 `processed` 字段（ADR 0010 fixup：读 manifest 字段）。"""
    p = isolated["project"]
    sub = isolated["sub"]

    _write_png(sub / "up.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("up.png"),
        {"origin": "up.png", "processed": True},
    )
    _write_png(sub / "raw.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("raw.png"),
        {"origin": "raw.png"},
    )

    items = preprocess.list_train_images(p, "v1")
    by_name = {it["name"]: it for it in items}
    assert by_name[_rel("up.png")]["processed"] is True
    assert by_name[_rel("raw.png")]["processed"] is False


# ---------------------------------------------------------------------------
# summary_train
# ---------------------------------------------------------------------------


def test_summary_train_counts_physical_plus_tombstone(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "A.png")
    _write_png(sub / "B.png")
    # duplicate_removed → 物理删 + tombstone 留
    _write_png(sub / "C.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", [_rel("C.png")]
    )

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
    with pytest.raises(ValidationError) as exc:
        preprocess.resolve_targets_train(p, "v1", mode="selected", names=[])
    assert exc.value.code == "preprocess.selection_empty"


def test_resolve_targets_unknown_mode_raises(isolated) -> None:
    with pytest.raises(ValidationError) as exc:
        preprocess.resolve_targets_train(
            isolated["project"], "v1", mode="bogus"
        )
    assert exc.value.code == "preprocess.mode_invalid"


def test_resolve_targets_name_with_traversal_rejected(isolated) -> None:
    """`..` 在 name 里被 _validate_name 拒。folder/file POSIX 形式 OK。"""
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "X.png")
    with pytest.raises(InvalidPathError) as exc:
        preprocess.resolve_targets_train(
            p, "v1", mode="selected", names=["../etc/passwd"]
        )
    assert exc.value.code == "path.invalid"


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
        with pytest.raises(ValidationError) as exc:
            preprocess.start_job_train(
                conn, project_id=p["id"], version_id=v["id"],
                mode="selected",
            )
        assert exc.value.code == "preprocess.selection_empty"


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
        with pytest.raises(ValidationError) as exc:
            preprocess.start_crop_job_train(
                conn, project_id=p["id"], version_id=v["id"],
                crops={_rel("X.png"): [{"x": 0, "y": 0, "w": 0.001, "h": 0.5}]},
            )
        assert exc.value.code == "preprocess.crop_too_small"


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
    """ADR 0010 fixup（2026-06-04）：`_is_processed` 直接读 manifest entry
    的 `processed` 字段（worker 写 True，curate 复制不写）。
    """
    p = isolated["project"]
    sub = isolated["sub"]

    # worker upscale 后写的 entry（processed=True）
    _write_png(sub / "up.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("up.png"),
        {"origin": "up.png", "processed": True},
    )
    # curate 复制原图（无 processed 字段）
    _write_png(sub / "raw.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("raw.png"),
        {"origin": "raw.png"},
    )

    out = preprocess.list_crop_workspace_train(p, "v1")
    by_name = {it["name"]: it for it in out}
    assert by_name[_rel("up.png")]["processed"] is True
    assert by_name[_rel("raw.png")]["processed"] is False


# ---------------------------------------------------------------------------
# inpaint_save_train
# ---------------------------------------------------------------------------


def _png_bytes(size: tuple[int, int] = (10, 10), color: str = "blue") -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, "PNG")
    return buf.getvalue()


def test_inpaint_save_overwrites_png_source(isolated) -> None:
    """png 源同名覆盖：像素被替换、manifest 标 processed、返回新 size/mtime。"""
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "X.png", (10, 10))  # 红色源

    res = preprocess.inpaint_save_train(
        p, "v1", name=_rel("X.png"), data=_png_bytes((10, 10), "blue"),
    )
    assert res["name"] == _rel("X.png")
    assert res["w"] == 10 and res["h"] == 10

    with Image.open(sub / "X.png") as im:
        assert im.getpixel((0, 0)) == (0, 0, 255)

    entry = preprocess_manifest.train_get_entry(
        preprocess.project_root(p), "v1", _rel("X.png")
    )
    assert entry is not None
    assert entry.get("processed") is True
    assert entry["size"] == (sub / "X.png").stat().st_size


def test_inpaint_save_converts_non_png_source(isolated) -> None:
    """jpg 源 → 产物统一 .png、旧源删除、caption sidecar 保留、manifest key 迁移。"""
    p = isolated["project"]
    sub = isolated["sub"]
    Image.new("RGB", (20, 10), color="red").save(sub / "Y.jpg", "JPEG")
    (sub / "Y.txt").write_text("1girl, solo", encoding="utf-8")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("Y.jpg"), {"origin": "Y.jpg"}
    )

    res = preprocess.inpaint_save_train(
        p, "v1", name=_rel("Y.jpg"), data=_png_bytes((20, 10)),
    )
    assert res["name"] == _rel("Y.png")
    assert res["origin"] == "Y.jpg"
    assert (sub / "Y.png").is_file()
    assert not (sub / "Y.jpg").exists()
    assert (sub / "Y.txt").is_file()  # caption 跟 stem 走，不动

    m = preprocess_manifest.train_load(preprocess.project_root(p), "v1")
    assert _rel("Y.jpg") not in m["images"]
    entry = m["images"][_rel("Y.png")]
    assert entry["origin"] == "Y.jpg"
    assert entry.get("processed") is True


def test_inpaint_save_size_mismatch_rejected(isolated) -> None:
    """尺寸不符 → ValidationError，源图字节不动。"""
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "Z.png", (10, 10))
    before = (sub / "Z.png").read_bytes()

    with pytest.raises(ValidationError):
        preprocess.inpaint_save_train(
            p, "v1", name=_rel("Z.png"), data=_png_bytes((11, 10)),
        )
    assert (sub / "Z.png").read_bytes() == before


def test_inpaint_save_invalid_bytes_rejected(isolated) -> None:
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "B.png")

    with pytest.raises(ValidationError):
        preprocess.inpaint_save_train(
            p, "v1", name=_rel("B.png"), data=b"not an image",
        )


def test_inpaint_save_missing_source_rejected(isolated) -> None:
    from studio.domain.errors import NotFoundError

    p = isolated["project"]
    with pytest.raises(NotFoundError):
        preprocess.inpaint_save_train(
            p, "v1", name=_rel("nope.png"), data=_png_bytes(),
        )


def test_inpaint_save_invalid_name_rejected(isolated) -> None:
    p = isolated["project"]
    for bad in ("../X.png", "X.png", "1_data/../X.png", "a/b/c.png", ""):
        with pytest.raises(InvalidPathError):
            preprocess.inpaint_save_train(p, "v1", name=bad, data=_png_bytes())


# ---------------------------------------------------------------------------
# list_duplicate_removed_workspace_train
# ---------------------------------------------------------------------------


def test_list_duplicate_removed_workspace_returns_marked(isolated) -> None:
    """mark 物理删 train/{name}；list 从 download/{origin} 现读 w/h。"""
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "Q.png", (40, 30))
    # download/Q.png 用作 origin 来源
    download = _download_dir(p)
    download.mkdir(parents=True, exist_ok=True)
    _write_png(download / "Q.png", (40, 30))
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", _rel("Q.png"), {"origin": "Q.png"},
    )
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", [_rel("Q.png")]
    )

    out = preprocess.list_duplicate_removed_workspace_train(p, "v1")
    assert len(out) == 1
    assert out[0]["name"] == _rel("Q.png")
    assert out[0]["w"] == 40 and out[0]["h"] == 30


def test_list_duplicate_removed_workspace_no_origin(isolated) -> None:
    """download/{origin} 缺失 → 仍报告 entry，w/h=None。"""
    p = isolated["project"]
    sub = isolated["sub"]
    _write_png(sub / "R.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", [_rel("R.png")]
    )

    out = preprocess.list_duplicate_removed_workspace_train(p, "v1")
    assert len(out) == 1
    assert out[0]["w"] is None


# ---------------------------------------------------------------------------
# restore_products_train
# ---------------------------------------------------------------------------


def test_restore_products_train_copies_download_to_train(isolated) -> None:
    """restore X.png (entry origin=X.jpg) → train/1_data/X.jpg + 删原 entry。"""
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
    # 新文件落在 {folder}/{origin}
    assert (sub / "X.jpg").read_bytes() == b"original" * 5
    # 老 entry / 文件清掉
    assert not (sub / "X.png").exists()
    assert preprocess_manifest.train_get_entry(
        preprocess.project_root(p), "v1", _rel("X.png")
    ) is None
    new_entry = preprocess_manifest.train_get_entry(
        preprocess.project_root(p), "v1", _rel("X.jpg")
    )
    assert new_entry is not None and new_entry["origin"] == "X.jpg"


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
    with pytest.raises(InvalidPathError) as exc:
        preprocess.restore_products_train(p, "v1", ["../etc"])
    assert exc.value.code == "path.invalid"
