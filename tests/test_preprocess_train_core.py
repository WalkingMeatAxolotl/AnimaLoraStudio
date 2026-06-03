"""ADR 0010 — train-scope core API（PR-2 step B）。

覆盖 `list_train_images / summary_train / resolve_targets_train / start_job_train
/ start_crop_job_train / list_crop_workspace_train /
list_duplicate_removed_workspace_train / restore_products_train`。

老 project-scope API 单测在 test_preprocess.py，本文件不覆盖。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from studio import db
from studio.services.preprocess import core as preprocess
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.projects import jobs as project_jobs, projects, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """复用 test_preprocess.py 的 isolated 模式 + 自动创建 v1 version。"""
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="PP")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    return {"db": dbfile, "project": p, "version": v}


def _write_png(path: Path, size: tuple[int, int] = (10, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="red").save(path, "PNG")


def _train_dir(p: dict, label: str = "v1") -> Path:
    return preprocess.version_train_dir(p, label)


def _download_dir(p: dict) -> Path:
    d, _ = preprocess.project_paths(p)
    return d


# ---------------------------------------------------------------------------
# list_train_images
# ---------------------------------------------------------------------------


def test_list_train_images_empty_when_no_train(isolated) -> None:
    items = preprocess.list_train_images(isolated["project"], "v1")
    assert items == []


def test_list_train_images_basic_entry(isolated) -> None:
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "X.png", (100, 80))
    download = _download_dir(p)
    download.mkdir(parents=True, exist_ok=True)
    (download / "X.jpg").write_bytes(b"orig")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", "X.png", {"origin": "X.jpg"}
    )

    items = preprocess.list_train_images(p, "v1")
    assert len(items) == 1
    item = items[0]
    assert item["name"] == "X.png"
    assert item["origin"] == "X.jpg"
    assert item["source"] == "X.jpg"
    assert item["w"] == 100 and item["h"] == 80
    assert item["orphan"] is False
    assert item["duplicate_removed"] is False
    # 老 schema 透传字段应为 None
    assert item["model"] is None
    assert item["scale"] is None


def test_list_train_images_orphan_when_download_missing(isolated) -> None:
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "Y.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", "Y.png", {"origin": "Y.jpg"}
    )
    # 不创建 download/Y.jpg

    items = preprocess.list_train_images(p, "v1")
    assert len(items) == 1
    assert items[0]["orphan"] is True


def test_list_train_images_marks_duplicate_removed(isolated) -> None:
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "A.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", ["A.png"]
    )

    items = preprocess.list_train_images(p, "v1")
    # 物理文件 + manifest 都在，只是 kind=duplicate_removed
    assert len(items) == 1
    assert items[0]["duplicate_removed"] is True


def test_list_train_images_includes_stale_duplicate_removed(isolated) -> None:
    """manifest 有 duplicate_removed entry 但 train/ 物理已删 → 仍报告（stale）。"""
    p = isolated["project"]
    train = _train_dir(p)
    train.mkdir(parents=True, exist_ok=True)
    _write_png(train / "S.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", ["S.png"]
    )
    # 模拟用户外部删 train/S.png
    (train / "S.png").unlink()

    items = preprocess.list_train_images(p, "v1")
    assert len(items) == 1
    assert items[0]["name"] == "S.png"
    assert items[0]["duplicate_removed"] is True
    assert items[0]["w"] is None  # stale 不读图头


def test_list_train_images_processed_state_from_origin_diff(isolated) -> None:
    """名字 != origin → 隐含"处理过"（新 schema 状态推断）。"""
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "P.png")
    download = _download_dir(p)
    download.mkdir(parents=True, exist_ok=True)
    (download / "P.jpg").write_bytes(b"o")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", "P.png", {"origin": "P.jpg"}
    )

    items = preprocess.list_train_images(p, "v1")
    # name "P.png" vs origin "P.jpg" 后缀不同 → 处理过；UI 看 source != name 推断
    assert items[0]["origin"] != items[0]["name"]


# ---------------------------------------------------------------------------
# summary_train
# ---------------------------------------------------------------------------


def test_summary_train_counts_physical_plus_stale(isolated) -> None:
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "A.png")
    _write_png(train / "B.png")
    # stale duplicate_removed（manifest 有 + 物理无）
    _write_png(train / "C.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", ["C.png"]
    )
    (train / "C.png").unlink()

    s = preprocess.summary_train(p, "v1")
    assert s["image_count"] == 3  # A + B 物理 + C stale


# ---------------------------------------------------------------------------
# resolve_targets_train
# ---------------------------------------------------------------------------


def test_resolve_targets_all_lists_train(isolated) -> None:
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "z.png")
    _write_png(train / "a.png")

    out = preprocess.resolve_targets_train(p, "v1", mode="all")
    assert out == ["a.png", "z.png"]  # 字典序


def test_resolve_targets_selected_intersects_train(isolated) -> None:
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "X.png")

    out = preprocess.resolve_targets_train(
        p, "v1", mode="selected", names=["X.png", "ghost.png"]
    )
    assert out == ["X.png"]


def test_resolve_targets_selected_empty_names_raises(isolated) -> None:
    p = isolated["project"]
    with pytest.raises(preprocess.PreprocessError):
        preprocess.resolve_targets_train(p, "v1", mode="selected", names=[])


def test_resolve_targets_unknown_mode_raises(isolated) -> None:
    with pytest.raises(preprocess.PreprocessError):
        preprocess.resolve_targets_train(
            isolated["project"], "v1", mode="bogus"
        )


def test_resolve_targets_name_with_slash_rejected(isolated) -> None:
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "X.png")
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
            crops={"X.png": [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}]},
        )
        assert job["version_id"] == v["id"]
        # 非法 rect 拒
        with pytest.raises(preprocess.PreprocessError):
            preprocess.start_crop_job_train(
                conn, project_id=p["id"], version_id=v["id"],
                crops={"X.png": [{"x": 0, "y": 0, "w": 0.001, "h": 0.5}]},
            )


# ---------------------------------------------------------------------------
# list_crop_workspace_train
# ---------------------------------------------------------------------------


def test_list_crop_workspace_excludes_duplicate_removed(isolated) -> None:
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "ok.png", (50, 40))
    _write_png(train / "dup.png", (60, 50))
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", ["dup.png"]
    )

    out = preprocess.list_crop_workspace_train(p, "v1")
    names = [it["name"] for it in out]
    assert names == ["ok.png"]
    assert out[0]["w"] == 50 and out[0]["h"] == 40


def test_list_crop_workspace_processed_flag(isolated) -> None:
    """name != origin → processed=True；name == origin → processed=False。"""
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "proc.png")
    _write_png(train / "raw.jpg")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", "proc.png", {"origin": "proc.jpg"}
    )
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", "raw.jpg", {"origin": "raw.jpg"}
    )

    out = preprocess.list_crop_workspace_train(p, "v1")
    by_name = {it["name"]: it for it in out}
    assert by_name["proc.png"]["processed"] is True
    assert by_name["raw.jpg"]["processed"] is False


# ---------------------------------------------------------------------------
# list_duplicate_removed_workspace_train
# ---------------------------------------------------------------------------


def test_list_duplicate_removed_workspace_returns_marked(isolated) -> None:
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "Q.png", (40, 30))
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", ["Q.png"]
    )

    out = preprocess.list_duplicate_removed_workspace_train(p, "v1")
    assert len(out) == 1
    assert out[0]["name"] == "Q.png"
    assert out[0]["w"] == 40 and out[0]["h"] == 30


def test_list_duplicate_removed_workspace_stale_entry(isolated) -> None:
    """物理已删 → 仍返回 stale entry，w/h None。"""
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "R.png")
    preprocess_manifest.train_mark_duplicate_removed(
        preprocess.project_root(p), "v1", ["R.png"]
    )
    (train / "R.png").unlink()

    out = preprocess.list_duplicate_removed_workspace_train(p, "v1")
    assert len(out) == 1
    assert out[0]["w"] is None


# ---------------------------------------------------------------------------
# restore_products_train
# ---------------------------------------------------------------------------


def test_restore_products_train_copies_download_to_train(isolated) -> None:
    """完整 restore 流程：download/X.jpg → train/X.png 覆盖。"""
    p = isolated["project"]
    train = _train_dir(p)
    download = _download_dir(p)
    download.mkdir(parents=True, exist_ok=True)
    (download / "X.jpg").write_bytes(b"original" * 5)
    _write_png(train / "X.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", "X.png", {"origin": "X.jpg"}
    )

    result = preprocess.restore_products_train(p, "v1", ["X.png"])
    assert result == {"restored": ["X.png"], "missing": [], "no_origin": []}
    assert (train / "X.png").read_bytes() == b"original" * 5


def test_restore_products_train_no_origin_when_download_missing(isolated) -> None:
    p = isolated["project"]
    train = _train_dir(p)
    _write_png(train / "Y.png")
    preprocess_manifest.train_add_processed(
        preprocess.project_root(p), "v1", "Y.png", {"origin": "Y.jpg"}
    )
    # 不创建 download/Y.jpg

    result = preprocess.restore_products_train(p, "v1", ["Y.png"])
    assert result == {"restored": [], "missing": [], "no_origin": ["Y.png"]}


def test_restore_products_train_validates_name(isolated) -> None:
    p = isolated["project"]
    with pytest.raises(preprocess.PreprocessError):
        preprocess.restore_products_train(p, "v1", ["../etc"])
