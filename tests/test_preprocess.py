"""preprocess 业务层：list_pending / list_processed / resolve_targets /
start_job / delete_products。

不跑真 upscaler — worker 的端到端走 test_supervisor_jobs / 手测。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio import db, preprocess, project_jobs, projects


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(projects, "TRASH_DIR", tmp_path / "_trash")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="PP")
    return {"db": dbfile, "project": p}


def _drop(d: Path, name: str, content: bytes = b"x") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_bytes(content)
    return f


def _seed_download(project: dict, names: list[str]) -> Path:
    download, _ = preprocess.project_paths(project)
    download.mkdir(parents=True, exist_ok=True)
    for n in names:
        _drop(download, n)
    return download


def _seed_processed(project: dict, source_to_meta: dict[str, dict]) -> Path:
    """source_to_meta: {source_name: {scale, model, ...}} → 写产物 + sidecar。"""
    _, pre = preprocess.project_paths(project)
    pre.mkdir(parents=True, exist_ok=True)
    for src, meta in source_to_meta.items():
        product = preprocess.product_path_for(pre, src)
        product.write_bytes(b"upscaled-bytes")
        side = preprocess.sidecar_for(product)
        full_meta = {"source": src, **meta}
        side.write_text(json.dumps(full_meta), encoding="utf-8")
    return pre


# ---------------------------------------------------------------------------
# list_pending / list_processed / summary
# ---------------------------------------------------------------------------


def test_list_pending_empty_when_no_download(isolated) -> None:
    assert preprocess.list_pending(isolated["project"]) == []


def test_list_pending_excludes_processed(isolated) -> None:
    p = isolated["project"]
    _seed_download(p, ["a.png", "b.webp", "c.jpg"])
    _seed_processed(p, {"a.png": {"scale": 4, "model": "4x-AnimeSharp"}})
    pending = preprocess.list_pending(p)
    names = [it["name"] for it in pending]
    assert names == ["b.webp", "c.jpg"]
    assert all("mtime" in it and "size" in it for it in pending)


def test_list_processed_reads_sidecar(isolated) -> None:
    p = isolated["project"]
    _seed_download(p, ["a.png"])
    _seed_processed(
        p,
        {"a.png": {
            "scale": 4, "model": "4x-AnimeSharp",
            "src_size": [32, 32], "dst_size": [128, 128],
            "elapsed_seconds": 1.2,
        }},
    )
    items = preprocess.list_processed(p)
    assert len(items) == 1
    it = items[0]
    assert it["name"] == "a.png"
    assert it["source"] == "a.png"
    assert it["model"] == "4x-AnimeSharp"
    assert it["scale"] == 4
    assert it["src_size"] == [32, 32]
    assert it["orphan"] is False


def test_list_processed_marks_orphans(isolated) -> None:
    """源已删的产物 orphan=True。"""
    p = isolated["project"]
    _seed_download(p, ["alive.png"])
    _seed_processed(
        p,
        {
            "alive.png": {"scale": 4, "model": "4x-AnimeSharp"},
            "deleted.jpg": {"scale": 4, "model": "4x-AnimeSharp"},
        },
    )
    items = {it["name"]: it for it in preprocess.list_processed(p)}
    # 注意产物文件名都是 .png（产物固定 png）
    assert items["alive.png"]["orphan"] is False
    assert items["deleted.png"]["orphan"] is True


def test_list_processed_handles_missing_sidecar(isolated) -> None:
    """产物存在但 sidecar 丢了：source/model 为 None，依旧返回。"""
    p = isolated["project"]
    _seed_download(p, ["foo.png"])
    _, pre = preprocess.project_paths(p)
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "foo.png").write_bytes(b"x")
    items = preprocess.list_processed(p)
    assert len(items) == 1
    assert items[0]["name"] == "foo.png"
    assert items[0]["source"] is None
    assert items[0]["scale"] is None


def test_summary_counts(isolated) -> None:
    p = isolated["project"]
    _seed_download(p, ["a.png", "b.png", "c.png"])
    _seed_processed(p, {"a.png": {"scale": 4}})
    s = preprocess.summary(p)
    assert s == {
        "download_count": 3,
        "processed_count": 1,
        "pending_count": 2,
    }


# ---------------------------------------------------------------------------
# resolve_targets
# ---------------------------------------------------------------------------


def test_resolve_all_returns_pending_only(isolated) -> None:
    p = isolated["project"]
    _seed_download(p, ["a.png", "b.png"])
    _seed_processed(p, {"a.png": {"scale": 4}})
    assert preprocess.resolve_targets(p, mode="all") == ["b.png"]


def test_resolve_all_force_returns_everything(isolated) -> None:
    p = isolated["project"]
    _seed_download(p, ["a.png", "b.png"])
    _seed_processed(p, {"a.png": {"scale": 4}})
    assert preprocess.resolve_targets(p, mode="all_force") == ["a.png", "b.png"]


def test_resolve_selected_intersects_with_existing(isolated) -> None:
    p = isolated["project"]
    _seed_download(p, ["a.png", "b.png"])
    chosen = preprocess.resolve_targets(
        p, mode="selected", names=["a.png", "ghost.png", "b.png", "a.png"]
    )
    # 去重 + 与磁盘交集
    assert chosen == ["a.png", "b.png"]


def test_resolve_selected_requires_names(isolated) -> None:
    p = isolated["project"]
    _seed_download(p, ["a.png"])
    with pytest.raises(preprocess.PreprocessError, match="names"):
        preprocess.resolve_targets(p, mode="selected", names=[])


def test_resolve_selected_rejects_path_traversal(isolated) -> None:
    p = isolated["project"]
    _seed_download(p, ["a.png"])
    with pytest.raises(preprocess.PreprocessError, match="非法文件名"):
        preprocess.resolve_targets(p, mode="selected", names=["../../etc/passwd"])


def test_resolve_unknown_mode(isolated) -> None:
    with pytest.raises(preprocess.PreprocessError, match="未知 mode"):
        preprocess.resolve_targets(isolated["project"], mode="weird")


# ---------------------------------------------------------------------------
# start_job
# ---------------------------------------------------------------------------


def test_start_job_creates_pending_preprocess(isolated) -> None:
    p = isolated["project"]
    with db.connection_for(isolated["db"]) as conn:
        job = preprocess.start_job(
            conn,
            project_id=p["id"],
            mode="all",
            model="4x-AnimeSharp",
            tile_size=128,
        )
    assert job["status"] == "pending"
    assert job["kind"] == "preprocess"
    assert job["params_decoded"]["mode"] == "all"
    assert job["params_decoded"]["tile_size"] == 128
    assert job["params_decoded"]["model"] == "4x-AnimeSharp"


def test_start_job_selected_requires_names(isolated) -> None:
    p = isolated["project"]
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(preprocess.PreprocessError, match="names"):
            preprocess.start_job(conn, project_id=p["id"], mode="selected")


def test_start_job_validates_names(isolated) -> None:
    p = isolated["project"]
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(preprocess.PreprocessError, match="非法文件名"):
            preprocess.start_job(
                conn, project_id=p["id"], mode="selected", names=["a/b"]
            )


def test_start_job_missing_project(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(preprocess.PreprocessError, match="不存在"):
            preprocess.start_job(conn, project_id=9999, mode="all")


# ---------------------------------------------------------------------------
# delete_products
# ---------------------------------------------------------------------------


def test_delete_products_removes_image_and_sidecar(isolated) -> None:
    p = isolated["project"]
    _seed_download(p, ["a.png"])
    _seed_processed(p, {"a.png": {"scale": 4}})

    _, pre = preprocess.project_paths(p)
    assert (pre / "a.png").exists()
    assert preprocess.sidecar_for(pre / "a.png").exists()

    res = preprocess.delete_products(p, ["a.png", "ghost.png"])
    assert res == {"deleted": ["a.png"], "missing": ["ghost.png"]}
    assert not (pre / "a.png").exists()
    assert not preprocess.sidecar_for(pre / "a.png").exists()


def test_delete_products_rejects_traversal(isolated) -> None:
    with pytest.raises(preprocess.PreprocessError, match="非法文件名"):
        preprocess.delete_products(isolated["project"], ["..\\foo"])
