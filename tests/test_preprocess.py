"""preprocess 业务层：list_pending / list_processed / resolve_targets /
start_job / restore_products。

ADR 0004：状态走 manifest，list_processed 不再从 sidecar 读。
不跑真 upscaler — worker 的端到端走 test_supervisor_jobs / 手测。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db, preprocess, project_jobs, projects
from studio.services import preprocess_manifest


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="PP")
    return {"db": dbfile, "project": p}


def _seed_download(project: dict, names: list[str]) -> Path:
    download, _ = preprocess.project_paths(project)
    download.mkdir(parents=True, exist_ok=True)
    for n in names:
        (download / n).write_bytes(b"src")
    return download


def _seed_processed(project: dict, source_to_meta: dict[str, dict]) -> Path:
    """source_to_meta: {source_name: meta} → 写 PNG 副本 + 老 schema manifest entry。

    产物按约定固定 `.png`（取 source 的 stem）。直接写 manifest.json 而非走
    `add_processed`，因为 0.9.x 后 `add_processed` 只写新 schema（origin/mtime/size），
    fixture 想保留老 schema 字段（kind/model/scale/...）来覆盖读兼容路径。
    新 schema 的写入路径有独立单测。
    """
    import json
    import time
    pdir = projects.project_dir(project["id"], project["slug"])
    _, pre = preprocess.project_paths(project)
    pre.mkdir(parents=True, exist_ok=True)
    manifest_p = preprocess_manifest.manifest_path(pdir)
    manifest_p.parent.mkdir(parents=True, exist_ok=True)
    images: dict[str, dict] = {}
    if manifest_p.exists():
        try:
            existing = json.loads(manifest_p.read_text(encoding="utf-8"))
            if isinstance(existing.get("images"), dict):
                images = existing["images"]
        except Exception:
            pass
    for src, meta in source_to_meta.items():
        product_name = Path(src).stem + ".png"
        (pre / product_name).write_bytes(b"upscaled")
        # 老 schema：kind + source + 任意 meta
        images[product_name] = {
            "kind": "processed",
            "source": src,
            "mtime": time.time(),
            **meta,
        }
    manifest_p.write_text(
        json.dumps({"images": images}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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


def test_list_processed_reads_manifest(isolated) -> None:
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
    """源已删的产物 orphan=True（按 manifest entry 的 source 字段比对）。"""
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
    # 产物文件名都是 .png（{stem}.png）
    assert items["alive.png"]["orphan"] is False
    assert items["deleted.png"]["orphan"] is True


def test_summary_image_count(isolated) -> None:
    """summary 只返回 image_count = grid 当前总图数（ADR 0004 Addendum 1
    §「Stage 不强制时序」，不再分 pending / processed）。"""
    p = isolated["project"]
    _seed_download(p, ["a.png", "b.png", "c.png"])
    _seed_processed(p, {"a.png": {"scale": 4}})
    s = preprocess.summary(p)
    # a.png 走派生路径（manifest entry）+ b.png, c.png 走 download 原图路径 = 3 张
    assert s == {"image_count": 3}


def test_summary_counts_multicrop_fanout(isolated) -> None:
    """multi-crop 派生让一张 download 原图对应多个 entry → image_count 反映 grid 实际看到的行数。"""
    p = isolated["project"]
    _seed_download(p, ["a.png", "b.png"])
    # 模拟 a.png 被多裁剪成 a_c0.png / a_c1.png（两 entry 共享 origin=a.png）
    import json
    import time
    pdir = projects.project_dir(p["id"], p["slug"])
    _, pre = preprocess.project_paths(p)
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "a_c0.png").write_bytes(b"crop0")
    (pre / "a_c1.png").write_bytes(b"crop1")
    manifest_p = preprocess_manifest.manifest_path(pdir)
    manifest_p.parent.mkdir(parents=True, exist_ok=True)
    manifest_p.write_text(json.dumps({"images": {
        "a_c0.png": {"origin": "a.png", "mtime": time.time(), "size": 1},
        "a_c1.png": {"origin": "a.png", "mtime": time.time(), "size": 1},
    }}), encoding="utf-8")
    # a.png 派生覆盖 → 不走原图路径；b.png 仍走原图。grid 看到 a_c0/a_c1/b = 3
    assert preprocess.summary(p) == {"image_count": 3}


# ---------------------------------------------------------------------------
# resolve_targets
# ---------------------------------------------------------------------------


def test_resolve_all_returns_all_current_images(isolated) -> None:
    """ADR 0004 Addendum 1 §「Stage 不强制时序」：mode='all' = grid 全部当前图，
    包括 manifest 已有派生（裁剪 / 上次放大产物），让"裁剪后→放大"/"再放大"链路可走。"""
    p = isolated["project"]
    _seed_download(p, ["a.png", "b.png"])
    _seed_processed(p, {"a.png": {"scale": 4}})
    # a.png 派生（manifest entry）+ b.png 原图 → 两个都该被处理
    assert preprocess.resolve_targets(p, mode="all") == ["a.png", "b.png"]


def test_resolve_all_force_alias_of_all(isolated) -> None:
    """all_force 跟 all 语义已统一（ADR 0004 Addendum 1）；保留别名兼容老前端。"""
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


def test_start_job_records_stage_upscale(isolated) -> None:
    """放大 job 的 params 现在带 stage=upscale，方便 worker 分发。"""
    p = isolated["project"]
    with db.connection_for(isolated["db"]) as conn:
        job = preprocess.start_job(conn, project_id=p["id"], mode="all")
    assert job["params_decoded"]["stage"] == preprocess.STAGE_UPSCALE


# ---------------------------------------------------------------------------
# start_crop_job
# ---------------------------------------------------------------------------


def test_start_crop_job_creates_pending(isolated) -> None:
    """裁剪 job 走 preprocess kind + stage=crop，params 带归一化 rects。"""
    p = isolated["project"]
    with db.connection_for(isolated["db"]) as conn:
        job = preprocess.start_crop_job(
            conn,
            project_id=p["id"],
            crops={
                "a.png": [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5, "label": "头像"}],
                "b.png": [
                    {"x": 0.0, "y": 0.0, "w": 0.4, "h": 0.4},
                    {"x": 0.5, "y": 0.5, "w": 0.4, "h": 0.4},
                ],
            },
        )
    assert job["status"] == "pending"
    assert job["kind"] == "preprocess"
    decoded = job["params_decoded"]
    assert decoded["stage"] == preprocess.STAGE_CROP
    assert set(decoded["crops"].keys()) == {"a.png", "b.png"}
    # label 保留；rect 字段齐
    assert decoded["crops"]["a.png"][0]["label"] == "头像"
    assert decoded["crops"]["a.png"][0]["x"] == 0.1
    assert len(decoded["crops"]["b.png"]) == 2


def test_start_crop_job_requires_non_empty(isolated) -> None:
    p = isolated["project"]
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(preprocess.PreprocessError, match="crops"):
            preprocess.start_crop_job(conn, project_id=p["id"], crops={})


def test_start_crop_job_rejects_empty_rects(isolated) -> None:
    p = isolated["project"]
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(preprocess.PreprocessError, match="rects"):
            preprocess.start_crop_job(
                conn, project_id=p["id"], crops={"a.png": []}
            )


def test_start_crop_job_rejects_tiny_rect(isolated) -> None:
    p = isolated["project"]
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(preprocess.PreprocessError, match="过小"):
            preprocess.start_crop_job(
                conn,
                project_id=p["id"],
                crops={"a.png": [{"x": 0.5, "y": 0.5, "w": 0.001, "h": 0.001}]},
            )


def test_start_crop_job_rejects_path_traversal(isolated) -> None:
    p = isolated["project"]
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(preprocess.PreprocessError, match="非法文件名"):
            preprocess.start_crop_job(
                conn,
                project_id=p["id"],
                crops={"../etc/passwd": [{"x": 0, "y": 0, "w": 0.5, "h": 0.5}]},
            )


def test_start_crop_job_clamps_out_of_bounds(isolated) -> None:
    """超出 [0,1] 的 rect 自动 clamp（不抛错；用户 UI bug 不该让 job 失败）。"""
    p = isolated["project"]
    with db.connection_for(isolated["db"]) as conn:
        job = preprocess.start_crop_job(
            conn,
            project_id=p["id"],
            crops={"a.png": [{"x": -0.1, "y": -0.1, "w": 2.0, "h": 2.0}]},
        )
    r = job["params_decoded"]["crops"]["a.png"][0]
    assert r["x"] == 0.0 and r["y"] == 0.0
    assert r["w"] == 1.0 and r["h"] == 1.0


def test_start_job_missing_project(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(preprocess.PreprocessError, match="不存在"):
            preprocess.start_job(conn, project_id=9999, mode="all")


# ---------------------------------------------------------------------------
# restore_products
# ---------------------------------------------------------------------------


def test_restore_products_removes_image_and_entry(isolated) -> None:
    p = isolated["project"]
    pdir = projects.project_dir(p["id"], p["slug"])
    _seed_download(p, ["a.png"])
    _seed_processed(p, {"a.png": {"scale": 4}})

    _, pre = preprocess.project_paths(p)
    assert (pre / "a.png").exists()
    assert preprocess_manifest.get_entry(pdir, "a.png") is not None

    res = preprocess.restore_products(p, ["a.png", "ghost.png"])
    assert res == {"restored": ["a.png"], "missing": ["ghost.png"]}
    assert not (pre / "a.png").exists()
    assert preprocess_manifest.get_entry(pdir, "a.png") is None


def test_restore_products_rejects_traversal(isolated) -> None:
    with pytest.raises(preprocess.PreprocessError, match="非法文件名"):
        preprocess.restore_products(isolated["project"], ["..\\foo"])
