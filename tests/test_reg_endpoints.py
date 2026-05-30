"""PP5 — /api/projects/.../reg/* HTTP。"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio import db, secrets, server
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services.reg import builder as reg_builder


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


@pytest.fixture
def client(env) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


def _make(client: TestClient) -> tuple[int, int]:
    p = client.post("/api/projects", json={"title": "P"}).json()
    return p["id"], p["versions"][0]["id"]


def _seed_train(
    client: TestClient, pid: int, vid: int, folder: str, files: dict[str, list[str]]
) -> Path:
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    train = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "train"
    d = train / folder
    d.mkdir(parents=True, exist_ok=True)
    for name, tags in files.items():
        # 真写一张 PNG（preview-tags 走 read_tags 但需要 IMAGE_EXTS 判定）
        img = Image.new("RGB", (32, 32), (255, 0, 0))
        img.save(d / name, "PNG")
        (d / name).with_suffix(".txt").write_text(", ".join(tags), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# preview-tags
# ---------------------------------------------------------------------------


def test_preview_tags_returns_top_freq(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {
        "a.png": ["1girl", "solo", "blue_hair"],
        "b.png": ["1girl", "long_hair"],
        "c.png": ["1girl", "outdoor"],
    })
    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg/preview-tags?top=3")
    assert r.status_code == 200
    items = r.json()["items"]
    # 1girl 最高（3 次）
    assert items[0]["tag"] == "1girl"
    assert items[0]["count"] == 3
    assert len(items) == 3


def test_preview_tags_empty_train(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg/preview-tags")
    assert r.status_code == 200
    assert r.json()["items"] == []


# ---------------------------------------------------------------------------
# GET /reg
# ---------------------------------------------------------------------------


def test_get_reg_when_not_exists(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg")
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is False
    assert body["meta"] is None
    assert body["image_count"] == 0


def test_get_reg_when_exists_returns_meta_and_files(
    client: TestClient, tmp_path: Path
) -> None:
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    rdir = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg"
    rdir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (16, 16), (0, 0, 255))
    img.save(rdir / "100.png", "PNG")
    img.save(rdir / "200.png", "PNG")
    meta = reg_builder.RegMeta(
        generated_at=1.0, based_on_version="baseline", api_source="gelbooru",
        target_count=5, actual_count=2, source_tags=["1girl"],
        excluded_tags=[], blacklist_tags=[], failed_tags=[],
        train_tag_distribution={"1girl": 5}, auto_tagged=True,
    )
    reg_builder.write_meta(rdir, meta)

    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg")
    body = r.json()
    assert body["exists"] is True
    assert body["image_count"] == 2
    assert sorted(body["files"]) == ["100.png", "200.png"]
    assert body["meta"]["actual_count"] == 2
    assert body["meta"]["auto_tagged"] is True


# ---------------------------------------------------------------------------
# POST /reg/build
# ---------------------------------------------------------------------------


def test_start_reg_build_requires_train_data(client: TestClient) -> None:
    pid, vid = _make(client)
    # train 空 → 400
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build", json={}
    )
    assert r.status_code == 400


def test_start_reg_build_creates_job(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {
        "1.png": ["1girl"],
    })
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"excluded_tags": ["character_x"], "auto_tag": False},
    )
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["kind"] == "reg_build"
    assert job["status"] == "pending"
    p_dict = json.loads(job["params"])
    # 目标数量永远镜像 train 总数 — params 不再有 target_count
    assert "target_count" not in p_dict
    assert p_dict["excluded_tags"] == ["character_x"]
    assert p_dict["auto_tag"] is False
    # 默认进阶参数也透传
    assert p_dict["skip_similar"] is True
    assert p_dict["postprocess_method"] == "smart"
    # ADR-0007 PR-5: reg job 不再自动推 stage；phase cursor 由用户推进


def test_start_reg_build_passes_through_advanced_params(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={
            "auto_tag": False,
            "skip_similar": False,
            "aspect_ratio_filter_enabled": True,
            "min_aspect_ratio": 0.6,
            "max_aspect_ratio": 1.8,
            "postprocess_method": "stretch",
            "postprocess_max_crop_ratio": 0.2,
        },
    )
    assert r.status_code == 200
    p_dict = json.loads(r.json()["params"])
    assert "batch_size" not in p_dict  # batch_size 不再暴露 / 不再写入 params
    assert p_dict["skip_similar"] is False
    assert p_dict["aspect_ratio_filter_enabled"] is True
    assert p_dict["min_aspect_ratio"] == 0.6
    assert p_dict["postprocess_method"] == "stretch"
    assert p_dict["postprocess_max_crop_ratio"] == 0.2


def test_start_reg_build_rejects_bad_postprocess_method(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"postprocess_method": "weird", "auto_tag": False},
    )
    assert r.status_code == 400


def test_start_reg_build_rejects_bad_max_crop(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"postprocess_max_crop_ratio": 0.9, "auto_tag": False},
    )
    assert r.status_code == 400


def test_start_reg_build_rejects_bad_api_source(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"api_source": "pixiv"},
    )
    assert r.status_code == 400


def test_start_reg_build_passes_through_incremental(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"incremental": True, "auto_tag": False},
    )
    assert r.status_code == 200
    p_dict = json.loads(r.json()["params"])
    assert p_dict["incremental"] is True


# ---------------------------------------------------------------------------
# DELETE /reg
# ---------------------------------------------------------------------------


def test_delete_reg_removes_content(client: TestClient) -> None:
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    rdir = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg"
    rdir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(rdir / "1.png", "PNG")

    r = client.delete(f"/api/projects/{pid}/versions/{vid}/reg")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    # 空目录保留
    assert rdir.exists()
    assert not (rdir / "1.png").exists()


def test_delete_reg_when_empty(client: TestClient) -> None:
    """version_dir 已自动建空 reg/，无内容 → deleted=False。"""
    pid, vid = _make(client)
    r = client.delete(f"/api/projects/{pid}/versions/{vid}/reg")
    assert r.status_code == 200
    assert r.json()["deleted"] is False


# ---------------------------------------------------------------------------
# GET /reg/caption
# ---------------------------------------------------------------------------


def test_get_reg_caption_returns_tags(client: TestClient) -> None:
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    rdir = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg" / "5_concept"
    rdir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(rdir / "100.png", "PNG")
    (rdir / "100.txt").write_text("a, b, c", encoding="utf-8")
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/reg/caption?path=5_concept/100.png"
    )
    assert r.status_code == 200
    assert r.json()["tags"] == ["a", "b", "c"]


def test_get_reg_caption_rejects_path_traversal(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/reg/caption?path=../train/x.png"
    )
    assert r.status_code == 400


def test_get_reg_caption_404_for_missing(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/reg/caption?path=ghost.png"
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# A1 — POST /reg/delete-files
# ---------------------------------------------------------------------------


def _seed_reg(
    client: TestClient,
    pid: int,
    vid: int,
    folder: str,
    files: dict[str, list[str]],
) -> Path:
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    base = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg"
    d = base / folder if folder else base
    d.mkdir(parents=True, exist_ok=True)
    for name, tags in files.items():
        Image.new("RGB", (8, 8)).save(d / name, "PNG")
        if tags:
            (d / name).with_suffix(".txt").write_text(
                ", ".join(tags), encoding="utf-8"
            )
    return base


def _write_meta(rdir: Path, actual: int) -> None:
    meta = reg_builder.RegMeta(
        generated_at=0.0,
        based_on_version="v1",
        api_source="gelbooru",
        target_count=actual,
        actual_count=actual,
        source_tags=[],
        excluded_tags=[],
        blacklist_tags=[],
        failed_tags=[],
        train_tag_distribution={},
        auto_tagged=False,
    )
    reg_builder.write_meta(rdir, meta)


def test_delete_reg_files_removes_image_and_caption(client: TestClient) -> None:
    pid, vid = _make(client)
    rdir = _seed_reg(
        client, pid, vid, "5_concept",
        {"100.png": ["a", "b"], "101.png": ["c"], "102.png": []},
    )
    _write_meta(rdir, actual=3)

    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": ["5_concept/100.png", "5_concept/101.png"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    assert set(body["deleted"]) == {"5_concept/100.png", "5_concept/101.png"}
    # 图 + 同名 .txt 都删了；剩下的图保留
    assert not (rdir / "5_concept" / "100.png").exists()
    assert not (rdir / "5_concept" / "100.txt").exists()
    assert not (rdir / "5_concept" / "101.png").exists()
    assert (rdir / "5_concept" / "102.png").exists()


def test_delete_reg_files_updates_meta_actual_count(client: TestClient) -> None:
    pid, vid = _make(client)
    rdir = _seed_reg(
        client, pid, vid, "5_concept",
        {"100.png": ["a"], "101.png": ["b"], "102.png": ["c"]},
    )
    _write_meta(rdir, actual=3)

    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": ["5_concept/100.png"]},
    )
    assert r.status_code == 200
    meta = reg_builder.read_meta(rdir)
    assert meta is not None
    assert meta.actual_count == 2


def test_delete_reg_files_writes_deleted_ids_json(client: TestClient) -> None:
    """删除的 booru ID（= filename stem）追加到 reg/.deleted_ids.json，
    增量补足时 builder 读这个文件做 exclude。"""
    pid, vid = _make(client)
    rdir = _seed_reg(
        client, pid, vid, "5_concept",
        {"42.png": ["a"], "99.png": ["b"]},
    )

    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": ["5_concept/42.png", "5_concept/99.png"]},
    )
    assert r.status_code == 200
    p = rdir / ".deleted_ids.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data) == {"42", "99"}


def test_delete_reg_files_rejects_path_traversal(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_reg(client, pid, vid, "5_concept", {"100.png": []})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": ["../train/secret.png"]},
    )
    # 路径越界 → safe_join 400
    assert r.status_code == 400


def test_delete_reg_files_empty_list_400(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": []},
    )
    assert r.status_code == 400


def test_delete_reg_files_silently_skips_missing(client: TestClient) -> None:
    """路径合法但文件不存在（已被别处删）→ 不算错，count 体现实际删的。"""
    pid, vid = _make(client)
    rdir = _seed_reg(client, pid, vid, "5_concept", {"100.png": []})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": ["5_concept/100.png", "5_concept/ghost.png"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["deleted"] == ["5_concept/100.png"]
    # ghost 不影响 .deleted_ids.json
    data = json.loads((rdir / ".deleted_ids.json").read_text(encoding="utf-8"))
    assert data == ["100"]


# ---------------------------------------------------------------------------
# A3 — auto_tag_kind 校验
# ---------------------------------------------------------------------------


def test_start_reg_build_default_auto_tag_kind_is_wd14(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"a.png": ["x"]})
    # 不传 auto_tag_kind → 默认 wd14
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    assert (job.get("params_decoded") or {}).get("auto_tag_kind") == "wd14"


def test_start_reg_build_accepts_cltagger(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"a.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"auto_tag_kind": "cltagger"},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    assert (job.get("params_decoded") or {}).get("auto_tag_kind") == "cltagger"


def test_start_reg_build_rejects_llm_auto_tag_kind(client: TestClient) -> None:
    """LLM 在底层 VALID_TAGGER_NAMES 里，但 reg 路径本轮只暴露 wd14/cltagger
    （reg 图量大，LLM 慢/贵），422 兜底防 contributor 误传。"""
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"a.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"auto_tag_kind": "llm"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# A4 — POST /reg/dedup-purge
# ---------------------------------------------------------------------------


def test_dedup_purge_empty_reg_returns_zero(client: TestClient) -> None:
    pid, vid = _make(client)
    # 没图，端点要能返回 0 而不是 500
    r = client.post(f"/api/projects/{pid}/versions/{vid}/reg/dedup-purge")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scanned"] == 0
    assert body["groups"] == 0
    assert body["count"] == 0


def test_dedup_purge_no_duplicates_keeps_all(client: TestClient) -> None:
    """完全不同的两张图 → 无 group → 不删。"""
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    rdir = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg" / "5_concept"
    rdir.mkdir(parents=True, exist_ok=True)
    # 两张明显不同的图（不同尺寸 + 不同颜色）
    Image.new("RGB", (64, 64), (255, 0, 0)).save(rdir / "1.png", "PNG")
    Image.new("RGB", (96, 128), (0, 255, 0)).save(rdir / "2.png", "PNG")

    r = client.post(f"/api/projects/{pid}/versions/{vid}/reg/dedup-purge")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scanned"] == 2
    assert body["count"] == 0
    # 两张图都还在
    assert (rdir / "1.png").exists()
    assert (rdir / "2.png").exists()


def test_dedup_purge_deletes_identical_copies(client: TestClient) -> None:
    """两张完全相同（同尺寸 + 同像素）的图 → 应该归到同一组 → 删一张 +
    写 .deleted_ids.json + meta.actual_count 递减。"""
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    base = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg"
    rdir = base / "5_concept"
    rdir.mkdir(parents=True, exist_ok=True)
    # 同样的图保存两次（不同文件名，像素一致）
    img = Image.new("RGB", (128, 128), (255, 128, 64))
    img.save(rdir / "100.png", "PNG")
    img.save(rdir / "200.png", "PNG")
    # meta：actual_count=2
    meta = reg_builder.RegMeta(
        generated_at=0.0, based_on_version="v1", api_source="gelbooru",
        target_count=2, actual_count=2, source_tags=[], excluded_tags=[],
        blacklist_tags=[], failed_tags=[], train_tag_distribution={},
        auto_tagged=False,
    )
    reg_builder.write_meta(base, meta)

    r = client.post(f"/api/projects/{pid}/versions/{vid}/reg/dedup-purge")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scanned"] == 2
    assert body["groups"] >= 1
    assert body["count"] == 1
    # 只剩一张
    remaining = sorted(p.name for p in rdir.glob("*.png"))
    assert len(remaining) == 1
    # .deleted_ids.json 写入了被删那张的 stem
    p = base / ".deleted_ids.json"
    assert p.exists()
    deleted_stems = set(json.loads(p.read_text(encoding="utf-8")))
    # group[0] 保留，剩下的删 — 我们不强保证 keep 是 100 还是 200，
    # 只校验"恰好其中一个"被记进 deleted_ids
    assert deleted_stems <= {"100", "200"}
    assert len(deleted_stems) == 1
    # meta.actual_count 递减
    m2 = reg_builder.read_meta(base)
    assert m2.actual_count == 1
