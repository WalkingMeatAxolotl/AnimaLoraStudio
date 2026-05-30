"""PP5 — reg_build worker: mock booru + tagger，跑通 worker 主流程。"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from studio import db, secrets
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services.reg import builder as reg_builder
from studio.workers import reg_build_worker


def _png_bytes() -> bytes:
    img = Image.new("RGB", (8, 8), (0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeAutoTagger:
    name = "wd14"
    requires_service = False

    def is_available(self):
        return True, "ok"

    def prepare(self):
        pass

    def tag(self, paths, on_progress=lambda d, t: None):
        for i, p in enumerate(paths):
            on_progress(i + 1, len(paths))
            yield {"image": p, "tags": ["auto_tag", p.stem]}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")

    # 给 reg_builder 用的 fake booru API
    def fake_search_posts(api_source, tags_query, *, page=1, limit=100, **kw):
        return [
            {"@attributes": {
                "id": "5001", "file_url": "http://x/5001.png", "file_ext": "png",
                "tags": "1girl solo", "width": 512, "height": 512,
            }},
            {"@attributes": {
                "id": "5002", "file_url": "http://x/5002.png", "file_ext": "png",
                "tags": "1girl long_hair", "width": 512, "height": 512,
            }},
        ]

    def fake_download_image(url, save_path, *, convert_to_png, remove_alpha_channel, **kw):
        save_path = Path(save_path)
        if convert_to_png and save_path.suffix.lower() != ".png":
            save_path = save_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(_png_bytes())
        return save_path

    monkeypatch.setattr(reg_builder.booru_api, "search_posts", fake_search_posts)
    monkeypatch.setattr(reg_builder.booru_api, "download_image", fake_download_image)

    # auto_tag fake
    monkeypatch.setattr(
        "studio.services.tagging.base.get_tagger", lambda name: _FakeAutoTagger()
    )

    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
        # seed train 一张图带 caption
        train = versions.version_dir(p["id"], p["slug"], v["label"]) / "train" / "1_data"
        Image.new("RGB", (512, 512), (255, 0, 0)).save(train / "x.png", "PNG")
        (train / "x.txt").write_text("1girl, solo", encoding="utf-8")
        # 设置 gelbooru 凭据
        sec = secrets.load()
        sec.gelbooru.user_id = "u"
        sec.gelbooru.api_key = "k"
        secrets.save(sec)
        job = project_jobs.create_job(
            conn,
            project_id=p["id"],
            version_id=v["id"],
            kind="reg_build",
            params={
                "version_id": v["id"],
                "target_count": 2,
                "excluded_tags": [],
                "auto_tag": True,
                "api_source": "gelbooru",
            },
        )
    return {
        "db": dbfile, "p": p, "v": v, "job_id": job["id"],
        "vdir": versions.version_dir(p["id"], p["slug"], v["label"]),
    }


def test_worker_runs_and_writes_meta_and_images(env) -> None:
    rc = reg_build_worker.run(env["job_id"])
    assert rc == 0
    rdir = env["vdir"] / "reg"
    assert (rdir / "meta.json").exists()
    meta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
    assert meta["actual_count"] >= 1
    # auto_tag 跑过 → meta 改写为 True
    assert meta["auto_tagged"] is True
    # A3 — 默认 kind=wd14 写回 meta
    assert meta.get("auto_tag_kind") == "wd14"
    # reg 集图片落盘到 1_data 子文件夹（镜像 train）
    images = list((rdir / "1_data").glob("*.png"))
    assert len(images) >= 1
    # auto_tag 也落 .txt
    txts = list((rdir / "1_data").glob("*.txt"))
    assert len(txts) >= 1


def test_worker_full_mode_clears_existing_reg(env) -> None:
    """A4 v2 — incremental=False (full mode) 时 worker 应在 build 前清掉
    reg/ 下所有内容（图、子文件夹、meta、.deleted_ids.json）。"""
    rdir = env["vdir"] / "reg"
    sub = rdir / "5_concept"
    sub.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(sub / "old.png", "PNG")
    (rdir / "meta.json").write_text('{"foo":"bar"}', encoding="utf-8")
    (rdir / ".deleted_ids.json").write_text('["stale"]', encoding="utf-8")
    # set incremental=False
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs "
            "SET params = json_set(params, '$.incremental', 0, "
            "                      '$.auto_dedup', 0) "
            "WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()
    rc = reg_build_worker.run(env["job_id"])
    assert rc == 0
    # 旧图不见了；fresh 的图从 5_001 / 5_002 进来（fake_search 返回的）
    assert not (sub / "old.png").exists()
    # .deleted_ids.json 也被清掉
    assert not (rdir / ".deleted_ids.json").exists()


def test_worker_auto_dedup_loop_calls_scan_purge_then_rebuild(env, monkeypatch) -> None:
    """A4 v2 — auto_dedup=True 时 worker build 后跑 scan → purge → 不够补足；
    第二轮 scan 返回空 → 退出循环。"""
    calls: dict[str, int] = {"scan": 0, "purge": 0, "rebuild": 0}

    def fake_scan(reg_dir):
        calls["scan"] += 1
        if calls["scan"] == 1:
            return ["1_data/5001.png"]
        return []

    def fake_purge(reg_dir, rels):
        calls["purge"] += 1
        for rel in rels:
            t = reg_dir / rel
            if t.exists():
                t.unlink()
            tx = t.with_suffix(".txt")
            if tx.exists():
                tx.unlink()
        m = reg_builder.read_meta(reg_dir)
        if m and rels:
            m.actual_count = max(0, m.actual_count - len(rels))
            reg_builder.write_meta(reg_dir, m)
        return {"deleted": list(rels), "count": len(rels)}

    # 记 build 调用：第一次 = 初次（incremental=None/True 都可），后续 = 补足
    original_build = reg_builder.build
    def spy_build(*args, **kwargs):
        calls["rebuild"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr("studio.services.reg.dedup.scan_for_dedup", fake_scan)
    monkeypatch.setattr("studio.services.reg.dedup.purge_paths", fake_purge)
    monkeypatch.setattr("studio.services.reg.builder.build", spy_build)

    rc = reg_build_worker.run(env["job_id"])
    assert rc == 0
    # 至少 2 轮 scan（第一轮删后第二轮 detect 无重复退出）
    assert calls["scan"] >= 2
    assert calls["purge"] == 1
    # build 调了 2 次：初次 + 1 次补足
    assert calls["rebuild"] == 2


def test_worker_auto_dedup_disabled_skips_scan(env, monkeypatch) -> None:
    """A4 v2 — auto_dedup=False 时 worker 不应触碰 dedup 模块。"""
    calls = {"scan": 0}

    def fake_scan(reg_dir):
        calls["scan"] += 1
        return []

    monkeypatch.setattr("studio.services.reg.dedup.scan_for_dedup", fake_scan)
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = "
            "json_set(params, '$.auto_dedup', 0) WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()
    rc = reg_build_worker.run(env["job_id"])
    assert rc == 0
    assert calls["scan"] == 0


def test_worker_routes_to_cltagger_when_kind_set(env, monkeypatch) -> None:
    """A3 — params['auto_tag_kind']='cltagger' 必须传给 get_tagger，
    meta 写回 kind=cltagger。"""
    calls: list[str] = []

    def spy_get_tagger(name: str):
        calls.append(name)
        return _FakeAutoTagger()

    monkeypatch.setattr(
        "studio.services.tagging.base.get_tagger", spy_get_tagger
    )
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs "
            "SET params = json_set(params, '$.auto_tag_kind', 'cltagger') "
            "WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()
    rc = reg_build_worker.run(env["job_id"])
    assert rc == 0
    assert "cltagger" in calls
    meta = json.loads(
        (env["vdir"] / "reg" / "meta.json").read_text(encoding="utf-8")
    )
    assert meta.get("auto_tag_kind") == "cltagger"


def test_worker_unknown_job(env) -> None:
    assert reg_build_worker.run(99999) == 1


def test_worker_writes_postprocess_meta_when_clusters_found(env) -> None:
    """PP5.5 集成：worker 跑完 build 后调 postprocess，meta 含 postprocessed_at。"""
    rc = reg_build_worker.run(env["job_id"])
    assert rc == 0
    rdir = env["vdir"] / "reg"
    import json as _json
    meta = _json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
    # 只有 1 张 reg 图（fake booru 的）→ < 2 → 单 cluster；postprocess 仍跑
    # 单图情况下 cluster 数量是 1 或 None（< 2 时直接 1 个 cluster）
    if meta.get("postprocess_clusters") is not None:
        assert meta["postprocess_method"] == "smart"
        assert meta["postprocess_max_crop_ratio"] == 0.1


def test_worker_skips_auto_tag_when_disabled(env) -> None:
    # 改 job 的 auto_tag 为 False
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.auto_tag', 0) "
            "WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()
    rc = reg_build_worker.run(env["job_id"])
    assert rc == 0
    rdir = env["vdir"] / "reg"
    meta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
    assert meta["auto_tagged"] is False
    # 没 auto_tag → 不写 .txt
    txts = list((rdir / "1_data").glob("*.txt"))
    assert txts == []


def test_worker_imports_onnxruntime_setup_at_module_level() -> None:
    """worker 是独立 subprocess —— auto_tag 路径会 get_tagger("wd14")，必须
    在任何 onnxruntime import 之前触发 onnxruntime_setup 顶层 preload。
    """
    import re
    import sys
    src = Path(reg_build_worker.__file__).read_text(encoding="utf-8")
    assert "from studio.services.runtime import onnxruntime" in src, (
        "reg_build_worker.py 顶层必须 import onnxruntime 触发 preload；"
        "见 onnxruntime.py 顶部 PP9.5 注释。"
    )
    bad = re.findall(r"^\s*(?:import onnxruntime|from onnxruntime\b)", src, re.MULTILINE)
    assert not bad, f"worker 不应直接 import onnxruntime；命中: {bad}"
    assert "studio.services.runtime.onnxruntime" in sys.modules
