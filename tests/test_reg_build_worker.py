"""PP5 — reg_build worker: mock booru + tagger，跑通 worker 主流程。"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from studio import db, project_jobs, projects, secrets, versions
from studio.services import reg_builder
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
    monkeypatch.setattr(projects, "TRASH_DIR", tmp_path / "_trash")
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
        "studio.services.tagger.get_tagger", lambda name: _FakeAutoTagger()
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
    rdir = env["vdir"] / "reg" / "1_general"
    assert (rdir / "meta.json").exists()
    meta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
    assert meta["actual_count"] >= 1
    # auto_tag 跑过 → meta 改写为 True
    assert meta["auto_tagged"] is True
    # reg 集图片落盘到 1_data 子文件夹（镜像 train）
    images = list((rdir / "1_data").glob("*.png"))
    assert len(images) >= 1
    # auto_tag 也落 .txt
    txts = list((rdir / "1_data").glob("*.txt"))
    assert len(txts) >= 1


def test_worker_unknown_job(env) -> None:
    assert reg_build_worker.run(99999) == 1


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
    rdir = env["vdir"] / "reg" / "1_general"
    meta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
    assert meta["auto_tagged"] is False
    # 没 auto_tag → 不写 .txt
    txts = list((rdir / "1_data").glob("*.txt"))
    assert txts == []
