"""/api/projects/{pid}/preprocess/* — start / status / files / restore / thumb。

ADR 0004：状态走 manifest，restore 替代 delete 语义。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, secrets, server
from studio.services.preprocess import core as preprocess_svc
from studio.services.projects import jobs as project_jobs, projects
from studio.services import models as model_downloader
from studio.services.preprocess import manifest as preprocess_manifest


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")

    # 让 upscaler_target 指到 tmp 内的"假权重"，避免端点检查 409。
    # 必须 patch `paths.models_root` 而不是 `model_downloader.models_root` —
    # 后者只是 __init__ 的 re-export，而 upscaler_target 在 paths.py 内部用
    # 本模块的 models_root 调用；patch re-export 不会改到调用点，会让 dummy
    # 权重写到 REPO_ROOT/models/upscalers/ 把真模型干掉。
    from studio.services.models import paths as _paths
    models_root = tmp_path / "models"
    monkeypatch.setattr(_paths, "models_root", lambda: models_root)
    weight = model_downloader.upscaler_target("4x-AnimeSharp")
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"dummy-weights")

    return {"db": dbfile, "weight_path": weight}


class _StubSupervisor:
    def cancel_job(self, _jid: int) -> bool:
        return True


@pytest.fixture
def client(isolated) -> TestClient:
    server.app.state.supervisor = _StubSupervisor()
    return TestClient(server.app)


def _make_project(client: TestClient) -> dict:
    return client.post(
        "/api/projects", json={"title": "P", "initial_version_label": None}
    ).json()


def _seed_download_image(p: dict, name: str = "a.png") -> Path:
    """造一张能通过 IMAGE_EXTS 检查的占位文件（端点只看扩展名+文件存在）。"""
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    pdir.mkdir(parents=True, exist_ok=True)
    f = pdir / name
    f.write_bytes(b"fake-image-bytes")
    return f


def _seed_processed(p: dict, product_name: str, meta: dict) -> Path:
    """造一张产物：preprocess/{product_name} + manifest entry。"""
    pdir = projects.project_dir(p["id"], p["slug"])
    pre = pdir / "preprocess"
    pre.mkdir(parents=True, exist_ok=True)
    (pre / product_name).write_bytes(b"upscaled")
    preprocess_manifest.add_processed(pdir, product_name, meta)
    return pre / product_name


# ---------------------------------------------------------------------------
# start_preprocess
# ---------------------------------------------------------------------------


def test_start_preprocess_creates_pending_job(client: TestClient) -> None:
    p = _make_project(client)
    _seed_download_image(p, "a.png")

    resp = client.post(
        f"/api/projects/{p['id']}/preprocess/start",
        json={"mode": "all", "tile_size": 256},
    )
    assert resp.status_code == 200, resp.text
    job = resp.json()
    assert job["kind"] == "preprocess"
    assert job["status"] == "pending"
    # ADR-0007 PR-5: project 无 stage；preprocess 状态由 job 派生


def test_start_preprocess_rejects_unknown_mode(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/preprocess/start", json={"mode": "weird"}
    )
    assert resp.status_code == 400


def test_start_preprocess_rejects_unknown_model(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/preprocess/start",
        json={"mode": "all", "model": "FakeModelX"},
    )
    assert resp.status_code == 400


def test_start_preprocess_requires_weights_downloaded(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    """权重不存在 → 409，引导用户去下载。"""
    isolated["weight_path"].unlink()
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/preprocess/start", json={"mode": "all"}
    )
    assert resp.status_code == 409
    assert "未下载" in resp.json()["detail"]


def test_start_preprocess_selected_requires_names(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/preprocess/start",
        json={"mode": "selected", "names": []},
    )
    assert resp.status_code == 400


def test_start_preprocess_unknown_project(client: TestClient) -> None:
    resp = client.post(
        "/api/projects/9999/preprocess/start", json={"mode": "all"}
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# status / files
# ---------------------------------------------------------------------------


def test_status_no_job_returns_empty(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.get(f"/api/projects/{p['id']}/preprocess/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job"] is None
    assert body["log_tail"] == ""
    assert body["summary"] == {"image_count": 0}


def test_list_files_returns_processed_and_pending(client: TestClient) -> None:
    p = _make_project(client)
    _seed_download_image(p, "a.png")
    _seed_download_image(p, "b.png")
    # 模拟一张已处理：preprocess/a.png 存在 + manifest entry
    _seed_processed(p, "a.png", {
        "source": "a.png", "scale": 4, "model": "4x-AnimeSharp",
    })

    resp = client.get(f"/api/projects/{p['id']}/preprocess/files")
    assert resp.status_code == 200
    body = resp.json()
    assert {it["name"] for it in body["processed"]} == {"a.png"}
    assert {it["name"] for it in body["pending"]} == {"b.png"}
    assert body["summary"]["image_count"] == 2


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------


def test_restore_preprocess_files_removes_product(client: TestClient) -> None:
    p = _make_project(client)
    _seed_download_image(p, "a.png")
    png = _seed_processed(p, "a.png", {"source": "a.png", "model": "X"})

    resp = client.post(
        f"/api/projects/{p['id']}/preprocess/files/restore",
        json={"names": ["a.png", "ghost.png"]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"restored": ["a.png"], "missing": ["ghost.png"]}
    assert not png.exists()


def test_restore_preprocess_rejects_traversal(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/preprocess/files/restore",
        json={"names": ["../etc/passwd"]},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# crop 端点
# ---------------------------------------------------------------------------


def test_start_crop_creates_pending_job(client: TestClient) -> None:
    """POST /preprocess/crop 返回 pending job，params.stage=crop。"""
    p = _make_project(client)
    _seed_download_image(p, "a.png")
    resp = client.post(
        f"/api/projects/{p['id']}/preprocess/crop",
        json={"crops": {"a.png": [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}]}},
    )
    assert resp.status_code == 200, resp.text
    job = resp.json()
    assert job["kind"] == "preprocess"
    assert job["status"] == "pending"
    assert job["params_decoded"]["stage"] == "crop"
    # ADR-0007 PR-5: project 无 stage（params.stage="crop" 是 job 内部字段，无关）


def test_start_crop_rejects_empty(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/preprocess/crop", json={"crops": {}}
    )
    assert resp.status_code == 400


def test_start_crop_rejects_unknown_project(client: TestClient) -> None:
    resp = client.post(
        "/api/projects/9999/preprocess/crop",
        json={"crops": {"a.png": [{"x": 0, "y": 0, "w": 0.5, "h": 0.5}]}},
    )
    assert resp.status_code == 404


def test_start_crop_rejects_traversal(client: TestClient) -> None:
    p = _make_project(client)
    resp = client.post(
        f"/api/projects/{p['id']}/preprocess/crop",
        json={
            "crops": {"../etc/passwd": [{"x": 0, "y": 0, "w": 0.5, "h": 0.5}]}
        },
    )
    assert resp.status_code == 400


def test_crop_workspace_returns_dimensions(client: TestClient) -> None:
    """workspace endpoint 返回 download + preprocess 的合并列表 + 每图像素尺寸。"""
    from PIL import Image
    p = _make_project(client)
    pdir = projects.project_dir(p["id"], p["slug"])
    (pdir / "download").mkdir(parents=True, exist_ok=True)
    (pdir / "preprocess").mkdir(parents=True, exist_ok=True)
    # download/A.png 未处理；download/B.jpg + preprocess/B.png 已处理
    Image.new("RGB", (320, 240), (255, 0, 0)).save(pdir / "download" / "A.png")
    Image.new("RGB", (200, 150), (0, 255, 0)).save(pdir / "download" / "B.jpg")
    Image.new("RGB", (800, 600), (0, 0, 255)).save(pdir / "preprocess" / "B.png")
    preprocess_manifest.add_processed(pdir, "B.png", {"source": "B.jpg"})

    resp = client.get(f"/api/projects/{p['id']}/preprocess/crop/workspace")
    assert resp.status_code == 200, resp.text
    images = {it["name"]: it for it in resp.json()["images"]}
    assert set(images.keys()) == {"A.png", "B.png"}
    assert images["A.png"]["w"] == 320
    assert images["A.png"]["h"] == 240
    assert images["A.png"]["processed"] is False
    assert images["A.png"]["source"] == "A.png"
    # B.png 是 preprocess 产物，origin 是 download/B.jpg
    assert images["B.png"]["w"] == 800
    assert images["B.png"]["h"] == 600
    assert images["B.png"]["processed"] is True
    assert images["B.png"]["source"] == "B.jpg"


# ---------------------------------------------------------------------------
# thumb 端点：ADR 0004 自动 resolve（已处理 → preprocess/，未处理 → download/）
# ---------------------------------------------------------------------------


def test_thumb_resolves_to_download_for_unprocessed(client: TestClient) -> None:
    """没在 manifest 里 → 走 download/{name}。"""
    p = _make_project(client)
    # 写一个 PNG 内容（Pillow 能读的最小 PNG）
    from PIL import Image
    download_dir = projects.project_dir(p["id"], p["slug"]) / "download"
    download_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(download_dir / "a.png")

    resp = client.get(f"/api/projects/{p['id']}/thumb?name=a.png&size=8")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")


def test_thumb_resolves_to_preprocess_when_processed(client: TestClient) -> None:
    """manifest kind=processed → 走 preprocess/{stem}.png（即使 URL 给的是 .jpg）。"""
    p = _make_project(client)
    from PIL import Image
    pdir = projects.project_dir(p["id"], p["slug"])
    download_dir = pdir / "download"
    download_dir.mkdir(parents=True, exist_ok=True)
    # download 给 jpg，preprocess 给 png（产物固定 png）
    Image.new("RGB", (8, 8), (255, 0, 0)).save(download_dir / "a.jpg")
    pre_dir = pdir / "preprocess"
    pre_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (0, 255, 0)).save(pre_dir / "a.png")
    preprocess_manifest.add_processed(pdir, "a.png", {"source": "a.jpg"})

    # 前端按原 download 名 a.jpg 请求 —— 后端应 resolve 到 preprocess/a.png
    resp = client.get(f"/api/projects/{p['id']}/thumb?name=a.jpg&size=8")
    assert resp.status_code == 200


def test_thumb_handles_multi_crop_derivative_name_via_download_bucket(client: TestClient) -> None:
    """筛选页 list_download 展开 multi-crop 派生（X_c0.png）后，缩略图请求经
    bucket=download 走过来。原 resolve_origin 按 origin 找不到（origin 是
    X.png），endpoint 兜底到 preprocess/{name} 直读。"""
    from PIL import Image
    p = _make_project(client)
    pdir = projects.project_dir(p["id"], p["slug"])
    (pdir / "preprocess").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(pdir / "preprocess" / "X_c0.png")
    preprocess_manifest.replace_with_crops(
        pdir,
        source_name="X.png",
        outputs=[
            {"name": "X_c0.png", "origin": "X.png", "size": 1, "mtime": 1.0},
        ],
    )
    # 用 bucket=download 默认 + 派生名直接请求；endpoint 应兜底 preprocess
    resp = client.get(f"/api/projects/{p['id']}/thumb?name=X_c0.png&size=8")
    assert resp.status_code == 200, resp.text
