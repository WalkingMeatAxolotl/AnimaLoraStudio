"""/api/projects/{pid}/preprocess/* — start / status / files / restore / thumb。

ADR 0004：状态走 manifest，restore 替代 delete 语义。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, preprocess as preprocess_svc, project_jobs, projects, secrets, server
from studio.services import model_downloader, preprocess_manifest


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(projects, "TRASH_DIR", tmp_path / "_trash")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")

    # 让 upscaler_target 指到 tmp 内的"假权重"，避免端点检查 409
    models_root = tmp_path / "models"
    monkeypatch.setattr(model_downloader, "models_root", lambda: models_root)
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

    # stage 推到 preprocessing
    p2 = client.get(f"/api/projects/{p['id']}").json()
    assert p2["stage"] == "preprocessing"


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
    assert body["summary"] == {
        "download_count": 0, "processed_count": 0, "pending_count": 0,
    }


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
    assert body["summary"]["pending_count"] == 1


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
