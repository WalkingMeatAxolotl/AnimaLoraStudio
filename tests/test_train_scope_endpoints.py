"""ADR 0010 — train-scope preprocess endpoint smoke tests（PR-3 step 2）。

老 endpoint 测试在 test_preprocess_endpoints.py / test_curation_endpoints.py，
本文件只覆盖新 `/api/projects/{pid}/versions/{vid}/preprocess/*` 路径。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from studio import db, secrets, server
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.preprocess import head_mask, masks as train_masks
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services import models as model_downloader
from studio.infrastructure import paths as studio_paths


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setattr(studio_paths, "TASKS_DIR", tmp_path / "tasks")

    from studio.services.models import paths as _paths
    models_root = tmp_path / "models"
    monkeypatch.setattr(_paths, "models_root", lambda: models_root)
    weight = model_downloader.upscaler_target("4x-AnimeSharp")
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"dummy-weights")
    return {"db": dbfile}


class _StubSupervisor:
    def cancel_job(self, _jid: int) -> bool:
        return True


@pytest.fixture
def client(isolated) -> TestClient:
    server.app.state.supervisor = _StubSupervisor()
    return TestClient(server.app)


def _make_pv(client: TestClient) -> tuple[dict, dict]:
    p = client.post(
        "/api/projects", json={"title": "P", "initial_version_label": "v1"}
    ).json()
    with db.connection_for() as conn:
        v = versions.list_versions(conn, project_id=p["id"])[0]
    return p, v


def _train_sub(p: dict, label: str = "v1", folder: str = "1_data") -> Path:
    d = projects.project_dir(p["id"], p["slug"]) / "versions" / label / "train" / folder
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_png(path: Path, size: tuple[int, int] = (40, 40)) -> None:
    Image.new("RGB", size, "red").save(path, "PNG")


# ---------------------------------------------------------------------------
# 404 path validation
# ---------------------------------------------------------------------------


def test_404_for_unknown_project(client: TestClient) -> None:
    resp = client.get("/api/projects/99999/versions/1/preprocess/files")
    assert resp.status_code == 404


def test_404_for_unknown_version(client: TestClient) -> None:
    p, _ = _make_pv(client)
    resp = client.get(f"/api/projects/{p['id']}/versions/99999/preprocess/files")
    assert resp.status_code == 404


def test_404_for_version_belonging_to_other_project(client: TestClient) -> None:
    p1, _ = _make_pv(client)
    p2 = client.post(
        "/api/projects", json={"title": "P2", "initial_version_label": "v1"}
    ).json()
    with db.connection_for() as conn:
        v2 = versions.list_versions(conn, project_id=p2["id"])[0]
    resp = client.get(
        f"/api/projects/{p1['id']}/versions/{v2['id']}/preprocess/files"
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# list_preprocess_files_train
# ---------------------------------------------------------------------------


def test_files_endpoint_returns_train_images_and_summary(client: TestClient) -> None:
    p, v = _make_pv(client)
    sub = _train_sub(p)
    _write_png(sub / "X.png")
    _write_png(sub / "Y.png")
    preprocess_manifest.train_add_processed(
        projects.project_dir(p["id"], p["slug"]),
        v["label"], "1_data/X.png", {"origin": "X.jpg"},
    )

    resp = client.get(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/files"
    )
    assert resp.status_code == 200
    body = resp.json()
    names = sorted(it["name"] for it in body["images"])
    assert names == ["1_data/X.png", "1_data/Y.png"]
    assert body["summary"]["image_count"] == 2


# ---------------------------------------------------------------------------
# preprocess_status_train
# ---------------------------------------------------------------------------


def test_status_endpoint_returns_null_job_when_none(client: TestClient) -> None:
    p, v = _make_pv(client)
    resp = client.get(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/status"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["job"] is None
    assert body["log_tail"] == ""
    assert body["summary"]["image_count"] == 0


# ---------------------------------------------------------------------------
# start_preprocess_train
# ---------------------------------------------------------------------------


def test_start_endpoint_creates_job_with_version_id(client: TestClient) -> None:
    p, v = _make_pv(client)
    resp = client.post(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/start",
        json={"mode": "all"},
    )
    assert resp.status_code == 200
    job = resp.json()
    assert job["version_id"] == v["id"]
    assert job["kind"] == "preprocess"


def test_start_endpoint_rejects_unknown_mode(client: TestClient) -> None:
    p, v = _make_pv(client)
    resp = client.post(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/start",
        json={"mode": "bogus"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# duplicate_removed_train
# ---------------------------------------------------------------------------


def test_duplicates_removed_endpoint(client: TestClient) -> None:
    p, v = _make_pv(client)
    sub = _train_sub(p)
    _write_png(sub / "dup.png")
    preprocess_manifest.train_mark_duplicate_removed(
        projects.project_dir(p["id"], p["slug"]),
        v["label"], ["1_data/dup.png"],
    )

    resp = client.get(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/duplicates/removed"
    )
    assert resp.status_code == 200
    body = resp.json()
    names = [it["name"] for it in body["images"]]
    assert names == ["1_data/dup.png"]


# ---------------------------------------------------------------------------
# crop workspace + crop start
# ---------------------------------------------------------------------------


def test_crop_workspace_endpoint(client: TestClient) -> None:
    p, v = _make_pv(client)
    sub = _train_sub(p)
    _write_png(sub / "A.png", (30, 30))

    resp = client.get(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/crop/workspace"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["images"]) == 1
    assert body["images"][0]["name"] == "1_data/A.png"
    assert body["images"][0]["w"] == 30


def test_crop_start_endpoint_creates_job(client: TestClient) -> None:
    p, v = _make_pv(client)
    resp = client.post(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/crop",
        json={"crops": {"1_data/X.png": [
            {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}
        ]}},
    )
    assert resp.status_code == 200
    job = resp.json()
    assert job["version_id"] == v["id"]


# ---------------------------------------------------------------------------
# automatic head-mask proposal / apply / undo
# ---------------------------------------------------------------------------


def test_head_mask_detect_defaults_and_missing_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    p, v = _make_pv(client)
    url = f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/head-mask/detect"
    monkeypatch.setattr(
        model_downloader, "head_detector_status",
        lambda: {"exists": False, "valid": False, "size": 0},
    )
    missing = client.post(url, json={})
    assert missing.status_code == 409
    assert missing.json()["error"]["code"] == "preprocess.head_mask_model_missing"

    monkeypatch.setattr(
        model_downloader, "head_detector_status",
        lambda: {"exists": True, "valid": True},
    )
    created = client.post(url, json={})
    assert created.status_code == 200
    params = created.json()["params_decoded"]
    assert params == {
        "stage": "head_mask",
        "scope": "all",
        "confidence": 0.413,
        "iou_threshold": 0.7,
        "padding_ratio": 0.1,
        "feather_ratio": 0.03,
    }


def test_head_mask_proposal_apply_and_undo_endpoints(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    p, v = _make_pv(client)
    sub = _train_sub(p)
    image_path = sub / "A.png"
    _write_png(image_path, (100, 80))
    monkeypatch.setattr(
        model_downloader, "head_detector_status",
        lambda: {"exists": True, "valid": True},
    )
    base = f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/head-mask"
    job = client.post(base + "/detect", json={"scope": "selected", "filenames": ["1_data/A.png"]}).json()
    proposal = head_mask.make_image_proposal(
        "1_data/A.png", image_path, (100, 80),
        [{"score": 0.9, "box": [20, 10, 50, 40]}],
        padding_ratio=0.1, feather_ratio=0.03,
    )
    head_mask.write_result(job["id"], head_mask.new_result(
        job["id"], confidence=0.413, iou_threshold=0.7,
        padding_ratio=0.1, feather_ratio=0.03,
        provider="CPUExecutionProvider", images=[proposal],
    ))

    proposals = client.get(base + f"/proposals/{job['id']}")
    assert proposals.status_code == 200
    assert proposals.json()["stale_count"] == 0
    assert proposals.json()["undo_available"] is False

    region_id = proposal["regions"][0]["id"]
    applied = client.post(base + "/apply", json={
        "job_id": job["id"],
        "selections": {"1_data/A.png": [region_id]},
    })
    assert applied.status_code == 200
    assert applied.json()["applied"] == 1
    mask_path = train_masks.mask_path_for(sub.parent, "1_data/A.png")
    assert mask_path.is_file()

    after = client.get(base + f"/proposals/{job['id']}").json()
    assert after["undo_available"] is True
    undone = client.post(base + "/undo", json={"job_id": job["id"]})
    assert undone.status_code == 200
    assert undone.json()["undone"] == 1
    assert not mask_path.exists()


def test_head_mask_rejects_invalid_parameters_and_foreign_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    p1, v1 = _make_pv(client)
    p2, _v2 = _make_pv(client)
    monkeypatch.setattr(
        model_downloader, "head_detector_status",
        lambda: {"exists": True, "valid": True},
    )
    base1 = f"/api/projects/{p1['id']}/versions/{v1['id']}/preprocess/head-mask"
    bad = client.post(base1 + "/detect", json={"confidence": 2})
    assert bad.status_code == 422
    job = client.post(base1 + "/detect", json={}).json()
    foreign = client.get(
        f"/api/projects/{p2['id']}/versions/{v1['id']}/preprocess/head-mask/proposals/{job['id']}"
    )
    assert foreign.status_code == 404


# ---------------------------------------------------------------------------
# reset (clear_all)
# ---------------------------------------------------------------------------


def test_reset_endpoint_clears_manifest_keeps_files(client: TestClient) -> None:
    p, v = _make_pv(client)
    sub = _train_sub(p)
    _write_png(sub / "X.png")
    pdir = projects.project_dir(p["id"], p["slug"])
    preprocess_manifest.train_add_processed(
        pdir, v["label"], "1_data/X.png", {"origin": "X.jpg"},
    )

    resp = client.post(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/files/reset"
    )
    assert resp.status_code == 200
    # manifest 空，物理文件保留
    m = preprocess_manifest.train_load(pdir, v["label"])
    assert m["images"] == {}
    assert (sub / "X.png").exists()


# ---------------------------------------------------------------------------
# restore — 3 组返回
# ---------------------------------------------------------------------------


def test_restore_endpoint_copies_from_download(client: TestClient) -> None:
    p, v = _make_pv(client)
    pdir = projects.project_dir(p["id"], p["slug"])
    sub = _train_sub(p)
    _write_png(sub / "X.png")  # 假设上调过
    # download/X.jpg 是 origin
    (pdir / "download").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), "blue").save(pdir / "download" / "X.jpg", "JPEG")
    preprocess_manifest.train_add_processed(
        pdir, v["label"], "1_data/X.png", {"origin": "X.jpg"},
    )

    resp = client.post(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/files/restore",
        json={"names": ["1_data/X.png"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["restored"] == ["1_data/X.png"]
    assert body["missing"] == []
    assert body["no_origin"] == []


def test_restore_endpoint_no_origin_when_download_missing(client: TestClient) -> None:
    p, v = _make_pv(client)
    pdir = projects.project_dir(p["id"], p["slug"])
    sub = _train_sub(p)
    _write_png(sub / "Y.png")
    preprocess_manifest.train_add_processed(
        pdir, v["label"], "1_data/Y.png", {"origin": "Y.jpg"},
    )
    # 不创建 download/Y.jpg
    resp = client.post(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/files/restore",
        json={"names": ["1_data/Y.png"]},
    )
    body = resp.json()
    assert body["no_origin"] == ["1_data/Y.png"]


def test_restore_endpoint_empty_names_returns_three_groups(client: TestClient) -> None:
    p, v = _make_pv(client)
    resp = client.post(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/files/restore",
        json={"names": []},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"restored": [], "missing": [], "no_origin": []}


# ---------------------------------------------------------------------------
# duplicates scan / apply
# ---------------------------------------------------------------------------


def test_duplicates_scan_endpoint_returns_train_target(client: TestClient) -> None:
    p, v = _make_pv(client)
    sub = _train_sub(p)
    _write_png(sub / "X.png")
    _write_png(sub / "Y.png")

    resp = client.post(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/duplicates/scan",
        json={},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["target"] == "train"
    assert body["total_images"] == 2


def test_duplicates_apply_endpoint_marks_manifest(client: TestClient) -> None:
    p, v = _make_pv(client)
    sub = _train_sub(p)
    _write_png(sub / "X.png")
    resp = client.post(
        f"/api/projects/{p['id']}/versions/{v['id']}/preprocess/duplicates/apply",
        json={"names": ["1_data/X.png"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["removed"] == ["1_data/X.png"]
    # 物理文件已删（tombstone 只在 manifest）
    assert not (sub / "X.png").exists()
