"""POST /api/generate/save 写 PNG metadata + sidecar、GET /api/generate/disk/history
扫描、GET /api/generate/disk/image/* 静态返回的覆盖测试。"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


def _png_bytes(color: tuple[int, int, int] = (0, 0, 0), size: tuple[int, int] = (8, 8)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _params(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "mode": "single",
        "prompts": ["a"],
        "negative_prompt": "",
        "width": 8,
        "height": 8,
        "steps": 5,
        "cfg_scale": 4.0,
        "count": 1,
        "seed": 7,
        "lora_configs": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    from studio.api.routers import generate as _gen

    test_dir = tmp_path / "test"
    monkeypatch.setattr(_gen, "TEST_IMAGES_DIR", test_dir)

    class _FakeGenCfg:
        save_test_images = True

    class _FakeSecrets:
        generate = _FakeGenCfg()

    monkeypatch.setattr(_gen.secrets, "load", lambda: _FakeSecrets())

    app = FastAPI()
    app.include_router(_gen.router)
    return TestClient(app), test_dir


def test_save_writes_png_metadata_and_sidecar(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params(seed=42))},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    saved = Path(data["path"])
    assert saved.exists()
    sidecar = saved.with_suffix(".json")
    assert sidecar.exists()
    sidecar_obj = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar_obj["mode"] == "single"
    assert sidecar_obj["schema_version"] == 1
    assert sidecar_obj["filename"] == saved.name
    assert sidecar_obj["params"]["seed"] == 42

    # PNG tEXt 注入：再读图应该带 anima_params
    img = Image.open(saved)
    img.load()
    assert "anima_params" in img.text
    parsed = json.loads(img.text["anima_params"])
    assert parsed["seed"] == 42


def test_save_without_params_skips_sidecar_and_metadata(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single"},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["sidecar"] is None
    saved = Path(data["path"])
    assert not saved.with_suffix(".json").exists()
    img = Image.open(saved)
    img.load()
    assert "anima_params" not in img.text


def test_save_rejects_invalid_params_json(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": "not-json{"},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400


def test_save_rejects_non_object_params(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": "[1, 2, 3]"},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400


def test_disk_history_lists_entries_with_sidecar(client) -> None:
    tc, _ = client
    for seed in (1, 2, 3):
        tc.post(
            "/api/generate/save",
            data={"mode": "single", "params": json.dumps(_params(seed=seed))},
            files={"image": (f"{seed}.png", _png_bytes(), "image/png")},
        )

    r = tc.get("/api/generate/disk/history")
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert len(entries) == 3
    # 按 created_at desc
    for a, b in zip(entries, entries[1:]):
        assert a["created_at"] >= b["created_at"]
    seeds = sorted(e["params"]["seed"] for e in entries)
    assert seeds == [1, 2, 3]
    # url 指向 disk-image 接口
    assert all(e["url"].startswith("/api/generate/disk/image/") for e in entries)
    # id 稳定且唯一
    assert len({e["id"] for e in entries}) == 3


def test_disk_history_skips_entry_without_sidecar(client) -> None:
    """没有 sidecar 的图（老数据 / 客户端没传 params）不入列表。"""
    tc, test_dir = client
    single_dir = test_dir / "2026-01-01" / "single"
    single_dir.mkdir(parents=True)
    (single_dir / "image_0.png").write_bytes(_png_bytes())

    r = tc.get("/api/generate/disk/history")
    assert r.status_code == 200
    assert r.json()["entries"] == []


def test_disk_history_skips_sidecar_without_image(client) -> None:
    """sidecar 留着但图被删 → 不入列表（避免历史栏点 404）。"""
    tc, test_dir = client
    single_dir = test_dir / "2026-01-01" / "single"
    single_dir.mkdir(parents=True)
    (single_dir / "image_0.json").write_text(
        json.dumps({"schema_version": 1, "mode": "single", "created_at": 0,
                    "filename": "image_0.png", "params": _params()}),
        encoding="utf-8",
    )

    r = tc.get("/api/generate/disk/history")
    assert r.status_code == 200
    assert r.json()["entries"] == []


def test_disk_image_serves_saved_file(client) -> None:
    tc, _ = client
    save_resp = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params(seed=99))},
        files={"image": ("a.png", _png_bytes((255, 0, 0)), "image/png")},
    )
    assert save_resp.status_code == 200

    listing = tc.get("/api/generate/disk/history").json()["entries"]
    assert len(listing) == 1
    url = listing[0]["url"]

    r = tc.get(url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 0


def test_disk_image_validates_inputs(client) -> None:
    tc, _ = client
    # 非法 date
    assert tc.get("/api/generate/disk/image/not-a-date/single/image_0.png").status_code == 400
    # 非法 mode
    assert tc.get("/api/generate/disk/image/2026-01-01/bad/image_0.png").status_code == 400
    # 非 png 扩展
    assert tc.get("/api/generate/disk/image/2026-01-01/single/image_0.jpg").status_code == 400
    # 文件不存在
    assert tc.get("/api/generate/disk/image/2026-01-01/single/image_0.png").status_code == 404


def test_save_disabled_returns_403(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings.save_test_images=False → 403。"""
    from studio.api.routers import generate as _gen

    monkeypatch.setattr(_gen, "TEST_IMAGES_DIR", tmp_path / "test")

    class _FakeGenCfg:
        save_test_images = False

    class _FakeSecrets:
        generate = _FakeGenCfg()

    monkeypatch.setattr(_gen.secrets, "load", lambda: _FakeSecrets())
    app = FastAPI()
    app.include_router(_gen.router)
    tc = TestClient(app)

    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 403
