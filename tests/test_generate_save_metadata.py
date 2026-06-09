"""POST /api/generate/save 写 PNG metadata (anima_params + a1111 parameters)、
GET /api/generate/disk/history 扫 PNG metadata、GET /api/generate/disk/image/*
静态返回的覆盖测试。"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin


def _png_bytes(color: tuple[int, int, int] = (0, 0, 0), size: tuple[int, int] = (8, 8)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _params(**overrides) -> dict:
    """前端 snapshot shape：loras 是 name+ids（无 path），跟 paramsSnapshot.ts 对齐。"""
    base = {
        "schema_version": 1,
        "mode": "single",
        "prompts": ["1girl, anime"],
        "negative_prompt": "blurry",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "cfg_scale": 7.0,
        "count": 1,
        "seed": 7,
        "loras": [],
        "xy_draft": None,
        "dataset_pick": None,
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


def _open_png_text(path: Path) -> dict[str, str]:
    with Image.open(path) as img:
        img.load()
        return dict(img.text)


def test_save_writes_anima_params_text_block(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params(seed=42))},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    saved = Path(r.json()["path"])
    assert saved.exists()

    text = _open_png_text(saved)
    assert "anima_params" in text
    parsed = json.loads(text["anima_params"])
    assert parsed["seed"] == 42
    # server 端 enrich 强制 schema_version=2（即使前端传 1 也覆盖）
    assert parsed["schema_version"] == 2
    # server 端 enrich 补 created_at + mode
    assert parsed["mode"] == "single"
    assert "created_at" in parsed


def test_save_writes_a1111_parameters_text_block(client) -> None:
    """a1111 兼容 `parameters` 块：ComfyUI / WebUI / Civitai 拖图能识别。"""
    tc, _ = client
    p = _params(
        seed=42, steps=20, cfg_scale=7.0, width=1024, height=1024,
        prompts=["1girl, anime"], negative_prompt="blurry",
        loras=[{"name": "my-lora.safetensors", "scale": 0.8,
                "project_id": 12, "version_id": 34}],
    )
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(p)},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    saved = Path(r.json()["path"])

    text = _open_png_text(saved)
    assert "parameters" in text
    a1111 = text["parameters"]
    # 第一行：prompt + <lora:name:scale> 嵌入
    first_line = a1111.split("\n", 1)[0]
    assert "1girl, anime" in first_line
    assert "<lora:my-lora:0.8>" in first_line  # 注意 a1111 语法去 .safetensors
    # 第二行：negative
    assert "Negative prompt: blurry" in a1111
    # 第三行：参数串
    assert "Steps: 20" in a1111
    assert "CFG scale: 7.0" in a1111
    assert "Seed: 42" in a1111
    assert "Size: 1024x1024" in a1111


def test_save_does_not_write_sidecar(client) -> None:
    """sidecar 已砍 —— 同目录不应出现 image_N.json。"""
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    saved = Path(r.json()["path"])
    assert not saved.with_suffix(".json").exists()
    # 返回结构也没 sidecar 字段了
    assert "sidecar" not in r.json()


def test_save_without_params_skips_metadata(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single"},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    saved = Path(r.json()["path"])
    text = _open_png_text(saved)
    assert "anima_params" not in text
    assert "parameters" not in text


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


def test_disk_history_lists_entries_from_png_metadata(client) -> None:
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
    for a, b in zip(entries, entries[1:]):
        assert a["created_at"] >= b["created_at"]
    seeds = sorted(e["params"]["seed"] for e in entries)
    assert seeds == [1, 2, 3]
    assert all(e["image_url"].startswith("/api/generate/disk/image/") for e in entries)
    assert len({e["id"] for e in entries}) == 3


def test_disk_history_skips_png_without_anima_params(client) -> None:
    """没有 anima_params tEXt 块的 PNG（老数据 / 客户端没传 params）不入列表。"""
    tc, test_dir = client
    single_dir = test_dir / "2026-01-01" / "single"
    single_dir.mkdir(parents=True)
    (single_dir / "image_0.png").write_bytes(_png_bytes())

    r = tc.get("/api/generate/disk/history")
    assert r.status_code == 200
    assert r.json()["entries"] == []


def test_disk_history_skips_png_with_only_a1111_block(client) -> None:
    """光有 a1111 parameters 块、没 anima_params 的 PNG 也跳过（避免半 entry）。"""
    tc, test_dir = client
    single_dir = test_dir / "2026-01-01" / "single"
    single_dir.mkdir(parents=True)
    # 手工写一个只含 a1111 块的 PNG
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "fake prompt\nSteps: 1, Sampler: x")
    img = Image.new("RGB", (8, 8))
    img.save(single_dir / "image_0.png", format="PNG", pnginfo=info)

    r = tc.get("/api/generate/disk/history")
    assert r.status_code == 200
    assert r.json()["entries"] == []


def test_disk_history_includes_xy_mode_entries(client) -> None:
    """xy 模式的合成大图也走 disk-history。"""
    tc, _ = client
    tc.post(
        "/api/generate/save",
        data={"mode": "xy", "params": json.dumps(_params(mode="xy", seed=99))},
        files={"image": ("xy.png", _png_bytes(), "image/png")},
    )
    entries = tc.get("/api/generate/disk/history").json()["entries"]
    assert len(entries) == 1
    assert entries[0]["mode"] == "xy"


def test_disk_image_serves_saved_file(client) -> None:
    tc, _ = client
    tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params(seed=99))},
        files={"image": ("a.png", _png_bytes((255, 0, 0)), "image/png")},
    )
    listing = tc.get("/api/generate/disk/history").json()["entries"]
    assert len(listing) == 1
    url = listing[0]["image_url"]

    r = tc.get(url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 0
    # 落盘图加了 strong cache header（决策：内容稳定可强 cache）
    assert "max-age" in r.headers.get("cache-control", "")


def test_disk_image_validates_inputs(client) -> None:
    tc, _ = client
    assert tc.get("/api/generate/disk/image/not-a-date/single/image_0.png").status_code == 400
    assert tc.get("/api/generate/disk/image/2026-01-01/bad/image_0.png").status_code == 400
    assert tc.get("/api/generate/disk/image/2026-01-01/single/image_0.jpg").status_code == 400
    assert tc.get("/api/generate/disk/image/2026-01-01/single/image_0.png").status_code == 404


# ---------------------------------------------------------------------------
# Step 1a/1b 新功能：文件命名 v2 / migrate / thumb / DELETE / path safety
# ---------------------------------------------------------------------------


def test_save_uses_v2_filename(client) -> None:
    """决策 #6：v2 命名 'single image 1.png' / 'xy plot 1.png'，1-based。"""
    tc, _ = client
    r1 = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    r2 = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    r3 = tc.post(
        "/api/generate/save",
        data={"mode": "xy", "params": json.dumps(_params(mode="xy"))},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r1.json()["filename"] == "single image 1.png"
    assert r2.json()["filename"] == "single image 2.png"
    assert r3.json()["filename"] == "xy plot 1.png"


def test_save_sets_task_id_from_form_field(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params()), "task_id": "42"},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    saved = Path(r.json()["path"])
    text = _open_png_text(saved)
    parsed = json.loads(text["anima_params"])
    assert parsed["task_id"] == 42


def test_save_atomic_write_no_tmp_remains(client, tmp_path) -> None:
    """决策 #11：atomic write 不留 .tmp 文件。"""
    tc, test_dir = client
    tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    leftover = list((test_dir).rglob("*.tmp*"))
    assert leftover == [], f"留有 atomic write tmp 文件: {leftover}"


def test_save_xy_skips_a1111_block(client) -> None:
    """决策 #7：XY PNG 不写 a1111 parameters 块（矩阵图单图对应不上）。"""
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "xy", "params": json.dumps(_params(mode="xy"))},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    saved = Path(r.json()["path"])
    text = _open_png_text(saved)
    assert "anima_params" in text
    assert "parameters" not in text


def test_disk_history_migrates_v1_to_v2(client, tmp_path) -> None:
    """决策 #18：v1 PNG（lora_configs[].path）扫到后 migrate 成 v2（loras[].name 无 path）。"""
    tc, test_dir = client
    # 手工写一个 v1 schema PNG
    v1_params = {
        "schema_version": 1,
        "mode": "single",
        "prompts": ["test"],
        "negative_prompt": "",
        "width": 8, "height": 8, "steps": 5, "cfg_scale": 4.0, "count": 1, "seed": 1,
        "lora_configs": [
            {"path": "G:/some/abs/path/my-lora.safetensors", "scale": 0.8,
             "project_id": 12, "version_id": 34}
        ],
    }
    from studio.api.routers import generate as _gen
    raw = _gen._inject_png_metadata(_png_bytes(), v1_params, mode="single")
    single_dir = test_dir / "2026-06-08" / "single"
    single_dir.mkdir(parents=True)
    (single_dir / "image_0.png").write_bytes(raw)  # 故意用 v1 命名 image_N.png

    r = tc.get("/api/generate/disk/history")
    entries = r.json()["entries"]
    assert len(entries) == 1
    params = entries[0]["params"]
    assert params["schema_version"] == 2
    # v1 lora_configs[].path → v2 loras[].name basename（无 path 字段）
    assert "lora_configs" not in params
    assert params["loras"] == [
        {"name": "my-lora.safetensors", "scale": 0.8,
         "project_id": 12, "version_id": 34}
    ]


def test_disk_history_skips_tmp_files(client, tmp_path) -> None:
    """atomic write 半途留下的 .tmp.png 不入历史。"""
    tc, test_dir = client
    single_dir = test_dir / "2026-06-08" / "single"
    single_dir.mkdir(parents=True)
    (single_dir / "single image 1.tmp.png").write_bytes(_png_bytes())
    assert tc.get("/api/generate/disk/history").json()["entries"] == []


def test_disk_thumb_returns_png_with_etag(client) -> None:
    tc, _ = client
    tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    entry = tc.get("/api/generate/disk/history").json()["entries"][0]
    thumb_url = entry["thumb_url"]
    assert "/disk/thumb/" in thumb_url
    r = tc.get(thumb_url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers.get("etag")
    assert "max-age" in r.headers.get("cache-control", "")


def test_disk_thumb_validates_inputs(client) -> None:
    tc, _ = client
    assert tc.get("/api/generate/disk/thumb/not-a-date/single/foo.png").status_code == 400
    assert tc.get("/api/generate/disk/thumb/2026-06-08/bad/foo.png").status_code == 400
    assert tc.get("/api/generate/disk/thumb/2026-06-08/single/foo.png").status_code == 404


def test_disk_delete_removes_file(client, tmp_path) -> None:
    tc, _ = client
    tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    entry = tc.get("/api/generate/disk/history").json()["entries"][0]
    path = Path(entry["path"])
    assert path.is_file()
    # 从 entry 解析出 disk delete URL；后端校验路径
    encoded = entry["image_url"].rsplit("/", 1)[-1]
    delete_url = f"/api/generate/disk/{entry['date']}/{entry['mode']}/{encoded}"
    r = tc.delete(delete_url)
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["noop"] is False
    assert not path.is_file()
    # 二次删 noop=True
    r2 = tc.delete(delete_url)
    assert r2.status_code == 200
    assert r2.json()["noop"] is True


def test_disk_path_traversal_attack_blocked(client) -> None:
    """安全：含 .. 的 path 必须 400。"""
    tc, _ = client
    # FastAPI path param 默认不允许 / —— 但 `..%2F` URL encoded 可能逃过
    for evil in [
        "%2E%2E%2Fsecret.png",        # ../secret.png URL encoded
        "..%5Csecret.png",            # ..\secret.png URL encoded (Windows)
    ]:
        r = tc.get(f"/api/generate/disk/image/2026-06-08/single/{evil}")
        assert r.status_code in (400, 404), f"path={evil} got {r.status_code}"


def test_save_disabled_returns_403(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
