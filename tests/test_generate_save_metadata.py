"""POST /api/generate/save 写 PNG metadata (anima_params + a1111 parameters)、
GET /api/generate/disk/history 扫 PNG metadata、GET /api/generate/disk/image/*
静态返回的覆盖测试。"""
from __future__ import annotations

import hashlib
import json
import re
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
    from studio.services import generation_metadata as _meta

    test_dir = tmp_path / "test"
    monkeypatch.setattr(_gen, "TEST_IMAGES_DIR", test_dir)
    monkeypatch.setattr(
        _meta, "manifest_path", lambda task_id: tmp_path / "tasks" / str(task_id) / _meta.MANIFEST_FILENAME,
    )
    monkeypatch.setattr(_meta, "HASH_CACHE_PATH", tmp_path / ".cache" / "resource-hashes.json")
    _meta._reset_hash_cache_for_tests()

    class _FakeGenCfg:
        save_test_images = True

    class _FakeSecrets:
        generate = _FakeGenCfg()

    monkeypatch.setattr(_gen.secrets, "load", lambda: _FakeSecrets())

    from studio.api.exception_handlers import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(_gen.router)
    return TestClient(app, raise_server_exceptions=False), test_dir


def _open_png_text(path: Path) -> dict[str, str]:
    with Image.open(path) as img:
        img.load()
        return dict(img.text)


def _post_xy_save(
    tc: TestClient,
    composite_params: dict,
    cells: list[tuple[int, int, dict]],
    task_id: int | None = None,
):
    """XY save 帮手：composite + N 张 cell + cells_manifest 一次 multipart 发出去。

    `cells` 每条 = (xi, yi, cell_params)；cell_params 是物化后的 single-snapshot。"""
    files: list[tuple[str, tuple[str, bytes, str]]] = [
        ("image", ("xy plot.png", _png_bytes(), "image/png")),
    ]
    for xi, yi, _cp in cells:
        files.append(("cells", (f"cell x{xi} y{yi}.png", _png_bytes(), "image/png")))
    manifest = [{"xi": xi, "yi": yi, "params": cp} for (xi, yi, cp) in cells]
    data = {
        "mode": "xy",
        "params": json.dumps(composite_params),
        "cells_manifest": json.dumps(manifest),
    }
    if task_id is not None:
        data["task_id"] = str(task_id)
    return tc.post("/api/generate/save", data=data, files=files)


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


def test_save_a1111_uses_effective_dataset_prompt_without_manifest(client) -> None:
    """旧客户端/无 task 档案也不能把 dataset picker 实际 prompt 写成空。"""
    tc, _ = client
    p = _params(
        prompts=[],
        dataset_pick={"tags": ["trigger", "orange hair", "flower crown"]},
    )
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(p)},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    a1111 = _open_png_text(Path(r.json()["path"]))["parameters"]
    assert a1111.splitlines()[0] == "trigger, orange hair, flower crown"


def test_save_writes_civitai_resource_hashes_from_private_task_manifest(
    client, tmp_path: Path,
) -> None:
    """实际 prompt + model/VAE/LoRA SHA256 写 parameters，绝对路径不进 PNG。"""
    from studio.services import generation_metadata as meta

    tc, _ = client
    model = tmp_path / "models" / "krea2-turbo.safetensors"
    vae = tmp_path / "models" / "qwen-vae.safetensors"
    lora = tmp_path / "loras" / "artist-style.safetensors"
    for path, payload in ((model, b"model"), (vae, b"vae"), (lora, b"lora")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    task_id = 123
    meta.write_manifest(
        task_id,
        prompts=["first prompt", "actual second prompt"],
        model_family="krea2",
        model_path=str(model),
        vae_path=str(vae),
        text_encoder="fp8",
        loras=[{"path": str(lora), "scale": 0.75}],
        xy_matrix=None,
    )
    p = _params(
        prompts=["UI raw prompt"],
        model_family="krea2",
        text_encoder="fp8",
        loras=[{"name": lora.name, "scale": 0.75}],
    )
    r = tc.post(
        "/api/generate/save",
        data={
            "mode": "single",
            "params": json.dumps(p),
            "task_id": str(task_id),
            "source_filename": "gen_0001_p1_c0_s7.png",
        },
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    text = _open_png_text(Path(r.json()["path"]))
    a1111 = text["parameters"]
    assert a1111.splitlines()[0] == "actual second prompt <lora:artist-style:0.75>"
    assert "Model: krea2-turbo" in a1111
    assert f"Model hash: {hashlib.sha256(b'model').hexdigest()}" in a1111
    assert "VAE: qwen-vae" in a1111
    assert f"VAE hash: {hashlib.sha256(b'vae').hexdigest()}" in a1111
    assert f"artist-style: {hashlib.sha256(b'lora').hexdigest()}" in a1111
    assert '"lora:artist-style"' in a1111
    assert "Model family: krea2" in a1111
    assert "Text encoder: fp8" in a1111
    assert "Software: AnimaLoraStudio" in a1111
    match = re.search(r", Hashes: (\{.*?\})(?:,|$)", a1111)
    assert match is not None
    assert json.loads(match.group(1)) == {
        "model": hashlib.sha256(b"model").hexdigest(),
        "vae": hashlib.sha256(b"vae").hexdigest(),
        "lora:artist-style": hashlib.sha256(b"lora").hexdigest(),
    }
    assert str(tmp_path) not in a1111
    assert str(tmp_path) not in text["anima_params"]


def test_resource_metadata_failure_does_not_block_png_save(
    client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.api.routers import generate as generate_router

    tc, _ = client

    def fail(*args, **kwargs):
        raise RuntimeError("hash backend unavailable")

    monkeypatch.setattr(generate_router, "build_external_metadata", fail)
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    assert "parameters" in _open_png_text(Path(r.json()["path"]))


def test_resource_hash_cache_reuses_and_invalidates_by_stat(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.services import generation_metadata as meta

    path = tmp_path / "resource.safetensors"
    path.write_bytes(b"v1")
    first = meta.file_sha256(path)
    assert first == hashlib.sha256(b"v1").hexdigest()

    def should_not_open(*args, **kwargs):
        raise AssertionError("cache hit unexpectedly reopened the resource")

    original_open = Path.open
    monkeypatch.setattr(Path, "open", should_not_open)
    assert meta.file_sha256(path) == first
    monkeypatch.setattr(Path, "open", original_open)

    path.write_bytes(b"version two")
    second = meta.file_sha256(path)
    assert second == hashlib.sha256(b"version two").hexdigest()
    assert second != first


def test_xy_cell_external_metadata_tracks_checkpoint_and_scale(
    client, tmp_path: Path,
) -> None:
    from studio.services import generation_metadata as meta

    first = tmp_path / "lora-a.safetensors"
    second = tmp_path / "lora-b.safetensors"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    meta.write_manifest(
        456,
        prompts=["xy prompt"],
        model_family="anima",
        model_path="",
        vae_path=None,
        text_encoder=None,
        loras=[{"path": str(first), "scale": 1.0}],
        xy_matrix={
            "x": {"axis": "lora_ckpt", "values": [str(first), str(second)], "lora_index": 0},
            "y": {"axis": "lora_scale", "values": [0.4, 0.8], "lora_index": None},
        },
    )
    params = _params(
        mode="single",
        xy_origin={"xi": 1, "yi": 0},
        loras=[{"name": second.name, "scale": 0.4}],
    )
    external = meta.build_external_metadata(456, params)
    assert external["prompt"] == "xy prompt"
    assert external["loras"] == [{
        "name": "lora-b",
        "scale": 0.4,
        "hash": hashlib.sha256(b"b").hexdigest(),
    }]


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
    assert r.json()["error"]["code"] == "generate.params_invalid"


def test_save_rejects_non_object_params(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": "[1, 2, 3]"},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "generate.params_invalid"


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
    """xy 模式的合成大图 + cell 文件夹走 disk-history。"""
    tc, _ = client
    _post_xy_save(
        tc,
        composite_params=_params(mode="xy", seed=99, xy_draft={
            "x": {"axis": "steps", "raw": "10, 20", "loraIndex": None},
            "y": None,
        }),
        cells=[
            (0, 0, _params(mode="single", seed=99, steps=10)),
            (1, 0, _params(mode="single", seed=99, steps=20)),
        ],
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
    """决策 #6：single v2 命名 'single image 1.png'，1-based。XY 见 test_save_xy_creates_folder。"""
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
    assert r1.json()["filename"] == "single image 1.png"
    assert r2.json()["filename"] == "single image 2.png"


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
    """决策 #7：XY composite PNG 不写 a1111 parameters 块（矩阵图单图对应不上）。
    cell PNG 是 single-snapshot，**会**写 a1111 block。"""
    tc, _ = client
    r = _post_xy_save(
        tc,
        composite_params=_params(mode="xy", xy_draft={
            "x": {"axis": "steps", "raw": "20", "loraIndex": None},
            "y": None,
        }),
        cells=[(0, 0, _params(mode="single", steps=20))],
    )
    assert r.status_code == 200, r.text
    composite = Path(r.json()["composite"])
    composite_text = _open_png_text(composite)
    assert "anima_params" in composite_text
    assert "parameters" not in composite_text
    # cell 是 single-snapshot：anima_params + a1111 块都在
    cell = Path(r.json()["cells"][0])
    cell_text = _open_png_text(cell)
    assert "anima_params" in cell_text
    assert "parameters" in cell_text


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

    from studio.api.exception_handlers import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(_gen.router)
    tc = TestClient(app, raise_server_exceptions=False)

    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "generate.save_disabled"


# ---------------------------------------------------------------------------
# XY 文件夹布局（恢复 PreviewXYGrid 历史回看）
# ---------------------------------------------------------------------------


def _xy_snapshot(*, x_raw: str = "10, 20, 30", y_raw: str | None = "3, 5") -> dict:
    return _params(mode="xy", xy_draft={
        "x": {"axis": "steps", "raw": x_raw, "loraIndex": None},
        "y": {"axis": "cfg_scale", "raw": y_raw, "loraIndex": None} if y_raw else None,
    })


def test_save_xy_creates_folder_with_composite_and_cells(client) -> None:
    """XY save 落出 <date>/xy/xy plot N/{xy plot.png + cell x<i> y<j>.png ...}，无残留 tmp。"""
    tc, test_dir = client
    r = _post_xy_save(
        tc,
        composite_params=_xy_snapshot(x_raw="10, 20", y_raw="3"),
        cells=[
            (0, 0, _params(mode="single", steps=10, cfg_scale=3.0)),
            (1, 0, _params(mode="single", steps=20, cfg_scale=3.0)),
        ],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    folder = Path(body["folder"])
    assert folder.is_dir()
    assert folder.name == "xy plot 1"
    assert (folder / "xy plot.png").is_file()
    assert (folder / "cell x0 y0.png").is_file()
    assert (folder / "cell x1 y0.png").is_file()
    # 无残留 tmp
    assert not list(folder.parent.glob(".xy plot *.tmp"))


def test_save_xy_cell_carries_single_snapshot_and_xy_origin(client) -> None:
    """cell PNG 是 single-snapshot：mode='single' + a1111 块 + xy_origin 链回 XY plot。"""
    tc, _ = client
    r = _post_xy_save(
        tc,
        composite_params=_xy_snapshot(x_raw="10, 20", y_raw=None),
        cells=[
            (0, 0, _params(mode="single", steps=10)),
            (1, 0, {**_params(mode="single", steps=20), "xy_origin": {
                "xi": 1, "yi": 0, "xv": "20", "yv": None,
                "x_axis": "steps", "y_axis": None,
            }}),
        ],
    )
    assert r.status_code == 200, r.text
    cell1 = Path(r.json()["cells"][1])
    text = _open_png_text(cell1)
    assert "anima_params" in text
    assert "parameters" in text  # single mode 写 a1111
    parsed = json.loads(text["anima_params"])
    assert parsed["mode"] == "single"
    assert parsed["steps"] == 20
    assert parsed["xy_origin"]["xi"] == 1
    assert parsed["xy_origin"]["x_axis"] == "steps"


def test_save_xy_rejects_cell_xy_collision(client) -> None:
    """两条 cell 同 (xi, yi) → 400."""
    tc, _ = client
    r = _post_xy_save(
        tc,
        composite_params=_xy_snapshot(),
        cells=[
            (0, 0, _params(mode="single")),
            (0, 0, _params(mode="single")),
        ],
    )
    assert r.status_code == 400
    assert "duplicate" in r.text


def test_save_xy_rejects_cell_count_mismatch(client) -> None:
    """cells_manifest 数 != cells 数 → 400."""
    tc, _ = client
    # 手工拼一个 mismatch：1 file vs 2 manifest entries
    r = tc.post(
        "/api/generate/save",
        data={
            "mode": "xy",
            "params": json.dumps(_xy_snapshot()),
            "cells_manifest": json.dumps([
                {"xi": 0, "yi": 0, "params": _params(mode="single")},
                {"xi": 1, "yi": 0, "params": _params(mode="single")},
            ]),
        },
        files=[
            ("image", ("xy plot.png", _png_bytes(), "image/png")),
            ("cells", ("cell x0 y0.png", _png_bytes(), "image/png")),
        ],
    )
    assert r.status_code == 400
    assert "length" in r.text


def test_save_xy_folder_index_skips_legacy_files(client) -> None:
    """残留的 legacy 平铺 `xy plot 7.png` 占位 → 新文件夹从 8 起，不撞号。"""
    tc, test_dir = client
    xy_dir = test_dir / "2026-06-08" / "xy"
    xy_dir.mkdir(parents=True)
    (xy_dir / "xy plot 7.png").write_bytes(_png_bytes())

    # 把"今天"设成 2026-06-08
    from datetime import date as _date
    from studio.api.routers import generate as _gen
    class _FixedDate(_date):
        @classmethod
        def today(cls):
            return cls(2026, 6, 8)
    import datetime
    orig_date = _gen.date
    try:
        _gen.date = _FixedDate
        r = _post_xy_save(
            tc,
            composite_params=_xy_snapshot(x_raw="10", y_raw=None),
            cells=[(0, 0, _params(mode="single", steps=10))],
        )
    finally:
        _gen.date = orig_date
    assert r.status_code == 200
    assert Path(r.json()["folder"]).name == "xy plot 8"


def test_disk_history_xy_folder_entry_has_xy_meta(client) -> None:
    """扫到 XY 文件夹时返回 xy_meta（per-cell 信息 + 直读 URL）。"""
    tc, _ = client
    _post_xy_save(
        tc,
        composite_params=_xy_snapshot(x_raw="10, 20", y_raw="3"),
        cells=[
            (0, 0, _params(mode="single", steps=10, cfg_scale=3.0)),
            (1, 0, _params(mode="single", steps=20, cfg_scale=3.0)),
        ],
    )
    entries = tc.get("/api/generate/disk/history").json()["entries"]
    assert len(entries) == 1
    e = entries[0]
    assert e["mode"] == "xy"
    assert e["folder"] == "xy plot 1"
    assert e["image_url"].endswith("xy%20plot.png")
    meta = e["xy_meta"]
    assert meta["x_axis"] == "steps"
    assert meta["y_axis"] == "cfg_scale"
    assert meta["x_values"] == ["10", "20"]
    assert meta["y_values"] == ["3"]
    assert len(meta["samples"]) == 2
    samples_by_pos = {(s["xy"]["xi"], s["xy"]["yi"]): s for s in meta["samples"]}
    assert samples_by_pos[(0, 0)]["xy"]["xv"] == "10"
    assert samples_by_pos[(1, 0)]["xy"]["xv"] == "20"
    # image_url 已 encode（空格 → %20）
    assert "%20" in samples_by_pos[(0, 0)]["image_url"]


def test_disk_history_skips_legacy_xy_flat_files(client) -> None:
    """legacy 平铺 `xy plot N.png` 文件即便带 anima_params 也不出现在 history（用户决策 hide）。"""
    tc, test_dir = client
    xy_dir = test_dir / "2026-06-08" / "xy"
    xy_dir.mkdir(parents=True)
    # 写一个带 anima_params 的 legacy 平铺文件
    info = PngImagePlugin.PngInfo()
    info.add_text("anima_params", json.dumps(_params(mode="xy")))
    Image.new("RGB", (8, 8)).save(xy_dir / "xy plot 99.png", format="PNG", pnginfo=info)

    entries = tc.get("/api/generate/disk/history").json()["entries"]
    assert entries == []


def test_disk_history_skips_xy_folder_without_composite(client) -> None:
    """文件夹存在但没 composite（半成品 / 手 mkdir）→ 跳过。"""
    tc, test_dir = client
    xy_dir = test_dir / "2026-06-08" / "xy" / "xy plot 1"
    xy_dir.mkdir(parents=True)
    # 只有 cell，没 composite
    info = PngImagePlugin.PngInfo()
    info.add_text("anima_params", json.dumps(_params(mode="single")))
    Image.new("RGB", (8, 8)).save(xy_dir / "cell x0 y0.png", format="PNG", pnginfo=info)

    entries = tc.get("/api/generate/disk/history").json()["entries"]
    assert entries == []


def test_disk_xy_folder_delete_removes_recursively(client) -> None:
    """DELETE /api/generate/disk/<date>/xy/<folder> → rmtree 整文件夹。"""
    tc, _ = client
    r = _post_xy_save(
        tc,
        composite_params=_xy_snapshot(x_raw="10", y_raw=None),
        cells=[(0, 0, _params(mode="single", steps=10))],
    )
    folder = Path(r.json()["folder"])
    assert folder.is_dir()
    entry = tc.get("/api/generate/disk/history").json()["entries"][0]
    url = f"/api/generate/disk/{entry['date']}/xy/{entry['folder'].replace(' ', '%20')}"
    delete_r = tc.delete(url)
    assert delete_r.status_code == 200
    assert delete_r.json()["noop"] is False
    assert not folder.exists()
    # 二次删 noop=True
    assert tc.delete(url).json()["noop"] is True


def test_disk_xy_image_route_resolves_subpath(client) -> None:
    """GET /api/generate/disk/image/<date>/xy/<folder>/<filename> → cell PNG bytes."""
    tc, _ = client
    r = _post_xy_save(
        tc,
        composite_params=_xy_snapshot(x_raw="10", y_raw=None),
        cells=[(0, 0, _params(mode="single", steps=10))],
    )
    entry = tc.get("/api/generate/disk/history").json()["entries"][0]
    cell_url = entry["xy_meta"]["samples"][0]["image_url"]
    cell_r = tc.get(cell_url)
    assert cell_r.status_code == 200
    assert cell_r.headers["content-type"] == "image/png"
    assert len(cell_r.content) > 0
    # composite 也可读
    composite_r = tc.get(entry["image_url"])
    assert composite_r.status_code == 200
    # thumb 也可读
    thumb_r = tc.get(entry["thumb_url"])
    assert thumb_r.status_code == 200


def test_disk_xy_path_traversal_blocked(client) -> None:
    """folder / filename 反 traversal：folder=`..` / 非法 folder name → 400."""
    tc, _ = client
    # folder name 不符 `xy plot N`
    r1 = tc.get("/api/generate/disk/image/2026-06-08/xy/not%20a%20folder/cell.png")
    assert r1.status_code == 400
    # date 非法
    r2 = tc.get("/api/generate/disk/image/bad-date/xy/xy%20plot%201/x.png")
    assert r2.status_code == 400
    # filename 非法
    r3 = tc.get("/api/generate/disk/image/2026-06-08/xy/xy%20plot%201/..%2Fsecret.png")
    assert r3.status_code in (400, 404)
    # DELETE 同样校验
    r4 = tc.delete("/api/generate/disk/2026-06-08/xy/bad%20folder")
    assert r4.status_code == 400


# ---------------------------------------------------------------------------
# _read_png_anima_params：手写 PNG chunk 扫描（替代 PIL.Image.open，~350× 提速）
# 三种文本块 × ascii/latin-1/CJK + 负例，逐值与 PIL 读法一致。
# ---------------------------------------------------------------------------


def _write_png_with_anima(path: Path, params: dict, *, zip: bool) -> None:
    info = PngImagePlugin.PngInfo()
    # add_text(zip=True) → zTXt；zip=False → tEXt；含非 latin-1（CJK）时 Pillow
    # 自动 fallback 到 iTXt（两种 zip 都是）。reader 三种 chunk 都要能读。
    info.add_text("anima_params", json.dumps(params, ensure_ascii=False), zip=zip)
    Image.new("RGB", (8, 8)).save(path, format="PNG", pnginfo=info)


@pytest.mark.parametrize("zip_flag", [True, False])
def test_read_png_anima_params_ascii_roundtrip(tmp_path: Path, zip_flag: bool) -> None:
    """ascii 内容：zip=True → zTXt，zip=False → tEXt，都要读出原 dict。"""
    from studio.api.routers import generate as _gen
    p = tmp_path / "a.png"
    params = _params(seed=123, prompts=["1girl, anime"])
    _write_png_with_anima(p, params, zip=zip_flag)
    assert _gen._read_png_anima_params(p) == params


def test_read_png_anima_params_cjk_uses_itxt(tmp_path: Path) -> None:
    """CJK 内容（ensure_ascii=False）→ Pillow 落 iTXt；reader utf-8 解出，逐值
    与 PIL 读法一致（防 latin-1 误解码导致的静默乱码）。"""
    from studio.api.routers import generate as _gen
    p = tmp_path / "a.png"
    params = _params(prompts=["1girl, 白发, 红眼"], negative_prompt="模糊")
    _write_png_with_anima(p, params, zip=True)
    out = _gen._read_png_anima_params(p)
    assert out == params
    with Image.open(p) as img:
        img.load()
        assert json.loads(img.text["anima_params"]) == out


def test_read_png_anima_params_missing_returns_none(tmp_path: Path) -> None:
    """没有 anima_params 块（只有像素 / 只有 a1111 parameters 块）→ None。"""
    from studio.api.routers import generate as _gen
    p = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(p, format="PNG")
    assert _gen._read_png_anima_params(p) is None
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "fake a1111 block")
    Image.new("RGB", (8, 8)).save(p, format="PNG", pnginfo=info)
    assert _gen._read_png_anima_params(p) is None


def test_read_png_anima_params_non_png_returns_none(tmp_path: Path) -> None:
    """非 PNG / 截断文件 → None（不抛）。"""
    from studio.api.routers import generate as _gen
    p = tmp_path / "a.png"
    p.write_bytes(b"not a png file at all")
    assert _gen._read_png_anima_params(p) is None
    p.write_bytes(b"\x89PNG\r\n\x1a\n")  # 只有签名，无 chunk
    assert _gen._read_png_anima_params(p) is None
