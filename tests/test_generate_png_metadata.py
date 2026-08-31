"""落盘 PNG metadata(a1111 / Civitai hash 链路)+ disk image/thumb 端点测试。

写路径走 services.generate_storage 直落(出图时间线单源;旧 POST /save 端点
已退役),读路径走 /api/generate/disk/image|thumb。
"""
from __future__ import annotations

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from studio import db
from studio.services import generate_storage as storage


def _png_bytes(color=(0, 0, 0), size=(8, 8)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _open_png_text(path: Path) -> dict[str, str]:
    with Image.open(path) as img:
        img.load()
        return dict(img.text)


def _params(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "mode": "single",
        "prompts": ["1girl, anime"],
        "negative_prompt": "blurry",
        "width": 1024, "height": 1024, "steps": 20, "cfg_scale": 7.0,
        "count": 1, "seed": 7,
        "loras": [],
        "xy_draft": None,
        "dataset_pick": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """tmp DB + tmp test/ + metadata 路径隔离 + 一条 generate task 行。"""
    from studio.services import generation_metadata as _meta

    monkeypatch.setattr(db, "STUDIO_DB", tmp_path / "studio.db")
    db.init_db()
    test_dir = tmp_path / "test"
    monkeypatch.setattr(storage, "TEST_IMAGES_DIR", test_dir)
    monkeypatch.setattr(
        _meta, "manifest_path",
        lambda task_id: tmp_path / "tasks" / str(task_id) / _meta.MANIFEST_FILENAME,
    )
    monkeypatch.setattr(_meta, "HASH_CACHE_PATH", tmp_path / ".cache" / "hashes.json")
    _meta._reset_hash_cache_for_tests()
    with db.connection_for() as conn:
        task_id = db.create_task(conn, name="generate", config_name="generate", priority=0)
        db.update_task(conn, task_id, task_type="generate")
    return task_id, test_dir, tmp_path


@pytest.fixture
def client(env, monkeypatch: pytest.MonkeyPatch):
    """disk image/thumb 端点的 TestClient(读路径)。"""
    from studio.api.exception_handlers import register_exception_handlers
    from studio.api.routers import generate as _gen

    task_id, test_dir, tmp_path = env
    monkeypatch.setattr(_gen, "TEST_IMAGES_DIR", test_dir)
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(_gen.router)
    return TestClient(app, raise_server_exceptions=False), task_id, test_dir


# ---------------------------------------------------------------------------
# PNG metadata(写路径:storage 直落)
# ---------------------------------------------------------------------------


def test_store_writes_anima_params_with_server_enrich(env) -> None:
    task_id, _, _ = env
    saved = storage._write_single(task_id, "a.png", _png_bytes(), _params(seed=42))
    parsed = json.loads(_open_png_text(saved)["anima_params"])
    assert parsed["seed"] == 42
    # server enrich 强制 schema_version=2 + 补 created_at / mode / task_id
    assert parsed["schema_version"] == 2
    assert parsed["mode"] == "single"
    assert parsed["task_id"] == task_id
    assert "created_at" in parsed


def test_store_writes_a1111_parameters_block(env) -> None:
    """a1111 兼容 `parameters` 块:ComfyUI / WebUI / Civitai 拖图能识别。"""
    task_id, _, _ = env
    p = _params(
        seed=42,
        loras=[{"name": "my-lora.safetensors", "scale": 0.8,
                "project_id": 12, "version_id": 34}],
    )
    saved = storage._write_single(task_id, "a.png", _png_bytes(), p)
    a1111 = _open_png_text(saved)["parameters"]
    first_line = a1111.split("\n", 1)[0]
    assert "1girl, anime" in first_line
    assert "<lora:my-lora:0.8>" in first_line  # a1111 语法去 .safetensors
    assert "Negative prompt: blurry" in a1111
    assert "Steps: 20" in a1111
    assert "CFG scale: 7.0" in a1111
    assert "Seed: 42" in a1111
    assert "Size: 1024x1024" in a1111


def test_a1111_prefers_dataset_prompt_over_legacy_tags_without_manifest(env) -> None:
    """新字段保存手工编辑值；旧 tags 仅作为缺字段时的兼容 fallback。"""
    task_id, _, _ = env
    p = _params(
        prompts=[],
        dataset_prompt="hand edited prompt",
        dataset_pick={"tags": ["stale", "legacy tags"]},
    )
    saved = storage._write_single(task_id, "a.png", _png_bytes(), p)
    assert _open_png_text(saved)["parameters"].splitlines()[0] == "hand edited prompt"


def test_a1111_preserves_explicit_empty_dataset_prompt_without_manifest(env) -> None:
    """显式空字符串也优先，不得重新灌入旧 tags。"""
    task_id, _, _ = env
    p = _params(
        prompts=["base prompt"],
        dataset_prompt="",
        dataset_pick={"tags": ["stale", "legacy tags"]},
    )
    saved = storage._write_single(task_id, "a.png", _png_bytes(), p)
    assert _open_png_text(saved)["parameters"].splitlines()[0] == "base prompt"


def test_a1111_uses_effective_dataset_prompt_without_manifest(env) -> None:
    """无 task 档案也不能把 dataset picker 实际 prompt 写成空。"""
    task_id, _, _ = env
    p = _params(
        prompts=[],
        dataset_pick={"tags": ["trigger", "orange hair", "flower crown"]},
    )
    saved = storage._write_single(task_id, "a.png", _png_bytes(), p)
    a1111 = _open_png_text(saved)["parameters"]
    assert a1111.splitlines()[0] == "trigger, orange hair, flower crown"


def test_store_writes_civitai_resource_hashes_from_manifest(env) -> None:
    """实际 prompt + model/VAE/LoRA SHA256 写 parameters,绝对路径不进 PNG。"""
    from studio.services import generation_metadata as meta

    task_id, _, tmp_path = env
    model = tmp_path / "models" / "krea2-turbo.safetensors"
    vae = tmp_path / "models" / "qwen-vae.safetensors"
    lora = tmp_path / "loras" / "artist-style.safetensors"
    for path, payload in ((model, b"model"), (vae, b"vae"), (lora, b"lora")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

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
    # source_filename 的 _pN_ 选择第二条 prompt
    saved = storage._write_single(task_id, "gen_0001_p1_c0_s7.png", _png_bytes(), p)
    text = _open_png_text(saved)
    a1111 = text["parameters"]
    assert a1111.splitlines()[0] == "actual second prompt <lora:artist-style:0.75>"
    assert "Model: krea2-turbo" in a1111
    assert f"Model hash: {hashlib.sha256(b'model').hexdigest()}" in a1111
    assert "VAE: qwen-vae" in a1111
    assert f"VAE hash: {hashlib.sha256(b'vae').hexdigest()}" in a1111
    assert f"artist-style: {hashlib.sha256(b'lora').hexdigest()}" in a1111
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


def test_metadata_failure_does_not_block_store(env, monkeypatch) -> None:
    task_id, _, _ = env

    def fail(*args, **kwargs):
        raise RuntimeError("hash backend unavailable")

    monkeypatch.setattr(storage, "build_external_metadata", fail)
    saved = storage._write_single(task_id, "a.png", _png_bytes(), _params())
    assert "parameters" in _open_png_text(saved)


def test_resource_hash_cache_reuses_and_invalidates_by_stat(
    env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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


def test_prewarm_resource_hashes_fills_cache(env, tmp_path: Path, monkeypatch) -> None:
    """enqueue 时预热 → 落盘时 file_sha256 缓存命中不再读文件。"""
    from studio.services import generation_metadata as meta

    res = tmp_path / "big-model.safetensors"
    res.write_bytes(b"weights")
    t = meta.prewarm_resource_hashes([str(res), None, ""])
    assert t is not None
    t.join(timeout=10)

    def should_not_open(*args, **kwargs):
        raise AssertionError("prewarmed hash unexpectedly reopened the resource")

    monkeypatch.setattr(Path, "open", should_not_open)
    assert meta.file_sha256(res) == hashlib.sha256(b"weights").hexdigest()
    # 全空列表 → 不起线程
    assert meta.prewarm_resource_hashes([None, ""]) is None


def test_xy_cell_external_metadata_tracks_checkpoint_and_scale(env) -> None:
    from studio.services import generation_metadata as meta

    task_id, _, tmp_path = env
    first = tmp_path / "lora-a.safetensors"
    second = tmp_path / "lora-b.safetensors"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    meta.write_manifest(
        task_id,
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
    external = meta.build_external_metadata(task_id, params)
    assert external["prompt"] == "xy prompt"
    assert external["loras"] == [{
        "name": "lora-b",
        "scale": 0.4,
        "hash": hashlib.sha256(b"b").hexdigest(),
    }]


# ---------------------------------------------------------------------------
# disk image / thumb 端点(读路径)
# ---------------------------------------------------------------------------


def test_disk_image_serves_saved_file(client) -> None:
    tc, task_id, _ = client
    saved = storage._write_single(task_id, "a.png", _png_bytes(), _params())
    r = tc.get(
        f"/api/generate/disk/image/{saved.parent.parent.name}/single/{saved.name}"
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_disk_image_validates_inputs(client) -> None:
    tc, _, _ = client
    assert tc.get("/api/generate/disk/image/bad-date/single/a.png").status_code == 400
    assert tc.get("/api/generate/disk/image/2026-01-01/nope/a.png").status_code == 400
    assert tc.get("/api/generate/disk/image/2026-01-01/single/a.txt").status_code == 400
    assert tc.get("/api/generate/disk/image/2026-01-01/single/missing.png").status_code == 404


def test_disk_thumb_returns_png_with_etag(client) -> None:
    tc, task_id, _ = client
    saved = storage._write_single(
        task_id, "a.png", _png_bytes(size=(64, 64)), _params(),
    )
    r = tc.get(
        f"/api/generate/disk/thumb/{saved.parent.parent.name}/single/{saved.name}?w=32"
    )
    assert r.status_code == 200
    assert "ETag" in r.headers
    with Image.open(BytesIO(r.content)) as img:
        assert max(img.size) <= 32


def test_disk_xy_image_route_resolves_subpath(client) -> None:
    tc, task_id, _ = client
    cell = storage._write_xy_cell(
        task_id, "c0.png", _png_bytes(),
        _params(mode="xy", xy_draft={
            "x": {"axis": "steps", "raw": "20", "loraIndex": None}, "y": None,
        }),
        {"xi": 0, "yi": 0, "xv": 20, "yv": None},
    )
    date_str = cell.parent.parent.parent.name
    r = tc.get(
        f"/api/generate/disk/image/{date_str}/xy/{cell.parent.name}/{cell.name}"
    )
    assert r.status_code == 200
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_disk_path_traversal_blocked(client) -> None:
    tc, _, _ = client
    # _PNG_NAME_SAFE_RE 不放行 / \ .. 等
    assert tc.get(
        "/api/generate/disk/image/2026-01-01/single/..%2Fsecret.png"
    ).status_code in (400, 404)
    assert tc.get(
        "/api/generate/disk/image/2026-01-01/xy/xy%20plot%201/..%2F..%2Fx.png"
    ).status_code in (400, 404)
