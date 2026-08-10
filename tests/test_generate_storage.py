"""services.generate_storage —— 出图 server 端落盘/记账闭环单测。

覆盖:single/xy 直落(命名、PNG anima_params 注入、generate_images 记账)、
cell snapshot 物化(前端 buildCellSnapshot 的 Python 移植)、temp 记账、
落盘 fallback 反查、composite 补传、cache 单图剔除。
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from studio import db
from studio.services import generate_storage as storage


def _png_bytes(color=(0, 0, 0), size=(8, 8)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _png_text(path: Path) -> dict[str, str]:
    with Image.open(path) as img:
        img.load()
        return dict(img.text)


def _snapshot(**overrides) -> dict:
    base = {
        "schema_version": 2,
        "mode": "single",
        "prompts": ["1girl"],
        "negative_prompt": "",
        "width": 512, "height": 512, "steps": 20, "cfg_scale": 4.0,
        "count": 1, "seed": 7,
        "loras": [{"name": "a.safetensors", "scale": 1.0,
                   "project_id": None, "version_id": None}],
        "xy_draft": None,
        "dataset_pick": None,
    }
    base.update(overrides)
    return base


def _xy_snapshot(**overrides) -> dict:
    base = {
        "mode": "xy",
        "xy_draft": {
            "x": {"axis": "steps", "raw": "20, 25", "loraIndex": None},
            "y": {"axis": "cfg_scale", "raw": "3, 4", "loraIndex": None},
        },
    }
    base.update(overrides)
    return _snapshot(**base)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """tmp DB + tmp test/ 目录 + 一条 generate task 行。"""
    from studio.services import generation_metadata as _meta

    dbfile = tmp_path / "studio.db"
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
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
    return task_id, test_dir


def _images(task_id: int) -> list[dict]:
    return storage.load_images(task_id)


# ---------------------------------------------------------------------------
# build_cell_snapshot(Python 版 buildCellSnapshot)
# ---------------------------------------------------------------------------


def test_build_cell_snapshot_scalar_axes() -> None:
    snap = storage.build_cell_snapshot(
        _xy_snapshot(), {"xi": 1, "yi": 0, "xv": 25, "yv": 3},
    )
    assert snap["mode"] == "single"
    assert snap["xy_draft"] is None
    assert snap["steps"] == 25
    assert snap["cfg_scale"] == 3.0
    assert snap["xy_origin"] == {
        "xi": 1, "yi": 0, "xv": 25, "yv": 3,
        "x_axis": "steps", "y_axis": "cfg_scale",
    }
    # 原 XY snapshot 不被污染
    assert _xy_snapshot()["steps"] == 20


def test_build_cell_snapshot_lora_axes() -> None:
    xy = _xy_snapshot(xy_draft={
        "x": {"axis": "lora_scale", "raw": "0.5, 1.0", "loraIndex": None},
        "y": {"axis": "lora_ckpt", "raw": "a, b", "loraIndex": 0},
    })
    snap = storage.build_cell_snapshot(
        xy, {"xi": 0, "yi": 1, "xv": 0.5, "yv": "sub/dir/epoch40.safetensors"},
    )
    assert snap["loras"][0]["scale"] == 0.5
    assert snap["loras"][0]["name"] == "epoch40.safetensors"
    assert snap["loras"][0]["project_id"] is None
    # 值转换失败静默跳过(轴保持原值)
    snap2 = storage.build_cell_snapshot(
        _xy_snapshot(), {"xi": 0, "yi": 0, "xv": "not-a-number", "yv": None},
    )
    assert snap2["steps"] == 20


# ---------------------------------------------------------------------------
# 直落:single / xy cell
# ---------------------------------------------------------------------------


def test_write_single_names_injects_and_records(env) -> None:
    task_id, test_dir = env
    t1 = storage._write_single(task_id, "img_p0_0.png", _png_bytes(), _snapshot())
    t2 = storage._write_single(task_id, "img_p0_1.png", _png_bytes(), _snapshot())
    assert t1.name == "single image 1.png"
    assert t2.name == "single image 2.png"
    assert t1.parent.name == "single"
    text = _png_text(t1)
    params = json.loads(text["anima_params"])
    assert params["task_id"] == task_id
    assert params["mode"] == "single"
    assert "parameters" in text  # a1111 块
    imgs = _images(task_id)
    assert [i["src"] for i in imgs] == ["img_p0_0.png", "img_p0_1.png"]
    assert imgs[0]["file"].endswith("/single/single image 1.png")
    assert "\\" not in imgs[0]["file"]  # DB 统一正斜杠


def test_write_xy_cells_share_folder_and_record_pos(env) -> None:
    task_id, test_dir = env
    info0 = {"xi": 0, "yi": 0, "xv": 20, "yv": 3}
    info1 = {"xi": 1, "yi": 0, "xv": 25, "yv": 3}
    c0 = storage._write_xy_cell(task_id, "c0.png", _png_bytes(), _xy_snapshot(), info0)
    c1 = storage._write_xy_cell(task_id, "c1.png", _png_bytes(), _xy_snapshot(), info1)
    assert c0.parent == c1.parent
    assert c0.parent.name == "xy plot 1"
    assert c0.name == "cell x0 y0.png"
    params = json.loads(_png_text(c1)["anima_params"])
    assert params["mode"] == "single"       # cell 按 single 物化
    assert params["steps"] == 25            # X 轴真值
    assert params["xy_origin"]["xi"] == 1
    imgs = _images(task_id)
    assert [(i["xi"], i["yi"]) for i in imgs] == [(0, 0), (1, 0)]


def test_find_disk_file_by_src(env) -> None:
    task_id, _ = env
    target = storage._write_single(task_id, "img_p0_0.png", _png_bytes(), _snapshot())
    assert storage.find_disk_file(task_id, "img_p0_0.png") == target
    assert storage.find_disk_file(task_id, "nope.png") is None


# ---------------------------------------------------------------------------
# handle_image_done:temp 记账(同步路径)
# ---------------------------------------------------------------------------


def test_handle_image_done_temp_records_cache_item(env) -> None:
    task_id, _ = env
    storage.handle_image_done(
        task_id, "img_p0_0.png", _png_bytes(), _snapshot(),
        mode="single", xy_info=None, save_to_disk=False,
    )
    storage.handle_image_done(
        task_id, "c0.png", _png_bytes(), _xy_snapshot(),
        mode="xy", xy_info={"xi": 2, "yi": 1, "xv": 20, "yv": 4},
        save_to_disk=False,
    )
    imgs = _images(task_id)
    assert imgs[0] == {"cache": "img_p0_0.png"}
    assert imgs[1] == {"cache": "c0.png", "xi": 2, "yi": 1}


# ---------------------------------------------------------------------------
# composite 补传
# ---------------------------------------------------------------------------


def test_attach_xy_composite(env, monkeypatch: pytest.MonkeyPatch) -> None:
    task_id, _ = env
    with pytest.raises(LookupError):
        storage._attach_xy_composite_sync(task_id, _png_bytes())
    storage._write_xy_cell(
        task_id, "c0.png", _png_bytes(), _xy_snapshot(),
        {"xi": 0, "yi": 0, "xv": 20, "yv": 3},
    )
    with db.connection_for() as conn:
        db.update_task(
            conn, task_id,
            generate_params=json.dumps(_xy_snapshot(), ensure_ascii=False),
        )
    target = storage._attach_xy_composite_sync(task_id, _png_bytes((9, 9, 9)))
    assert target.name == "xy plot.png"
    assert target.parent.name == "xy plot 1"
    params = json.loads(_png_text(target)["anima_params"])
    assert params["mode"] == "xy"
    # composite 不入台账(应用内回看用 cells)
    assert all("xy plot.png" not in str(i.get("file", "")) for i in _images(task_id))


# ---------------------------------------------------------------------------
# disk_cache.drop_image(落盘后剔中转副本)
# ---------------------------------------------------------------------------


def test_disk_cache_drop_image(tmp_path: Path) -> None:
    from studio.services.inference import disk_cache

    sc = disk_cache.SessionCache(root=tmp_path / "cache")
    sc.ensure_dir()
    sc.put(1, "a.png", b"data-a", {})
    sc.put(1, "b.png", b"data-b", {})
    assert sc.drop_image(1, "a.png") is True
    assert sc.drop_image(1, "a.png") is False  # 已剔,幂等
    assert sc.get_image(1, "a.png") is None
    assert sc.get_image(1, "b.png") == b"data-b"
    assert sc.total_count() == 1
