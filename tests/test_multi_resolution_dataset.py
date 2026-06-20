"""多分辨率数据集：文件夹 px 覆盖 + repeat × 分辨率 fan-out + 多 base 分桶。

见 docs/design/multi-resolution-training-design.md §3-§6。
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _write_img(path: Path, size=(640, 512), color=(127, 127, 127)) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    path.with_suffix(".txt").write_text("1girl", encoding="utf-8")


def test_parse_folder_meta() -> None:
    from runtime.training.dataset import ImageDataset
    f = ImageDataset._parse_folder_meta
    assert f("1024px_2_data") == (1024, 2, "data")
    assert f("768px_concept") == (768, 1, "concept")
    assert f("1024px_data") == (1024, 1, "data")
    assert f("5_concept") == (None, 5, "concept")          # Kohya 向后兼容
    assert f("concept") == (None, 1, "concept")
    # 分辨率值 snap 到 64 倍数 + clamp
    assert f("1000px_data") == (1024, 1, "data")           # round(1000/64)*64
    assert f("100px_data") == (256, 1, "data")             # clamp 下限 256


def test_as_resolutions_normalizes() -> None:
    from runtime.training.phases.dataset import _as_resolutions
    assert _as_resolutions(1024) == [1024]
    assert _as_resolutions([512, 768, 1024]) == [512, 768, 1024]
    assert _as_resolutions((512, 768)) == [512, 768]
    assert _as_resolutions([]) == [1024]                   # 空兜底


def test_per_folder_resolution_override(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from runtime.training.dataset import BucketManager, ImageDataset
    _write_img(tmp_path / "data" / "a.png")
    _write_img(tmp_path / "512px_hires" / "b.png")
    base_mgr = BucketManager(1024)
    ds = ImageDataset(
        tmp_path, 1024, base_mgr,
        resolutions=[1024], aspect_ratio_limit=2.0, prefer_json=False,
    )
    target = {Path(s["image"]).parent.name: s["target_reso"] for s in ds.samples}
    assert target["data"] == 1024            # 无 px → config 档
    assert target["512px_hires"] == 512      # px 覆盖
    # 非 base 档的 manager 按需建
    assert ds._bucket_mgr_for(512).base_reso == 512
    assert ds._bucket_mgr_for(1024) is base_mgr   # base 档复用传入的 mgr


def test_target_reso_drives_bucket_size(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from runtime.training.dataset import BucketManager, ImageDataset
    _write_img(tmp_path / "data" / "a.png", size=(640, 512))
    _write_img(tmp_path / "512px_hires" / "b.png", size=(640, 512))
    base_mgr = BucketManager(1024)
    ds = ImageDataset(
        tmp_path, 1024, base_mgr,
        resolutions=[1024], aspect_ratio_limit=2.0, prefer_json=False,
    )
    by_folder = {Path(s["image"]).parent.name: i for i, s in enumerate(ds.samples)}
    big = ds.get_with_flip(by_folder["data"], flip=False)["pixel_values"]
    small = ds.get_with_flip(by_folder["512px_hires"], flip=False)["pixel_values"]
    big_area = big.shape[1] * big.shape[2]      # ≈ 1024²
    small_area = small.shape[1] * small.shape[2]  # ≈ 512²
    assert big_area > small_area * 2            # 面积约 4 倍


def test_resolution_list_fans_out_per_image(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from runtime.training.dataset import BucketManager, ImageDataset
    _write_img(tmp_path / "2_data" / "a.png")   # repeat 2, 无 px
    ds = ImageDataset(
        tmp_path, 768, BucketManager(768),
        resolutions=[512, 768], aspect_ratio_limit=2.0, prefer_json=False,
    )
    # 1 图 × repeat 2 × 2 分辨率 = 4 样本
    assert len(ds.samples) == 4
    assert sorted(s["target_reso"] for s in ds.samples) == [512, 512, 768, 768]


def test_px_folder_overrides_list_no_fanout(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from runtime.training.dataset import BucketManager, ImageDataset
    _write_img(tmp_path / "1024px_2_hires" / "a.png")  # px 覆盖 + repeat 2
    ds = ImageDataset(
        tmp_path, 768, BucketManager(768),
        resolutions=[512, 768], aspect_ratio_limit=2.0, prefer_json=False,
    )
    # px 文件夹：repeat 2 × 1 档(1024) = 2 样本，不参与 [512,768] fan-out
    assert len(ds.samples) == 2
    assert all(s["target_reso"] == 1024 for s in ds.samples)
