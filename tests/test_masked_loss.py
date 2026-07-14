"""masked loss（PR-B B2）——dataset mask 管线 / npz 缓存失效 / collate / loss 数学。

设计：docs/design/preprocess-inpaint-mask-design.md §5 / §9（D5 修订）。
mask sidecar 路径 = 与图同目录的 {stem}.mask（内容灰度 PNG 字节），
灰度 255=学 0=不学；数据层随图同几何变换（NEAREST）后 area 下采样到
latent /8，loss 层加权均值 reduction `(loss*mask).sum()/mask.sum()`。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from runtime.training.dataset import (  # noqa: E402
    BucketManager,
    CachedLatentDataset,
    ImageDataset,
    collate_fn,
    collate_fn_cached,
)
from runtime.training.loop import _masked_mean, _masked_mean_per_sample  # noqa: E402


class _FakeVAEModel:
    def encode(self, pixels_5d, scale):
        b, c, t, h, w = pixels_5d.shape
        return torch.zeros(b, 16, 1, h // 8, w // 8, dtype=pixels_5d.dtype)


class _FakeVAE:
    def __init__(self):
        self.model = _FakeVAEModel()
        self.scale = 1.0

    def encode(self, pixels):
        return self.model.encode(pixels, self.scale)


def _write_img(path: Path, size=(256, 256)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(127, 127, 127)).save(path)
    path.with_suffix(".txt").write_text("test", encoding="utf-8")


def _write_mask(img_path: Path, builder) -> Path:
    """builder(Image) 生成 mask 内容；路径按落盘约定 = 同目录 {stem}.mask。

    .mask 后缀 PIL 推断不出格式，必须显式 format="PNG"。
    """
    mp = img_path.parent / f"{img_path.stem}.mask"
    builder().save(mp, format="PNG")
    return mp


def _half_mask(size=(256, 256)):
    """上半 0（不学）下半 255（学）。"""
    m = Image.new("L", size, 255)
    m.paste(0, (0, 0, size[0], size[1] // 2))
    return m


def _square_dataset(tmp_path: Path, **kwargs) -> ImageDataset:
    mgr = BucketManager(256, min_reso=256, max_reso=256, step=64)
    return ImageDataset(tmp_path, 256, mgr, **kwargs)


# ---------------------------------------------------------------------------
# ImageDataset：mask 加载 + 几何变换 + latent 下采样
# ---------------------------------------------------------------------------


def test_mask_transforms_with_image(tmp_path: Path) -> None:
    img_path = tmp_path / "X.png"
    _write_img(img_path)
    _write_mask(img_path, _half_mask)

    ds = _square_dataset(tmp_path, load_masks=True)
    item = ds.get_with_flip(0, flip=False)
    mask = item["mask"]
    assert mask is not None
    assert mask.shape == (32, 32)  # 256/8
    # 上半不学（0）下半学（1）
    assert float(mask[:16].max()) == 0.0
    assert float(mask[16:].min()) == 1.0


def test_mask_flips_with_image(tmp_path: Path) -> None:
    img_path = tmp_path / "X.png"
    _write_img(img_path)

    def left_mask():
        m = Image.new("L", (256, 256), 255)
        m.paste(0, (0, 0, 128, 256))  # 左半不学
        return m

    _write_mask(img_path, left_mask)
    ds = _square_dataset(tmp_path, load_masks=True)
    m0 = ds.get_with_flip(0, flip=False)["mask"]
    m1 = ds.get_with_flip(0, flip=True)["mask"]
    assert float(m0[:, :16].max()) == 0.0 and float(m0[:, 16:].min()) == 1.0
    # flip 后左右对调
    assert float(m1[:, 16:].max()) == 0.0 and float(m1[:, :16].min()) == 1.0


def test_mask_size_mismatch_degrades_to_none(tmp_path: Path) -> None:
    img_path = tmp_path / "X.png"
    _write_img(img_path)
    _write_mask(img_path, lambda: Image.new("L", (128, 128), 0))

    ds = _square_dataset(tmp_path, load_masks=True)
    assert ds.get_with_flip(0, flip=False)["mask"] is None


def test_mask_not_loaded_when_disabled(tmp_path: Path) -> None:
    img_path = tmp_path / "X.png"
    _write_img(img_path)
    _write_mask(img_path, _half_mask)

    ds = _square_dataset(tmp_path, load_masks=False)
    assert ds.get_with_flip(0, flip=False)["mask"] is None


def test_mask_missing_is_none(tmp_path: Path) -> None:
    img_path = tmp_path / "X.png"
    _write_img(img_path)
    ds = _square_dataset(tmp_path, load_masks=True)
    assert ds.get_with_flip(0, flip=False)["mask"] is None


def test_gray_mask_partial_weight(tmp_path: Path) -> None:
    """灰度中间值 = 部分权重（软边笔刷语义）。"""
    img_path = tmp_path / "X.png"
    _write_img(img_path)
    _write_mask(img_path, lambda: Image.new("L", (256, 256), 128))

    ds = _square_dataset(tmp_path, load_masks=True)
    mask = ds.get_with_flip(0, flip=False)["mask"]
    assert abs(float(mask.mean()) - 128 / 255) < 0.01


# ---------------------------------------------------------------------------
# CachedLatentDataset：npz mask 键 + 缓存失效三条
# ---------------------------------------------------------------------------


def test_cache_stores_mask_keys(tmp_path: Path) -> None:
    img_path = tmp_path / "X.png"
    _write_img(img_path)
    _write_mask(img_path, _half_mask)

    ds = _square_dataset(tmp_path, load_masks=True, flip_augment=True)
    cached = CachedLatentDataset(ds, _FakeVAE(), device="cpu", dtype=torch.float32)

    with np.load(img_path.with_suffix(".npz")) as data:
        assert "mask" in data.files
        assert "mask_flipped" in data.files
        assert data["mask"].shape == (32, 32)

    item = cached[0]
    assert item["mask"] is not None
    assert item["mask"].shape == (32, 32)


def test_cache_no_mask_keys_without_mask(tmp_path: Path) -> None:
    img_path = tmp_path / "X.png"
    _write_img(img_path)
    ds = _square_dataset(tmp_path, load_masks=True)
    CachedLatentDataset(ds, _FakeVAE(), device="cpu", dtype=torch.float32)
    with np.load(img_path.with_suffix(".npz")) as data:
        assert "mask" not in data.files


def test_cache_invalidation_three_rules(tmp_path: Path) -> None:
    """§9 决策 3：新画 / 重画 / 清除 三条都必须让缓存失效。"""
    img_path = tmp_path / "X.png"
    _write_img(img_path)

    ds = _square_dataset(tmp_path, load_masks=True)
    cached = CachedLatentDataset(ds, _FakeVAE(), device="cpu", dtype=torch.float32)
    npz_path = img_path.with_suffix(".npz")
    assert cached._is_cache_valid(img_path, npz_path) is True

    # 1) 新画 mask（文件出现但 npz 无 mask 键）→ 失效
    mp = _write_mask(img_path, _half_mask)
    assert cached._is_cache_valid(img_path, npz_path) is False

    # 重建缓存（含 mask）后有效
    ds2 = _square_dataset(tmp_path, load_masks=True)
    cached2 = CachedLatentDataset(ds2, _FakeVAE(), device="cpu", dtype=torch.float32)
    assert cached2._is_cache_valid(img_path, npz_path) is True

    # 2) 重画 mask（mtime 新于 npz）→ 失效
    future = npz_path.stat().st_mtime + 10
    os.utime(mp, (future, future))
    assert cached2._is_cache_valid(img_path, npz_path) is False

    # 恢复 mask mtime 到当前（消掉人为的未来时间戳），否则第 3 步重建的
    # npz 永远"旧于" mask
    os.utime(mp, None)

    # 3) 清除 mask（文件删了但 npz 还有 mask 键）→ 失效
    ds3 = _square_dataset(tmp_path, load_masks=True)
    cached3 = CachedLatentDataset(ds3, _FakeVAE(), device="cpu", dtype=torch.float32)
    assert cached3._is_cache_valid(img_path, npz_path) is True
    mp.unlink()
    assert cached3._is_cache_valid(img_path, npz_path) is False


def test_cache_with_mask_keys_valid_when_masks_disabled(tmp_path: Path) -> None:
    """load_masks=False 时带 mask 键的缓存仍有效（超集语义，对齐 latent_flipped）。"""
    img_path = tmp_path / "X.png"
    _write_img(img_path)
    _write_mask(img_path, _half_mask)
    ds = _square_dataset(tmp_path, load_masks=True)
    CachedLatentDataset(ds, _FakeVAE(), device="cpu", dtype=torch.float32)

    ds_off = _square_dataset(tmp_path, load_masks=False)
    cached_off = CachedLatentDataset(ds_off, _FakeVAE(), device="cpu", dtype=torch.float32)
    assert cached_off._is_cache_valid(img_path, img_path.with_suffix(".npz")) is True
    # 且不返回 mask
    assert cached_off[0]["mask"] is None


# ---------------------------------------------------------------------------
# collate：混合有 / 无 mask
# ---------------------------------------------------------------------------


def test_collate_cached_fills_ones_for_unmasked() -> None:
    latent = torch.zeros(16, 1, 32, 32)
    half = torch.zeros(32, 32)
    half[16:] = 1.0
    batch = [
        {"latent": latent, "caption": "a", "mask": half},
        {"latent": latent, "caption": "b", "mask": None},
    ]
    out = collate_fn_cached(batch)
    assert out["masks"].shape == (2, 32, 32)
    assert float(out["masks"][1].min()) == 1.0  # 无 mask 图全 1
    assert float(out["masks"][0][:16].max()) == 0.0


def test_collate_omits_masks_when_all_none() -> None:
    latent = torch.zeros(16, 1, 32, 32)
    batch = [{"latent": latent, "caption": "a", "mask": None}]
    assert "masks" not in collate_fn_cached(batch)

    pixels = torch.zeros(3, 256, 256)
    pbatch = [{"pixel_values": pixels, "caption": "a", "mask": None}]
    assert "masks" not in collate_fn(pbatch)


def test_collate_pixel_path_mask_shape() -> None:
    pixels = torch.zeros(3, 256, 256)
    mask = torch.ones(32, 32)
    batch = [
        {"pixel_values": pixels, "caption": "a", "mask": mask},
        {"pixel_values": pixels, "caption": "b", "mask": None},
    ]
    out = collate_fn(batch)
    assert out["masks"].shape == (2, 32, 32)


# ---------------------------------------------------------------------------
# loss 数学：加权均值 reduction（§8）
# ---------------------------------------------------------------------------


def test_masked_mean_ignores_masked_region() -> None:
    """mask=0 区域的 loss 值完全不影响结果。"""
    loss = torch.ones(2, 16, 1, 4, 4)
    loss[:, :, :, :2, :] = 999.0  # 上半灌大值
    mask = torch.zeros(2, 1, 1, 4, 4)
    mask[:, :, :, 2:, :] = 1.0  # 只学下半
    assert abs(float(_masked_mean(loss, mask)) - 1.0) < 1e-6


def test_masked_mean_equals_mean_with_full_mask() -> None:
    """全 1 mask 时与朴素 mean 等价（无 mask 图的行为不变性）。"""
    torch.manual_seed(0)
    loss = torch.rand(2, 16, 1, 4, 4)
    mask = torch.ones(2, 1, 1, 4, 4)
    assert abs(float(_masked_mean(loss, mask)) - float(loss.mean())) < 1e-6


def test_masked_mean_area_normalization() -> None:
    """不同 mask 面积的样本不被隐性降权：分母是 mask 元素和。

    样本 0 学一半（loss=2）、样本 1 全学（loss=1）→
    (2*8*16 + 1*16*16) / (8*16 + 16*16) = 512/384 = 4/3。
    """
    loss = torch.ones(2, 16, 1, 4, 4)
    loss[0] = 2.0
    mask = torch.ones(2, 1, 1, 4, 4)
    mask[0, :, :, :2, :] = 0.0
    expected = (2.0 * 8 * 16 + 1.0 * 16 * 16) / (8 * 16 + 16 * 16)
    assert abs(float(_masked_mean(loss, mask)) - expected) < 1e-6


def test_masked_mean_per_sample() -> None:
    loss = torch.ones(2, 16, 1, 4, 4)
    loss[0, :, :, :2, :] = 999.0
    mask = torch.ones(2, 1, 1, 4, 4)
    mask[0, :, :, :2, :] = 0.0
    per = _masked_mean_per_sample(loss, mask)
    assert per.shape == (2,)
    assert abs(float(per[0]) - 1.0) < 1e-6
    assert abs(float(per[1]) - 1.0) < 1e-6
