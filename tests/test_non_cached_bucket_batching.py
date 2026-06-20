"""非缓存路径（cache_latents=False）也按 ARB 桶分批。

回归：此前非缓存路径用普通 DataLoader(batch_size=N)，不按桶分组 → 一个 batch 混入
不同长宽比（不同桶 = 不同 H×W）的图，collate_fn 的 torch.stack 直接 RuntimeError。
修复：ImageDataset 预扫每图尺寸建 bucket_for_index，非缓存路径改用 BucketBatchSampler。
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _write(d: Path, name: str, size) -> None:
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(d / f"{name}.png")
    (d / f"{name}.txt").write_text("1girl", encoding="utf-8")


def test_image_dataset_builds_bucket_for_index(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from runtime.training.dataset import BucketManager, ImageDataset
    _write(tmp_path / "1_data", "a", (1024, 1024))
    _write(tmp_path / "1_data", "b", (1536, 1024))  # 不同 AR → 不同桶
    ds = ImageDataset(tmp_path, 1024, BucketManager(1024), prefer_json=False)
    assert len(ds.bucket_for_index) == len(ds.samples)
    assert all(b is not None for b in ds.bucket_for_index)
    assert len(set(ds.bucket_for_index)) >= 2  # 两种 AR 落不同桶


def test_no_bucket_mgr_yields_all_none(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from runtime.training.dataset import ImageDataset
    _write(tmp_path / "1_data", "a", (1024, 1024))
    ds = ImageDataset(tmp_path, 1024, bucket_mgr=None, prefer_json=False)
    assert ds.bucket_for_index == [None]  # 不分桶 → sampler 退回普通切批


def test_non_cached_batches_are_shape_uniform(tmp_path: Path) -> None:
    # 核心回归：非缓存 + 多长宽比 + bs>1 迭代不崩，且每个 batch 内尺寸统一。
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    from runtime.training.dataset import (
        BucketBatchSampler,
        BucketManager,
        ImageDataset,
        collate_fn,
    )

    # 4 种长宽比各 2 张 → 4 个桶，每桶 2 张刚好凑 bs=2
    ars = [(1024, 1024), (1536, 1024), (1024, 1536), (1280, 1024)]
    for i, size in enumerate(ars * 2):
        _write(tmp_path / "1_data", f"img{i}", size)

    ds = ImageDataset(tmp_path, 1024, BucketManager(1024), prefer_json=False)
    sampler = BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=True, seed=0)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn)

    n = 0
    for batch in dl:
        px = batch["pixel_values"]   # stack 成功本身就证明 batch 内尺寸一致
        assert px.ndim == 4
        n += 1
    assert n > 0
