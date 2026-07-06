"""NaViT / Patch-n-Pack 块对角打包——token 预算打包器单测。

纯 Python（无 torch）：验证 ``pack_indices_by_budget`` / ``pack_indices_ffd_windowed``
/ ``NavitPackBatchSampler`` 产出的包 (a) 除不可避免的超大单图外 token 总和 ≤ 预算、
(b) 覆盖每个样本恰好一次、(c) 遵守可选的 per-pack 图片上限、(d) 跨 epoch reshuffle。
"""
from __future__ import annotations

import pytest

from training.dataset import (
    NavitPackBatchSampler,
    dataset_token_counts,
    pack_indices_by_budget,
    pack_indices_ffd_windowed,
)


class _FakeDataset:
    """暴露 token_count_for_index 的最小数据集（模拟 CachedLatentDataset）。"""

    def __init__(self, token_counts):
        self.token_count_for_index = list(token_counts)

    def __len__(self):
        return len(self.token_count_for_index)


class _FakeCachedDataset:
    """模拟 NaViT 路径：token_count_for_index 全 0，每图尺寸在 bucket_for_index = (h, w)。"""

    def __init__(self, latent_shapes):
        self.bucket_for_index = list(latent_shapes)
        self.token_count_for_index = [0] * len(latent_shapes)

    def __len__(self):
        return len(self.bucket_for_index)


# ----------------------------------------------------------------- next-fit


def test_next_fit_sum_within_budget_and_full_coverage():
    counts = [10, 20, 30, 40, 15, 25, 5, 50]
    packs = pack_indices_by_budget(counts, token_budget=60, order=list(range(len(counts))))
    seen = sorted(i for p in packs for i in p)
    assert seen == list(range(len(counts)))
    for p in packs:
        s = sum(counts[i] for i in p)
        assert s <= 60 or len(p) == 1


def test_next_fit_oversized_image_becomes_singleton():
    packs = pack_indices_by_budget([100, 10, 10], token_budget=50, order=[0, 1, 2])
    assert [0] in packs


def test_next_fit_flushes_on_overflow():
    packs = pack_indices_by_budget([30, 30, 30], token_budget=60, order=[0, 1, 2])
    assert packs == [[0, 1], [2]]


def test_next_fit_max_images_per_pack_cap():
    packs = pack_indices_by_budget(
        [1] * 10, token_budget=1000, order=list(range(10)), max_images_per_pack=3,
    )
    assert all(len(p) <= 3 for p in packs)
    assert sum(len(p) for p in packs) == 10


# --------------------------------------------------------------------- FFD


def test_ffd_full_coverage_and_budget():
    counts = [10, 20, 30, 40, 15, 25, 5, 50]
    packs = pack_indices_ffd_windowed(counts, token_budget=60, order=list(range(len(counts))), window=0)
    seen = sorted(i for p in packs for i in p)
    assert seen == list(range(len(counts)))
    for p in packs:
        s = sum(counts[i] for i in p)
        assert s <= 60 or len(p) == 1


def test_ffd_packs_no_more_bins_than_next_fit():
    counts = [40, 35, 20, 25, 30, 10, 50, 15]
    nf = pack_indices_by_budget(counts, token_budget=60, order=list(range(len(counts))))
    ffd = pack_indices_ffd_windowed(counts, token_budget=60, order=list(range(len(counts))), window=0)
    assert len(ffd) <= len(nf)


def test_ffd_oversized_image_becomes_singleton():
    packs = pack_indices_ffd_windowed([100, 10, 10], token_budget=50, order=[0, 1, 2], window=0)
    assert [0] in packs


def test_ffd_max_images_per_pack_cap():
    packs = pack_indices_ffd_windowed(
        [1] * 10, token_budget=1000, order=list(range(10)),
        max_images_per_pack=3, window=0,
    )
    assert all(len(p) <= 3 for p in packs)
    assert sum(len(p) for p in packs) == 10


def test_ffd_windowing_preserves_coverage():
    counts = [7, 11, 13, 17, 19, 23, 29, 31, 37, 5, 8]
    packs = pack_indices_ffd_windowed(counts, token_budget=50, order=list(range(len(counts))), window=4)
    seen = sorted(i for p in packs for i in p)
    assert seen == list(range(len(counts)))


def test_ffd_sampler_varies_across_epochs_with_window():
    ds = _FakeDataset([7, 11, 13, 17, 19, 23, 29, 31, 37, 5, 8, 12])
    sampler = NavitPackBatchSampler(ds, token_budget=50, shuffle=True, seed=1, strategy="ffd", ffd_window=4)
    sampler.set_epoch(0)
    p0 = list(sampler)
    sampler.set_epoch(1)
    p1 = list(sampler)
    assert sorted(i for p in p0 for i in p) == list(range(len(ds)))
    assert sorted(i for p in p1 for i in p) == list(range(len(ds)))
    assert p0 != p1  # windowed shuffle varies packs


def test_invalid_strategy_raises():
    ds = _FakeDataset([10, 20, 30])
    with pytest.raises(ValueError):
        NavitPackBatchSampler(ds, token_budget=60, strategy="bogus")


# ------------------------------------------------------- NavitPackBatchSampler


def test_sampler_full_coverage_default():
    ds = _FakeDataset([10, 20, 30, 40, 15, 25, 5, 50])
    sampler = NavitPackBatchSampler(ds, token_budget=60, shuffle=True, seed=1)
    packs = list(sampler)
    seen = sorted(i for p in packs for i in p)
    assert seen == list(range(len(ds)))
    assert len(sampler) == len(packs)


def test_sampler_epoch_changes_packing():
    ds = _FakeDataset([7, 11, 13, 17, 19, 23, 29, 31, 37])
    sampler = NavitPackBatchSampler(ds, token_budget=50, shuffle=True, seed=1)
    sampler.set_epoch(0)
    p0 = list(sampler)
    sampler.set_epoch(1)
    p1 = list(sampler)
    assert sorted(i for p in p0 for i in p) == list(range(len(ds)))
    assert sorted(i for p in p1 for i in p) == list(range(len(ds)))
    assert p0 != p1  # reshuffle


def test_token_counts_derived_from_latent_shape_when_token_count_zero():
    """NaViT 路径 token_count_for_index 全 0 时，从 bucket_for_index (h//2)*(w//2) 推导。"""
    ds = _FakeCachedDataset([(128, 128), (96, 160), (160, 96)])
    counts = dataset_token_counts(ds, patch_spatial=2)
    assert counts == [64 * 64, 48 * 80, 80 * 48]


def test_pack_respects_budget_on_cached_path():
    ds = _FakeCachedDataset([(128, 128)] * 5)  # 4096 tokens each
    sampler = NavitPackBatchSampler(ds, token_budget=8192, shuffle=False)
    packs = list(sampler)
    for p in packs:
        assert len(p) <= 2  # 2*4096 == 8192, never 3
    assert sum(len(p) for p in packs) == 5


def test_fail_fast_on_all_zero_token_counts():
    ds = _FakeDataset([0, 0, 0])
    with pytest.raises(RuntimeError):
        NavitPackBatchSampler(ds, token_budget=8192)


def test_drop_last_removes_trailing_underfilled_pack():
    ds = _FakeDataset([60, 60, 5])
    sampler = NavitPackBatchSampler(ds, token_budget=60, shuffle=False, drop_last=True)
    packs = list(sampler)
    assert packs == [[0], [1]]  # last pack (5 < 60) dropped
