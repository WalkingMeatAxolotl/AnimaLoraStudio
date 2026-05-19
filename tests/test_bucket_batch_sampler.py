"""ARB BucketBatchSampler 回归：桶尾零头不丢图 + __len__ 按桶算。

旧默认 drop_last=True 写死 → 桶 size < batch_size 时整桶被丢（单图奇形比例图永远训不到）；
__len__ 用全局 n/bs，没按桶各自零头算 → steps_per_epoch / total_steps / scheduler 都偏。
对齐 kohya sd-scripts / ostris ai-toolkit 的「短 batch 不丢」语义。
"""
from __future__ import annotations

from training.dataset import BucketBatchSampler


class _MockBucketedDataset:
    """模拟 CachedLatentDataset：暴露 bucket_for_index 给 BucketBatchSampler 走桶分支。"""
    def __init__(self, bucket_for_index):
        self.bucket_for_index = list(bucket_for_index)

    def __len__(self):
        return len(self.bucket_for_index)

    def __getitem__(self, idx):
        return idx


def test_small_bucket_yields_short_batch_when_drop_last_false():
    bucket_for_index = [(1, 1), (2, 2), (2, 2)]
    ds = _MockBucketedDataset(bucket_for_index)
    sampler = BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=False)
    all_indices = sorted(idx for batch in sampler for idx in batch)
    assert all_indices == [0, 1, 2]


def test_small_bucket_dropped_when_drop_last_true():
    """记录历史行为：drop_last=True 时单图桶被整桶丢（回归保护，防误改默认）。"""
    bucket_for_index = [(1, 1), (2, 2), (2, 2)]
    ds = _MockBucketedDataset(bucket_for_index)
    sampler = BucketBatchSampler(ds, batch_size=2, drop_last=True, shuffle=False)
    all_indices = sorted(idx for batch in sampler for idx in batch)
    assert all_indices == [1, 2]


def test_len_counts_per_bucket_not_globally_drop_last_false():
    bucket_for_index = [(1, 1), (1, 1), (1, 1), (2, 2)]
    ds = _MockBucketedDataset(bucket_for_index)
    sampler = BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=False)
    assert len(sampler) == 3
    assert sum(1 for _ in sampler) == 3


def test_len_counts_per_bucket_not_globally_drop_last_true():
    bucket_for_index = [(1, 1)] * 3 + [(2, 2)] * 5
    ds = _MockBucketedDataset(bucket_for_index)
    sampler = BucketBatchSampler(ds, batch_size=2, drop_last=True, shuffle=False)
    assert len(sampler) == 3
    assert sum(1 for _ in sampler) == 3


def test_len_matches_iter_count_realistic_distribution():
    sizes = [30, 27, 25, 24, 23]
    assert sum(sizes) == 129
    bucket_for_index = []
    for i, n in enumerate(sizes):
        bucket_for_index += [(i, i)] * n
    ds = _MockBucketedDataset(bucket_for_index)

    sampler_drop = BucketBatchSampler(ds, batch_size=2, drop_last=True, shuffle=False)
    actual_drop = sum(1 for _ in sampler_drop)
    assert len(sampler_drop) == actual_drop == 63

    sampler_keep = BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=False)
    actual_keep = sum(1 for _ in sampler_keep)
    assert len(sampler_keep) == actual_keep == 66


def test_len_falls_back_to_global_when_no_bucket_info():
    class _PlainDataset:
        def __init__(self, n):
            self._n = n
        def __len__(self):
            return self._n
        def __getitem__(self, idx):
            return idx

    sampler = BucketBatchSampler(_PlainDataset(10), batch_size=3, drop_last=True, shuffle=False)
    assert len(sampler) == 3
    sampler = BucketBatchSampler(_PlainDataset(10), batch_size=3, drop_last=False, shuffle=False)
    assert len(sampler) == 4
