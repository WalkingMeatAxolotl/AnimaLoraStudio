"""ARB 切桶时归还 allocator 缓存（runtime/training/loop.py::_BucketSwitchCacheRelease，issue #505）。

背景：BucketBatchSampler 逐桶连续产出；切桶后新形状塞不进旧桶的 cached block，
reserved ≈ 旧峰值 + 新峰值。Windows WDDM 下 cudaMalloc 不失败而是溢到共享内存，
allocator 的「OOM → 释放缓存 → 重试」自愈永不触发，训练速度永久掉一半以上。
修法：latent 空间形状变化时 empty_cache 一次；同桶连续 batch 不清；navit 路径不走。

torch.cuda.empty_cache 在 CPU 上是 no-op，这里 monkeypatch 成计数器验证调用时机。
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from runtime.training import loop as loop_mod  # noqa: E402
from tests.test_loop_nonfinite_loss import _make_ctx  # noqa: E402


class _EmptyCacheSpy:
    def __init__(self, monkeypatch):
        self.calls = 0
        monkeypatch.setattr(torch.cuda, "empty_cache", self)

    def __call__(self):
        self.calls += 1


def _lat(h, w, bs=1):
    return torch.randn(bs, 4, 1, h, w)


# ---------------------------------------------------------------- 类级语义

def test_same_shape_never_releases(monkeypatch):
    spy = _EmptyCacheSpy(monkeypatch)
    tracker = loop_mod._BucketSwitchCacheRelease("cpu")
    for _ in range(5):
        tracker.observe(_lat(8, 8))
    assert spy.calls == 0


def test_release_only_on_shape_change(monkeypatch):
    spy = _EmptyCacheSpy(monkeypatch)
    tracker = loop_mod._BucketSwitchCacheRelease("cpu")
    tracker.observe(_lat(8, 8))     # 首个 batch：没有「上一个桶」，不清
    tracker.observe(_lat(8, 8))
    assert spy.calls == 0
    tracker.observe(_lat(6, 10))    # 切桶
    assert spy.calls == 1
    tracker.observe(_lat(6, 10))    # 同桶
    assert spy.calls == 1
    tracker.observe(_lat(8, 8))     # 再切
    assert spy.calls == 2


def test_batch_size_change_within_bucket_does_not_release(monkeypatch):
    # 同桶尾批（drop_last=False）bs 变小：张量更小、塞得进 cached block，不需要清
    spy = _EmptyCacheSpy(monkeypatch)
    tracker = loop_mod._BucketSwitchCacheRelease("cpu")
    tracker.observe(_lat(8, 8, bs=2))
    tracker.observe(_lat(8, 8, bs=1))
    assert spy.calls == 0


def test_each_switch_logged_at_debug_only(monkeypatch, caplog):
    _EmptyCacheSpy(monkeypatch)
    tracker = loop_mod._BucketSwitchCacheRelease("cpu")
    with caplog.at_level("DEBUG", logger=loop_mod.logger.name):
        tracker.observe(_lat(8, 8))
        tracker.observe(_lat(6, 10))
        tracker.observe(_lat(8, 8))
        tracker.observe(_lat(6, 10))
    recs = [r for r in caplog.records if "ARB bucket switch" in r.getMessage()]
    assert [r.levelname for r in recs] == ["DEBUG"] * 3
    assert "8x8 -> 6x10" in recs[0].getMessage()
    assert "6x10 -> 8x8" in recs[1].getMessage()

    caplog.clear()
    with caplog.at_level("INFO", logger=loop_mod.logger.name):
        tracker.observe(_lat(8, 8))
    assert not [r for r in caplog.records if "ARB bucket switch" in r.getMessage()]


# ---------------------------------------------------------------- loop 接线

def _batch(h, w):
    return {"captions": ["c"], "latents": _lat(h, w)}


def test_loop_releases_between_buckets_not_within(tmp_path, monkeypatch):
    # 桶 A ×2 → 桶 B ×2 → 桶 A ×1：两次切桶 → 恰好 2 次 empty_cache
    spy = _EmptyCacheSpy(monkeypatch)
    batches = [_batch(8, 8), _batch(8, 8), _batch(6, 10), _batch(6, 10), _batch(8, 8)]
    ctx = _make_ctx(tmp_path, batches, monkeypatch, grad_accum=1)
    loop_mod.run(ctx)
    assert ctx.global_step == 5
    assert spy.calls == 2


def test_loop_single_bucket_never_releases(tmp_path, monkeypatch):
    spy = _EmptyCacheSpy(monkeypatch)
    ctx = _make_ctx(tmp_path, [_batch(8, 8)] * 4, monkeypatch, grad_accum=1)
    loop_mod.run(ctx)
    assert ctx.global_step == 4
    assert spy.calls == 0
