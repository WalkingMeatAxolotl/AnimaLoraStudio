"""BucketManager ARB 桶生成单测。

锁定两件事：
1. 默认参数（base=1024, R=2.0）= 37 桶，且与前端 `trainBuckets.ts` 镜像一致
   （`trainBuckets.test.ts` 也断言 37）。
2. min/max 由 (base, R) 派生后小 base 不再退化——base=512 曾因写死 min_reso=512
   只剩 512×512 一个方桶（ARB 失效），见
   `docs/design/multi-resolution-training-design.md` §6.3。
"""
from __future__ import annotations

from runtime.training.dataset import BucketManager


def test_default_base_yields_37_buckets() -> None:
    m = BucketManager()  # base=1024, R=2.0
    assert len(m.buckets) == 37
    # min/max 现在派生自 (base, R)，不再是写死的 512 / 2048
    assert (m.min_reso, m.max_reso) == (640, 1536)


def test_default_contains_canonical_anchors() -> None:
    m = BucketManager()
    for wh in [(1024, 1024), (1152, 896), (1216, 832), (896, 1152), (832, 1216)]:
        assert wh in m.buckets


def test_all_buckets_within_area_band_and_ar_cap() -> None:
    m = BucketManager()
    base_area = 1024 * 1024
    for w, h in m.buckets:
        assert abs(w * h - base_area) / base_area <= 0.1 + 1e-9
        assert max(w / h, h / w) <= 2.0 + 1e-9
        assert w % 64 == 0 and h % 64 == 0


def test_small_base_keeps_aspect_ratio_variety() -> None:
    # 回归：base=512 在写死 min_reso=512 时只能生成 512×512（面积 ±10% 容不下
    # 任何非方桶），整个数据集被压成正方、ARB 失效。派生 min/max 后恢复多样性。
    m = BucketManager(512)
    assert len(m.buckets) > 1
    assert any(w != h for w, h in m.buckets)
    assert any(w > h for w, h in m.buckets)  # 有横桶
    assert any(h > w for w, h in m.buckets)  # 有竖桶


def test_aspect_ratio_limit_widens_bucket_set() -> None:
    narrow = BucketManager(1024, aspect_ratio_limit=2.0)
    wide = BucketManager(1024, aspect_ratio_limit=3.0)
    narrow_max = max(max(w / h, h / w) for w, h in narrow.buckets)
    wide_max = max(max(w / h, h / w) for w, h in wide.buckets)
    assert narrow_max <= 2.0 + 1e-9
    assert 2.0 < wide_max <= 3.0 + 1e-9
    assert len(wide.buckets) > len(narrow.buckets)


def test_explicit_min_max_override_is_respected() -> None:
    # 显式传 min/max 跳过派生（cache 测试依赖：强制单一方桶）。
    m = BucketManager(256, min_reso=256, max_reso=256, step=64)
    assert m.buckets == [(256, 256)]


def test_get_bucket_snaps_by_aspect_ratio_only() -> None:
    m = BucketManager()
    # 极宽 → 最宽桶（AR≈2.0）；图片绝对尺寸不影响选桶。
    bw, bh = m.get_bucket(5000, 1000)
    assert abs(bw / bh - 2.0) < 0.1
    # 同一 AR、不同绝对尺寸 → 同一桶
    assert m.get_bucket(800, 600) == m.get_bucket(1600, 1200)
