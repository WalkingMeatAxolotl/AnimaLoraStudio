"""generate_cache 模块单测（commit 10 + commit 11 LRU）。"""
from __future__ import annotations

import pytest

from studio.services.inference import cache as generate_cache


@pytest.fixture(autouse=True)
def _clear():
    # 测试用 LRU 上限调到默认（避免上一个测试改了配置漏出）
    generate_cache.configure(
        max_count=generate_cache.DEFAULT_MAX_COUNT,
        max_bytes=generate_cache.DEFAULT_MAX_BYTES,
    )
    generate_cache.clear_all()
    yield
    generate_cache.clear_all()
    generate_cache.configure(
        max_count=generate_cache.DEFAULT_MAX_COUNT,
        max_bytes=generate_cache.DEFAULT_MAX_BYTES,
    )


def test_cache_and_get() -> None:
    generate_cache.cache_image(1, "a.png", b"PNG-A")
    generate_cache.cache_image(1, "b.png", b"PNG-B")
    generate_cache.cache_image(2, "c.png", b"PNG-C")

    assert generate_cache.get_image(1, "a.png") == b"PNG-A"
    assert generate_cache.get_image(1, "b.png") == b"PNG-B"
    assert generate_cache.get_image(2, "c.png") == b"PNG-C"
    assert generate_cache.get_image(99, "x.png") is None


def test_overwrite_same_key() -> None:
    generate_cache.cache_image(1, "a.png", b"V1")
    generate_cache.cache_image(1, "a.png", b"V2")
    assert generate_cache.get_image(1, "a.png") == b"V2"
    assert generate_cache.total_count() == 1


def test_list_filenames_per_task() -> None:
    generate_cache.cache_image(1, "b.png", b"B")
    generate_cache.cache_image(1, "a.png", b"A")
    generate_cache.cache_image(2, "z.png", b"Z")
    assert generate_cache.list_filenames(1) == ["a.png", "b.png"]
    assert generate_cache.list_filenames(2) == ["z.png"]
    assert generate_cache.list_filenames(99) == []


def test_drop_task() -> None:
    generate_cache.cache_image(1, "a.png", b"A")
    generate_cache.cache_image(1, "b.png", b"B")
    generate_cache.cache_image(2, "c.png", b"C")
    n = generate_cache.drop_task(1)
    assert n == 2
    assert generate_cache.get_image(1, "a.png") is None
    assert generate_cache.get_image(2, "c.png") == b"C"
    # drop 不存在的 task → 0
    assert generate_cache.drop_task(99) == 0


def test_total_bytes() -> None:
    generate_cache.cache_image(1, "a.png", b"x" * 100)
    generate_cache.cache_image(2, "b.png", b"x" * 50)
    assert generate_cache.total_bytes() == 150
    assert generate_cache.total_count() == 2


def test_clear_all() -> None:
    generate_cache.cache_image(1, "a.png", b"A")
    generate_cache.cache_image(2, "b.png", b"B")
    generate_cache.clear_all()
    assert generate_cache.total_count() == 0
    assert generate_cache.total_bytes() == 0


# ---------- commit 11：LRU ------------------------------------------------------


def test_lru_evicts_by_count() -> None:
    generate_cache.configure(max_count=3, max_bytes=10**9)
    for i in range(5):
        generate_cache.cache_image(1, f"img{i}.png", b"x")
    # 只保留最新 3 张
    assert generate_cache.total_count() == 3
    # img0/img1 被剔；img2/img3/img4 留下
    assert generate_cache.get_image(1, "img0.png") is None
    assert generate_cache.get_image(1, "img1.png") is None
    assert generate_cache.get_image(1, "img4.png") == b"x"


def test_lru_evicts_by_bytes() -> None:
    generate_cache.configure(max_count=10**6, max_bytes=300)
    generate_cache.cache_image(1, "a.png", b"x" * 100)
    generate_cache.cache_image(1, "b.png", b"x" * 100)
    generate_cache.cache_image(1, "c.png", b"x" * 100)
    # 总 300，处于上限；再加一张 100 → 应该剔最旧的 a
    generate_cache.cache_image(1, "d.png", b"x" * 100)
    assert generate_cache.get_image(1, "a.png") is None
    assert generate_cache.get_image(1, "d.png") == b"x" * 100
    assert generate_cache.total_bytes() == 300


def test_lru_get_marks_recent() -> None:
    """get_image 命中 → move_to_end，被 LRU 看作最新。"""
    generate_cache.configure(max_count=3, max_bytes=10**9)
    generate_cache.cache_image(1, "a.png", b"A")
    generate_cache.cache_image(1, "b.png", b"B")
    generate_cache.cache_image(1, "c.png", b"C")
    # 访问 a，让它变最新
    assert generate_cache.get_image(1, "a.png") == b"A"
    # 加 d → 现在最旧应该是 b（不是 a）
    generate_cache.cache_image(1, "d.png", b"D")
    assert generate_cache.get_image(1, "b.png") is None
    assert generate_cache.get_image(1, "a.png") == b"A"
    assert generate_cache.get_image(1, "d.png") == b"D"


def test_total_bytes_after_overwrite_and_drop() -> None:
    generate_cache.cache_image(1, "a.png", b"x" * 100)
    assert generate_cache.total_bytes() == 100
    # 同 key 覆盖，总字节应该跟着新 value
    generate_cache.cache_image(1, "a.png", b"x" * 50)
    assert generate_cache.total_bytes() == 50
    # drop_task 后归零
    generate_cache.drop_task(1)
    assert generate_cache.total_bytes() == 0


def test_configure_shrink_triggers_eviction() -> None:
    """运行时 configure 把上限调小，应该立刻 evict 多余的。"""
    for i in range(10):
        generate_cache.cache_image(1, f"i{i}.png", b"x")
    assert generate_cache.total_count() == 10
    generate_cache.configure(max_count=4)
    assert generate_cache.total_count() == 4
    # 留下的是最新 4 张：i6..i9
    assert generate_cache.get_image(1, "i9.png") == b"x"
    assert generate_cache.get_image(1, "i5.png") is None
