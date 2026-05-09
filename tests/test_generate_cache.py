"""generate_cache 模块单测（commit 10）。"""
from __future__ import annotations

import pytest

from studio.services import generate_cache


@pytest.fixture(autouse=True)
def _clear():
    generate_cache.clear_all()
    yield
    generate_cache.clear_all()


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
