"""block swap 的 pinned 内存归还（docs/design/block-swap.md §9.7）。

**丢引用不等于还内存**：pinned 走 PyTorch 独立的 host caching allocator，
真机实测 pin 6GB 后 ``del`` + ``gc.collect()`` 归还 **0 字节**，必须显式清
host cache 才归还 8GB。block swap 的主副本可达 11GB+，漏掉这步 = 卸载后仍
长期占着内存，且页锁定内存连换页都不行，其他程序完全用不到。

本文件测的是**接线**（unload 路径有没有调到），不是分配器算法 —— 那正是
第一版写错的地方（注释写「随之释放」，实际一字节没还）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _daemon():
    import anima_daemon

    return anima_daemon


def test_unload_releases_pinned_host_cache_when_swap_was_active(monkeypatch):
    d = _daemon()
    calls = []
    monkeypatch.setattr(d, "_release_pinned_host_cache", lambda: calls.append(1))

    cache = d.ModelCache()
    cache.model = object()          # 让 loaded 为 True
    cache.blocks_to_swap = 14

    class _FakeSwap:
        def __init__(self):
            self.detached = False

        def detach(self):
            self.detached = True

    swap = _FakeSwap()
    cache.block_swap = swap

    cache.unload()

    assert swap.detached, "必须摘钩子"
    assert cache.block_swap is None and cache.blocks_to_swap == 0
    assert calls, "block swap 用过就必须归还 pinned host cache"


def test_unload_skips_host_cache_release_without_swap(monkeypatch):
    """没用过 block swap 就不动 host cache —— 只清理自己分配的，别影响别人。"""
    d = _daemon()
    calls = []
    monkeypatch.setattr(d, "_release_pinned_host_cache", lambda: calls.append(1))

    cache = d.ModelCache()
    cache.model = object()
    cache.unload()

    assert not calls


def test_release_helper_is_silent_when_api_missing(monkeypatch):
    """内部 API 缺失/失败要静默 —— 清理失败不该让卸载崩掉。"""
    import torch

    d = _daemon()

    def _boom():
        raise RuntimeError("no such API")

    monkeypatch.setattr(torch._C, "_host_emptyCache", _boom, raising=False)
    d._release_pinned_host_cache()  # 不应抛出
