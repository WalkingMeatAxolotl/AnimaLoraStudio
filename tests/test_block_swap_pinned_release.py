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
            self.closed = False

        def close(self):
            self.closed = True

    swap = _FakeSwap()
    cache.block_swap = swap

    cache.unload()

    # close() 而非 detach()：光摘钩子不放开 param.data，主副本仍被模型钉住
    assert swap.closed, "必须 close（摘钩子 + 参数指走 + 丢主副本）"
    assert cache.block_swap is None and cache.blocks_to_swap == 0
    assert calls, "block swap 用过就必须归还 pinned host cache"


def test_unload_releases_pinned_even_without_cuda_available(monkeypatch):
    """**回归**：pinned 归还不能嵌在 `if torch.cuda.is_available()` 里。

    用过 block swap 就必然有 pinned 内存待还，这件事不该取决于「卸载这一刻
    CUDA 还可不可用」。首版嵌在 CUDA 分支内 —— 本地（有 GPU）绿、CI（Linux
    无 GPU）红，正是 CONTRIBUTING 测试卫生 #2 说的平台分支陷阱。
    """
    import torch

    d = _daemon()
    calls = []
    monkeypatch.setattr(d, "_release_pinned_host_cache", lambda: calls.append(1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cache = d.ModelCache()
    cache.model = object()
    cache.blocks_to_swap = 28

    class _FakeSwap:
        def close(self):
            pass

    cache.block_swap = _FakeSwap()
    cache.unload()

    assert calls, "无 CUDA 环境下也必须归还 pinned（函数本身是安全 no-op）"


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


def test_unload_reclaims_leftovers_even_when_model_absent(monkeypatch):
    """**回归**：模型未加载 ≠ 显存干净。加载中途 OOM 后 model 仍是 None，
    但异常 traceback 的循环引用钉着半上卡的 state_dict（refcount 收不掉）——
    手动「释放缓存」落到早退分支时必须无条件清扫，否则按钮空转
    （真机实测 20GB 纹丝不动，daemon 却汇报「已卸载」）。"""
    d = _daemon()
    calls = []
    monkeypatch.setattr(d, "_reclaim_cuda_leftovers", lambda: calls.append(1))

    cache = d.ModelCache()
    assert not cache.loaded
    cache.unload()

    assert calls, "早退分支必须清扫（gc + empty_cache + pinned 归还）"


def test_generate_worker_reclaims_leftovers_on_failure(monkeypatch):
    """任务失败后 worker 必须清扫一次 —— OOM 残骸不该留到下次成功 load。
    清扫必须发生在 except 块结束（异常对象被隐式 del）之后，traceback
    钉住的 frame locals 才收得掉；这里只测接线（有没有调 + 失败才调）。"""
    from pathlib import Path as _P

    d = _daemon()
    calls = []
    monkeypatch.setattr(d, "_reclaim_cuda_leftovers", lambda: calls.append(1))
    monkeypatch.setattr(d, "_emit_for", lambda *a, **k: None)

    def _boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(d, "_run_generate", _boom)
    d._run_generate_worker("req-x", 1, {}, _P("."), __import__("threading").Event())
    assert calls, "失败路径必须清扫"

    calls.clear()
    monkeypatch.setattr(d, "_run_generate", lambda *a, **k: None)
    d._run_generate_worker("req-y", 2, {}, _P("."), __import__("threading").Event())
    assert not calls, "成功路径不清扫（别把有用的 allocator cache 也倒掉）"
