"""PinnedPacker —— 换出层权重打包进 2 的幂大块的 pinned 内存（5080 真机案例修复）。

背景：PyTorch host caching allocator 把**每次** pinned 分配向上取整到 2 的幂
（CachingHostAllocator.h PowerOf2Ceil）。krea2 逐张量 pin 时 28 层 11.32GB 的权重
实际锁定 16.63GB（1.47×），Windows 对 cudaHostAlloc 的上限约为物理内存一半，32GB
内存机器就此撞死；而预算护栏按 11.32GB 放行。

不需要 CUDA：packer 的分配函数可注入，用普通 uint8 张量模拟 pinned 大块；
断言的是**分配的尺寸序列**（必须全是 2 的幂、总量贴近权重字节）与 view 语义。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training import block_swap as bs  # noqa: E402
from training.block_swap import PinnedAllocationError, PinnedPacker  # noqa: E402

_MIB = 1024 ** 2


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class _FakeAlloc:
    """记录每次分配尺寸的假 pinned 分配器（返回普通 uint8 张量）。"""

    def __init__(self, fail_at_total: int | None = None) -> None:
        self.sizes: list[int] = []
        self.fail_at_total = fail_at_total

    def __call__(self, nbytes: int) -> torch.Tensor:
        if self.fail_at_total is not None and sum(self.sizes) + nbytes > self.fail_at_total:
            raise RuntimeError("CUDA error: out of memory")
        self.sizes.append(nbytes)
        return torch.empty(nbytes, dtype=torch.uint8)


def test_pow2_helpers():
    assert [bs._pow2_ceil(n) for n in (0, 1, 2, 3, 4, 5, 1000)] == [1, 1, 2, 4, 4, 8, 1024]
    assert bs._pow2_decomposition(0) == []
    assert bs._pow2_decomposition(11) == [8, 2, 1]
    assert bs._pow2_decomposition(64 * _MIB) == [64 * _MIB]


def test_preallocates_binary_decomposition_of_total():
    """总量按粒度取整后二进制分解、每块恰为 2 的幂，且在构造时全部分配完。"""
    alloc = _FakeAlloc()
    total = 100 * _MIB  # 取整到 128MB → 一块 128MB
    packer = PinnedPacker(total, granularity=64 * _MIB, allocate=alloc)
    assert alloc.sizes == [128 * _MIB]
    assert packer.num_chunks == 1 and packer.allocated_bytes == 128 * _MIB

    alloc2 = _FakeAlloc()
    total2 = 11 * 1024 * _MIB + 300 * _MIB  # 11.29GB → 取整 11.3125GB = 8G+2G+1G+256M+64M
    packer2 = PinnedPacker(total2, granularity=64 * _MIB, allocate=alloc2)
    assert alloc2.sizes == [8192 * _MIB, 2048 * _MIB, 1024 * _MIB, 256 * _MIB, 64 * _MIB]
    assert all(_is_pow2(s) for s in alloc2.sizes)
    assert packer2.allocated_bytes - total2 < 64 * _MIB


def test_pin_returns_aligned_views_with_same_content_dtype_shape():
    alloc = _FakeAlloc()
    tensors = [
        torch.randn(37, 53, dtype=torch.float32),
        torch.randn(5, dtype=torch.bfloat16),
        torch.tensor(3.5, dtype=torch.float32),          # 0-dim 标量
        torch.randn(0, 8, dtype=torch.float16),          # 空张量
        torch.randn(16, 16).to(torch.float8_e4m3fn),     # fp8 原样
        torch.randn(9, 7, dtype=torch.float32).t(),      # 非连续
    ]
    total = sum(t.numel() * t.element_size() for t in tensors)
    packer = PinnedPacker(total, allocate=alloc)
    outs = [packer.pin(t) for t in tensors]

    assert packer.num_chunks == 1 and packer.overflow_chunks == 0
    chunk_ptr = packer._chunks[0][0].data_ptr()
    for src, out in zip(tensors, outs):
        assert out.shape == src.shape and out.dtype == src.dtype
        assert out.is_contiguous()
        if src.dtype == torch.float8_e4m3fn:
            assert torch.equal(out.view(torch.uint8), src.contiguous().view(torch.uint8))
        else:
            assert torch.equal(out, src)
        # 同一大块上的 view，偏移 256B 对齐
        storage_ptr = out.untyped_storage().data_ptr()
        assert storage_ptr == chunk_ptr
        assert (out.data_ptr() - chunk_ptr) % PinnedPacker.ALIGN == 0
    assert packer.packed_bytes == total

    # 改 view 等于改大块（param.data = view 后就是靠这个语义工作的）
    outs[0].fill_(1.0)
    assert torch.equal(outs[0], torch.ones_like(tensors[0]))


def test_pin_with_dtype_casts_on_the_fly():
    alloc = _FakeAlloc()
    src = torch.randn(8, 8, dtype=torch.float32)
    packer = PinnedPacker(src.numel() * 2, allocate=alloc)
    out = packer.pin(src, dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, src.to(torch.bfloat16))
    assert packer.packed_bytes == src.numel() * 2


def test_overflow_opens_pow2_chunk_never_worse_than_per_tensor_pin():
    """装不进任何块的张量单独开 pow2_ceil(nbytes) 的块 —— 等价于旧的逐张量 pin。"""
    alloc = _FakeAlloc()
    packer = PinnedPacker(64 * _MIB, allocate=alloc)  # 一块 64MB
    big = torch.empty(100 * _MIB, dtype=torch.uint8)  # 100MB 塞不进
    out = packer.pin(big)
    assert out.shape == big.shape
    assert packer.overflow_chunks == 1
    assert alloc.sizes == [64 * _MIB, 128 * _MIB]
    # 溢出块同样参与后续 best-fit：128MB 块剩 28MB < 64MB 块剩 64MB → 小张量落溢出块
    small = torch.empty(10 * _MIB, dtype=torch.uint8)
    out2 = packer.pin(small)
    ptr2 = out2.untyped_storage().data_ptr()  # 先取出来再断言，避免失败时 repr 整块存储
    assert ptr2 == packer._chunks[1][0].data_ptr()
    assert alloc.sizes == [64 * _MIB, 128 * _MIB]  # 没有再开块


def test_zero_total_means_lazy_allocation():
    alloc = _FakeAlloc()
    packer = PinnedPacker(0, allocate=alloc)
    assert alloc.sizes == [] and packer.num_chunks == 0
    packer.pin(torch.empty(3 * _MIB, dtype=torch.uint8))
    assert alloc.sizes == [4 * _MIB]


def test_allocation_failure_is_actionable_and_fail_fast():
    """cudaHostAlloc 失败也报 "CUDA error: out of memory"，必须翻译成用户能行动的话。"""
    alloc = _FakeAlloc(fail_at_total=1024 * _MIB)
    with pytest.raises(PinnedAllocationError) as info:
        PinnedPacker(4096 * _MIB, allocate=alloc)
    msg = str(info.value)
    assert "blocks_to_swap" in msg and "不是显存不足" in msg and "GB" in msg
    assert isinstance(info.value, RuntimeError)  # 上层按 RuntimeError 兜底


def _krea2_swapped_sizes(blocks_to_swap: int, *, fp8: bool) -> list[int]:
    """真实 krea2 换出层的逐张量字节序列（meta 模型，不读盘）。"""
    from training.families.krea2.loader import (
        KREA2_CONFIG, SingleStreamDiT, _swapped_block_prefixes,
    )

    prefixes = _swapped_block_prefixes(KREA2_CONFIG, blocks_to_swap)
    with torch.device("meta"):
        probe = SingleStreamDiT(KREA2_CONFIG)
    sizes = []
    for name, p in probe.named_parameters():
        if not name.startswith(prefixes):
            continue
        per_elem = 1 if (fp8 and p.dim() == 2) else 2
        sizes.append(p.numel() * per_elem)
    return sizes


@pytest.mark.parametrize("blocks,fp8", [(28, True), (18, True), (14, True), (28, False)])
def test_real_krea2_layout_packs_within_three_percent(blocks: int, fp8: bool):
    """**回归**：真实 krea2 尺寸序列下，实际锁定 ≤ 权重字节 × 1.03（旧行为 1.47×）。

    用 ``alloc`` 直接给假块、用零内容 uint8 张量代替权重走完整 best-fit 装填，
    断言分配序列全是 2 的幂且总和贴近权重字节 —— 这就是 cudaHostAlloc 真正会
    锁住的量。
    """
    sizes = _krea2_swapped_sizes(blocks, fp8=fp8)
    raw = sum(sizes)
    per_tensor_pow2 = sum(bs._pow2_ceil(n) for n in sizes)
    assert per_tensor_pow2 / raw > 1.4  # 旧行为的浪费（问题成立的前提）

    # 只走装填记账（_reserve），不真搬 11GB：假块用零存储的 expand 张量顶替
    # （numel / dtype 与真块一致，_reserve 只看 numel）
    alloc = _ZeroStorageAlloc()
    packer = PinnedPacker(raw, allocate=alloc)
    for n in sizes:
        packer._reserve(n)
    assert all(_is_pow2(s) for s in alloc.sizes)
    assert packer.packed_bytes == raw
    assert packer.allocated_bytes == sum(alloc.sizes)
    assert packer.allocated_bytes / raw <= 1.03, (
        f"{blocks} 层 fp8={fp8}: 实际锁定 {packer.allocated_bytes / raw:.3f}× 权重"
    )


class _ZeroStorageAlloc(_FakeAlloc):
    def __call__(self, nbytes: int) -> torch.Tensor:
        self.sizes.append(nbytes)
        return torch.empty(1, dtype=torch.uint8).expand(nbytes)
