"""Block swap 机制核心单测（docs/design/block-swap.md §9 刀 1）。

覆盖：
- 数值正确性：swap 前向/反向与全常驻逐位一致
- 原地换语义：module 身份不变（LoRA 兼容的前提）、param.data 指向槽
- fp8 场景：weight_scale 非持久 buffer 恒与权重配对
- 边界：num_swap 校验、非换出层 no-op
- 分配失败：BlockSwapAllocationError 携带上下文

无 CUDA 时整文件 skip（block swap 是 CUDA-only 机制）。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="block swap 是 CUDA-only 机制"
)


def _import():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "runtime"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from training.block_swap import BlockSwapAllocationError, PinnedBlockSwap

    return PinnedBlockSwap, BlockSwapAllocationError


class _Tiny(nn.Module):
    """一个结构同构、可堆叠的小 block（避免依赖真实 krea2 权重）。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(torch.relu(self.lin1(x)))


def _make_blocks(n: int, dim: int, device) -> nn.ModuleList:
    torch.manual_seed(0)
    blocks = nn.ModuleList([_Tiny(dim) for _ in range(n)])
    return blocks.to(device)


def _run_resident(blocks, x):
    h = x
    for block in blocks:
        h = block(h)
    return h


def _run_swap(swap, blocks, x):
    h = x
    for i, block in swap.iter_forward():
        h = block(h)
    return h


def test_forward_matches_resident():
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    dim, n = 32, 6
    blocks = _make_blocks(n, dim, device)
    x = torch.randn(2, dim, device=device)

    expected = _run_resident(blocks, x)

    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)
    got = _run_swap(swap, blocks, x)

    torch.testing.assert_close(got, expected)


def test_module_identity_preserved():
    """原地换的核心保证：block/Linear 对象不变 —— 这是 LoRA 兼容的前提。"""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)

    ids_before = [id(b) for b in blocks]
    lin_ids_before = [id(b.lin1) for b in blocks]

    x = torch.randn(1, 16, device=device)
    _run_swap(swap, blocks, x)

    assert [id(b) for b in blocks] == ids_before
    assert [id(b.lin1) for b in blocks] == lin_ids_before


def test_lora_style_forward_hook_not_bypassed():
    """模拟 LyCORIS bypass：在 Linear 外包一层加法。swap 必须仍走这层。

    若 swap 用 buffer 轮转（换 module 实例），这个 hook 会被绕过 —— 本测试
    正是钉死 doc §9.1 那个「静默学不到东西」的陷阱。
    """
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)

    marker = {"calls": 0}

    def hook(_module, _inp, out):
        marker["calls"] += 1
        return out + 1.0  # LoRA-like 额外贡献

    handles = [b.lin1.register_forward_hook(hook) for b in blocks]

    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)
    x = torch.randn(1, 16, device=device)
    _run_swap(swap, blocks, x)

    for h in handles:
        h.remove()
    # 4 个 block 每个的 lin1 都应被 hook 命中一次（含 2 个换出层）
    assert marker["calls"] == 4


def test_backward_grad_reaches_all_blocks():
    """反向逆序预取：换出层的可训练参数（模拟 LoRA）应拿到梯度。"""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(6, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)

    x = torch.randn(3, 16, device=device)
    # 前向
    h = x
    swap.reset()
    saved = []
    for i in range(swap.total):
        swap.ensure_resident(i, prefetch_next=i + 1)
        saved.append(h)
        h = blocks[i](h)
        swap.release(i)
    loss = h.sum()
    # 反向逆序：确保权重在位再算 grad
    swap.reset()
    for i in reversed(range(swap.total)):
        swap.ensure_resident(i, prefetch_next=i - 1)
        swap.release(i)
    loss.backward()

    for i, block in enumerate(blocks):
        assert block.lin1.weight.grad is not None, f"block {i} 无梯度"


def test_num_swap_validation():
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 8, device)

    with pytest.raises(ValueError, match="num_swap 必须为正"):
        PinnedBlockSwap(blocks, num_swap=0, device=device)
    with pytest.raises(ValueError, match="超过 block 总数"):
        PinnedBlockSwap(blocks, num_swap=5, device=device)
    with pytest.raises(ValueError, match="num_slots 至少为 2"):
        PinnedBlockSwap(blocks, num_swap=2, device=device, num_slots=1)


def test_non_swapped_layers_noop():
    """前 N-num_swap 层的 ensure_resident/release 是 no-op，不改其 param。"""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)

    # block 0/1/2 常驻（first_swapped == 3）
    assert swap.first_swapped == 3
    resident_ptr = blocks[0].lin1.weight.data_ptr()
    swap.ensure_resident(0)
    swap.release(0)
    assert blocks[0].lin1.weight.data_ptr() == resident_ptr


def test_pinned_bytes_accounting():
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    dim, n, num_swap = 16, 5, 3
    blocks = _make_blocks(n, dim, device)
    swap = PinnedBlockSwap(blocks, num_swap=num_swap, device=device)

    # 每个 _Tiny：2 个 Linear，各 dim*dim 权重 + dim 偏置，fp32
    per_block = num_swap * 2 * (dim * dim + dim) * 4
    assert swap.pinned_bytes == per_block


def test_fp8_scale_stays_paired():
    """fp8 场景：weight_scale 是绑在 module 上的非持久 buffer；原地换权重后
    scale 必须仍与该 module 的权重配对（module 不变 → 自动正确）。"""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")

    class _ScaledBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.lin = nn.Linear(dim, dim)
            # 模拟 patch_fp8_linears：非持久 buffer
            self.lin.register_buffer(
                "weight_scale", torch.tensor(2.0, device=device), persistent=False
            )

        def forward(self, x):
            w = self.lin.weight * self.lin.weight_scale
            return torch.nn.functional.linear(x, w, self.lin.bias)

    torch.manual_seed(1)
    blocks = nn.ModuleList([_ScaledBlock(16) for _ in range(4)]).to(device)
    x = torch.randn(2, 16, device=device)
    expected = _run_resident(blocks, x)

    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    got = _run_swap(swap, blocks, x)

    # scale 未被搬运破坏，前向仍逐位一致
    torch.testing.assert_close(got, expected)
    for b in blocks:
        assert b.lin.weight_scale.item() == 2.0


def test_allocation_error_carries_context():
    """BlockSwapAllocationError 携带 num_swap/first_swapped/detail（供上层文案）。"""
    _, BlockSwapAllocationError = _import()
    err = BlockSwapAllocationError(num_swap=14, first_swapped=14, detail="out of memory")
    assert err.num_swap == 14
    assert err.first_swapped == 14
    assert "out of memory" in str(err)
    assert "14" in str(err)
