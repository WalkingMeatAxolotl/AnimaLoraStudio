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
    """底模 frozen —— 与真实流程一致（loader 里 model.requires_grad_(False)）。

    组件只管理冻结的基权重，可训练参数（LoRA）不归它管，所以测试基线必须冻结，
    否则什么都不会被换出。
    """
    torch.manual_seed(0)
    blocks = nn.ModuleList([_Tiny(dim) for _ in range(n)])
    blocks.requires_grad_(False)
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


def test_backward_grad_flows_through_swapped_blocks():
    """底模 frozen 时梯度仍须穿过换出层传到输入（LoRA 训练依赖这条链路）。"""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(6, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)
    swap.attach()

    x = torch.randn(3, 16, device=device, requires_grad=True)
    h = x
    for block in blocks:
        h = block(h)
    h.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


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


def test_shape_readable_after_construct():
    """构造后（权重已下 CPU）仍能读到正确 shape/dtype —— LyCORIS 在 swap 之后
    注入时要读基权重形状，指向 empty(0) 会让它读到错误形状（doc §9.1）。"""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    dim = 24
    blocks = _make_blocks(4, dim, device)
    shapes_before = [tuple(b.lin1.weight.shape) for b in blocks]

    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)

    for i, block in enumerate(blocks):
        assert tuple(block.lin1.weight.shape) == shapes_before[i]
        assert block.lin1.weight.dtype == torch.float32
    # 换出层的权重此刻应在 CPU（显存已释放），常驻层仍在 GPU
    assert blocks[swap.first_swapped].lin1.weight.device.type == "cpu"
    assert blocks[0].lin1.weight.device.type == "cuda"


def test_attach_hooks_forward_matches_resident():
    """attach() 的钩子路径：前向数值与全常驻一致，且不需要改模型循环。"""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(6, 32, device)
    x = torch.randn(2, 32, device=device)
    expected = _run_resident(blocks, x)

    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)
    swap.attach()
    got = _run_resident(blocks, x)  # 普通循环，钩子自动接管

    torch.testing.assert_close(got, expected)


def test_attach_with_gradient_checkpointing_backward():
    """**核心 claim 验证**：开 gradient checkpointing 后，反向的重算会再次触发
    block forward，pre-hook 随之按逆序换回权重 —— 所以反向无需单独编排。

    对照组是同一份权重的全常驻模型，比较 LoRA-style 可训练参数的梯度。
    """
    from torch.utils.checkpoint import checkpoint

    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    dim, n = 32, 6
    blocks = _make_blocks(n, dim, device)

    # 模拟 LoRA：每个 block 挂一个可训练小参数，底模 frozen
    for b in blocks:
        b.lora = nn.Parameter(torch.ones(dim, device=device) * 0.1)
    for b in blocks:
        b.lin1.requires_grad_(False)
        b.lin2.requires_grad_(False)

    def run(use_checkpoint: bool):
        h = torch.randn(2, dim, device=device, generator=torch.Generator(device).manual_seed(7))
        for b in blocks:
            def fwd(t, blk=b):
                return blk(t) * blk.lora
            h = checkpoint(fwd, h, use_reentrant=False) if use_checkpoint else fwd(h)
        return h.sum()

    # 基线：全常驻 + checkpoint
    run(True).backward()
    expected = [b.lora.grad.clone() for b in blocks]
    for b in blocks:
        b.lora.grad = None

    # swap + attach + checkpoint
    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)
    swap.attach()
    run(True).backward()

    for i, b in enumerate(blocks):
        assert b.lora.grad is not None, f"block {i} 无梯度"
        torch.testing.assert_close(b.lora.grad, expected[i], msg=f"block {i} 梯度不一致")


def test_trainable_params_are_not_managed():
    """可训练参数（LoRA）必须原地不动、常驻 GPU —— 不被当基权重换出。"""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)
    for b in blocks:
        b.lin1.requires_grad_(False)
        b.lin2.requires_grad_(False)
        b.lora = nn.Parameter(torch.ones(16, device=device))  # 可训练

    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)

    for b in list(blocks)[swap.first_swapped:]:
        assert b.lora.device.type == "cuda", "LoRA 参数被错误地换到了 CPU"
        assert b.lin1.weight.device.type == "cpu", "冻结基权重应已下 CPU"
    # 登记的 spec 里不应出现 lora
    for rel in range(swap.num_swap):
        names = [n for n, _s, _d in swap._param_specs[rel]]
        assert "lora" not in names


def test_params_added_after_construct_do_not_break_rebind():
    """构造后新增参数（LoRA 在 block 内建子模块的情形）不应让换入崩溃。

    回归：_rebind 曾遍历 named_parameters() 直接查 buf[name] → KeyError。
    """
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)
    for b in blocks:
        b.requires_grad_(False)

    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)
    # 构造之后再挂参数
    for b in blocks:
        b.late = nn.Parameter(torch.ones(16, device=device))
    swap.attach()

    x = torch.randn(1, 16, device=device)
    h = x
    for b in blocks:
        h = b(h)
    assert torch.isfinite(h).all()


def test_detach_removes_hooks():
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)
    swap.attach()
    swap.attach()  # 幂等
    n_handles = len(swap._handles)
    assert n_handles == 2 * 2  # 每个换出 block 2 个钩子
    swap.detach()
    assert swap._handles == []


def test_adopts_cpu_weights_without_recopy():
    """loader 已把尾部层放到 CPU pinned 时，组件就地接管不重复拷贝。"""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)
    # 模拟 loader：把末尾 2 层放到 CPU pinned
    for b in list(blocks)[2:]:
        for p in b.parameters():
            p.data = p.detach().to("cpu").pin_memory()
    ptrs = {id(p): p.data_ptr() for b in list(blocks)[2:] for p in b.parameters()}

    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)

    # 主副本应就是原来那批 pinned 张量（未重新分配）
    for rel in range(swap.num_swap):
        for _name, t in swap._cpu_weights[rel].items():
            assert t.is_pinned()
    for b in list(blocks)[2:]:
        for p in b.parameters():
            assert p.data_ptr() == ptrs[id(p)]


def test_fp8_base_with_swap_forward_matches_resident():
    """fp8 底模 + block swap（B7 的核心组合）：走真的 patch_fp8_linears。

    钉死两件事：
    - fp8 权重能 pin / H2D 搬运（dtype 原样，不 cast）
    - weight_scale 必须常驻计算设备。它跟随 module.weight.device 的话，换出层
      patch 时权重在 CPU → scale 落 CPU → 前向时权重已上 GPU，device 不匹配。
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "runtime"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from training.families.krea2.quant_fp8 import patch_fp8_linears

    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    dim, n = 32, 6

    class _Fp8Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return x + self.lin(x)

    torch.manual_seed(3)
    blocks = nn.ModuleList([_Fp8Block() for _ in range(n)]).to(device)
    blocks.requires_grad_(False)
    # 转 fp8 存储 + per-layer scale（模拟 fp8_scaled checkpoint）
    scales = {}
    for i, b in enumerate(blocks):
        b.lin.weight.data = b.lin.weight.data.to(torch.float8_e4m3fn)
        scales[f"{i}.lin"] = torch.tensor(0.5, device=device)
    patch_fp8_linears(blocks, scales, device=device)

    x = torch.randn(2, dim, device=device, dtype=torch.bfloat16)
    expected = _run_resident(blocks, x)

    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)
    swap.attach()
    got = _run_resident(blocks, x)

    torch.testing.assert_close(got, expected)
    # scale 全程在 GPU（不随权重下 CPU）
    for b in blocks:
        assert b.lin.weight_scale.device.type == "cuda"
    # 换出层的 fp8 权重主副本确实是 fp8 且 pinned
    for rel in range(swap.num_swap):
        for _name, t in swap._cpu_weights[rel].items():
            assert t.dtype == torch.float8_e4m3fn
            assert t.is_pinned()


def test_restore_masters_points_params_back_to_cpu():
    """restore_masters 把参数指回 CPU 主副本 —— fp8 merge 前必须先做。"""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    swap.attach()

    # 跑一次前向：换出层的 .data 此刻指向 GPU 槽
    _run_resident(blocks, torch.randn(1, 16, device=device))
    assert blocks[swap.first_swapped].lin1.weight.device.type == "cuda"

    swap.restore_masters()
    for b in list(blocks)[swap.first_swapped:]:
        assert b.lin1.weight.device.type == "cpu"
        assert b.lin1.weight.is_pinned()


def test_write_after_restore_masters_takes_effect_in_forward():
    """**核心保证**：restore 后写进权重的改动（= fp8 merge 的 delta）会被后续
    换入带上卡、真实影响前向输出，不会被 GPU 槽轮转吞掉。

    注意断言的是**前向输出**而不是换出层的 `.data` —— 一次 pass 结束后，某个
    换出层的 `.data` 仍指向它当时用的槽，而那个槽早已被后面的层覆盖（双缓冲
    轮转）。换出层的权重只在它自己的 forward 窗口内有效；窗口外要读权重必须
    先 restore_masters()。
    """
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    swap.attach()

    x = torch.randn(1, 16, device=device)
    before = _run_resident(blocks, x).clone()

    # 模拟 fp8 merge：restore 后就地改主副本
    swap.restore_masters()
    blocks[swap.first_swapped].lin1.weight.data.add_(1.0)

    after = _run_resident(blocks, x)
    assert not torch.allclose(before, after), "merge 的改动没有生效（被槽轮转吞了）"

    # 主副本是那份持久的：restore 后应仍带着改动
    swap.restore_masters()
    w = blocks[swap.first_swapped].lin1.weight
    assert w.device.type == "cpu" and w.is_pinned()


def test_swapped_weight_outside_forward_window_is_stale():
    """钉死上面那条语义：pass 结束后换出层的 .data 是被覆盖过的槽，不可信。

    这不是 bug 而是双缓冲的必然结果 —— 记录下来防止后来者按 `.data` 读权重。
    """
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    swap.attach()
    _run_resident(blocks, torch.randn(1, 16, device=device))

    first, last = swap.first_swapped, swap.total - 1
    # rel=0 与 rel=2 共用槽 0（rel % 2）→ pass 后 rel=0 的 .data 里其实是 rel=2
    torch.testing.assert_close(
        blocks[first].lin1.weight.data, blocks[last].lin1.weight.data,
    )
    # restore 之后各归各位
    swap.restore_masters()
    assert not torch.allclose(
        blocks[first].lin1.weight.data, blocks[last].lin1.weight.data,
    )


def test_move_module_excluding_keeps_swapped_on_cpu():
    """一刀切 module.to(device) 会把主副本搬上卡、swap 白做；本 helper 必须跳过。"""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "runtime"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from training.block_swap import move_module_excluding

    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    model = nn.Sequential(blocks)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    swap.restore_masters()

    # 先把一个常驻层挪到 CPU，模拟 offload 后要搬回的场景
    blocks[0].lin1.weight.data = blocks[0].lin1.weight.data.cpu()

    move_module_excluding(model, device, swap)

    assert blocks[0].lin1.weight.device.type == "cuda", "常驻层应被搬回 GPU"
    for b in list(blocks)[swap.first_swapped:]:
        assert b.lin1.weight.device.type == "cpu", "换出层必须留在 CPU"
        assert b.lin1.weight.is_pinned()


def test_move_module_excluding_without_swap_is_plain_move():
    """swap=None 时退化为普通 .to()，零行为变化。"""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "runtime"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from training.block_swap import move_module_excluding

    device = torch.device("cuda")
    blocks = _make_blocks(3, 16, torch.device("cpu"))
    model = nn.Sequential(blocks)
    move_module_excluding(model, device, None)
    for b in blocks:
        assert b.lin1.weight.device.type == "cuda"


def test_allocation_error_carries_context():
    """BlockSwapAllocationError 携带 num_swap/first_swapped/detail（供上层文案）。"""
    _, BlockSwapAllocationError = _import()
    err = BlockSwapAllocationError(num_swap=14, first_swapped=14, detail="out of memory")
    assert err.num_swap == 14
    assert err.first_swapped == 14
    assert "out of memory" in str(err)
    assert "14" in str(err)
