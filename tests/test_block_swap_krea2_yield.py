"""block swap × krea2 逐格「DiT 让位给 TE」（真 CUDA）。

Krea2Family.sample_image 在 prompt 未命中在线 LRU、显存装不下 TE 时把 DiT 先撤到
CPU、编码完再搬回。block swap 下若这两步是一刀切 ``model.to()``，搬回时会把换出层
的 CPU pinned 主副本整份传上卡——swap 白做、峰值回到完整模型（12/16GB 卡 OOM）。

本文件钉死：让位前后换出层主副本仍是同一份 CPU pinned 存储、搬回峰值不含换出层，
且让位后前向与全常驻逐位一致。无 CUDA 时整文件 skip。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="block swap 是 CUDA-only 机制"
)

_DIM = 1024          # 每个 Linear 4 MiB（fp32），block 8 MiB：峰值差远大于噪声
_N_BLOCKS = 6
_N_SWAP = 4


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(torch.relu(self.lin1(x)))


class _DiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.blocks = nn.ModuleList([_Block(_DIM) for _ in range(_N_BLOCKS)])
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _Text:
    """duck-type Krea2TextStack：未命中 LRU、TE 不在卡上 → 触发让位判断。"""

    is_model_on_device = False

    def online_conditions_cached(self, captions):
        return False

    def offload_model(self):
        pass


def _block_bytes(block: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in block.parameters())


def _build():
    from training.block_swap import PinnedBlockSwap

    device = torch.device("cuda")
    model = _DiT().to(device)
    x = torch.randn(4, _DIM, device=device)
    with torch.no_grad():
        reference = model(x).clone()
    swap = PinnedBlockSwap(model.blocks, _N_SWAP, device)
    swap.attach()
    with torch.no_grad():
        model(x)  # 跑一遍让换出层 .data 绑到 GPU 槽（真实出图后的状态）
    return model, swap, x, reference


def _swapped_params(model, swap):
    return [
        (name, p)
        for block in list(model.blocks)[swap.first_swapped:]
        for name, p in block.named_parameters()
    ]


def test_find_block_swap_registered_and_cleared_on_close():
    from training.block_swap import find_block_swap

    model, swap, _x, _ref = _build()
    assert find_block_swap(model) is swap
    swap.close()
    assert find_block_swap(model) is None


def test_krea2_yield_keeps_swapped_layers_pinned_on_cpu(monkeypatch):
    import training.families.krea2 as krea2_module
    from training.families.krea2 import Krea2Family

    model, swap, x, reference = _build()
    device = torch.device("cuda")
    master_ptrs = [{n: t.data_ptr() for n, t in w.items()}
                   for w in swap._cpu_weights]
    resident_bytes = sum(
        _block_bytes(b) for b in list(model.blocks)[:swap.first_swapped]
    )
    swapped_bytes = sum(
        _block_bytes(b) for b in list(model.blocks)[swap.first_swapped:]
    )
    seen = {}

    def _fake_sample(m, *a, **k):
        # 搬回之后、采样之前：换出层张量全部仍归 swap 管（主副本或 GPU 槽）
        managed = swap.managed_data_ptrs()
        seen["swapped_managed"] = all(
            p.data.data_ptr() in managed for _n, p in _swapped_params(m, swap)
        )
        seen["resident_on_gpu"] = all(
            p.device.type == "cuda"
            for b in list(m.blocks)[:swap.first_swapped] for p in b.parameters()
        )
        seen["peak"] = torch.cuda.max_memory_allocated(device)
        with torch.no_grad():
            seen["out"] = m(x).clone()
        return "image"

    monkeypatch.setattr(
        krea2_module, "prepare_sampling_condition", lambda *a, **k: "cond",
    )
    monkeypatch.setattr(krea2_module, "sample_image", _fake_sample)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = Krea2Family().sample_image(
        model, object(), _Text(), "a prompt",
        distilled=True, device="cuda", vram_policy="save_vram",
    )
    assert result == "image"

    # 主副本：同一份存储、仍在 CPU 且 pinned
    for rel, weights in enumerate(swap._cpu_weights):
        for name, tensor in weights.items():
            assert tensor.device.type == "cpu" and tensor.is_pinned()
            assert tensor.data_ptr() == master_ptrs[rel][name]
    assert seen["swapped_managed"], "换出层被 .to() 搬出了 swap 的管理范围"
    assert seen["resident_on_gpu"], "常驻层应已搬回 GPU"
    # 搬回峰值只含常驻层（base 里已有 GPU 槽 + 输入）；退回一刀切 .to() 时
    # 会额外多出整份换出层（_N_SWAP 个 block）
    assert seen["peak"] - base < resident_bytes + swapped_bytes // 2, (
        f"峰值 {(seen['peak'] - base) / 2**20:.1f} MiB 含换出层"
        f"（常驻 {resident_bytes / 2**20:.0f} MiB / 换出 {swapped_bytes / 2**20:.0f} MiB）"
    )
    # 让位往返后前向仍与全常驻逐位一致
    assert torch.equal(seen["out"], reference)
    swap.close()
