"""block swap × anima 采样期 VAE decode offload（真 CUDA）。

anima sample_image 在显存紧张但整图峰值仍在崖下时，把 DiT（+Qwen）挪到 CPU 腾
显存，decode 完再搬回。block swap 下若这两步是一刀切 ``module.to()``，搬回时会
把换出层的 CPU pinned 主副本整份传上卡——swap 白做、峰值回到完整模型。

本文件钉死：offload 只腾出常驻部分、换出层主副本始终是同一份 CPU pinned 存储、
搬回峰值不含换出层，且往返后前向与全常驻逐位一致。无 CUDA 时整文件 skip。
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


def _block_bytes(block: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in block.parameters())


def test_vae_decode_offload_keeps_swapped_layers_pinned_on_cpu():
    from training.block_swap import PinnedBlockSwap
    from training.families.anima import sampling

    device = torch.device("cuda")
    model = _DiT().to(device)
    x = torch.randn(4, _DIM, device=device)
    with torch.no_grad():
        reference = model(x).clone()
    swap = PinnedBlockSwap(model.blocks, _N_SWAP, device)
    swap.attach()
    with torch.no_grad():
        model(x)  # 跑一遍让换出层 .data 绑到 GPU 槽（采样结束、decode 前的真实状态）

    master_ptrs = [{n: t.data_ptr() for n, t in w.items()}
                   for w in swap._cpu_weights]
    resident = list(model.blocks)[:swap.first_swapped]
    swapped = list(model.blocks)[swap.first_swapped:]
    resident_bytes = sum(_block_bytes(b) for b in resident)
    swapped_bytes = sum(_block_bytes(b) for b in swapped)

    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    offloaded = sampling._offload_modules_for_vae_decode(model)
    torch.cuda.synchronize(device)
    after_offload = torch.cuda.memory_allocated(device)

    assert all(p.device.type == "cpu" for b in resident for p in b.parameters())
    # 腾出的就是常驻层（GPU 槽仍归 swap，不动）
    assert before - after_offload == resident_bytes

    torch.cuda.reset_peak_memory_stats(device)
    sampling._restore_offloaded_modules(offloaded)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)

    # 主副本：同一份存储、仍在 CPU 且 pinned
    for rel, weights in enumerate(swap._cpu_weights):
        for name, tensor in weights.items():
            assert tensor.device.type == "cpu" and tensor.is_pinned()
            assert tensor.data_ptr() == master_ptrs[rel][name]
    managed = swap.managed_data_ptrs()
    assert all(
        p.data.data_ptr() in managed for b in swapped for p in b.parameters()
    ), "换出层被 .to() 搬出了 swap 的管理范围"
    assert all(p.device.type == "cuda" for b in resident for p in b.parameters())
    # 搬回峰值只含常驻层；退回一刀切 .to() 时会额外多出整份换出层
    assert peak - after_offload < resident_bytes + swapped_bytes // 2, (
        f"峰值 {(peak - after_offload) / 2**20:.1f} MiB 含换出层"
        f"（常驻 {resident_bytes / 2**20:.0f} MiB / 换出 {swapped_bytes / 2**20:.0f} MiB）"
    )
    # offload 往返后前向仍与全常驻逐位一致
    with torch.no_grad():
        assert torch.equal(model(x), reference)
    swap.close()
