"""fp8 底模 LoRA merge × block swap 的交互（docs/design/block-swap.md §9.6）。

推理侧 fp8 的 LoRA 走**权重 merge**（ComfyUI 语义），与 block swap 有两处会互相
踩脚，都是真机跑出来的：

1. `weight_scale` 恒在计算设备（swap 前向需要），而换出层的权重是 CPU pinned
   主副本 → `w16 * scale` 跨设备崩。
2. merge 必须落在**主副本**上：跑过前向后权重指向的是会被下一层换入覆盖的
   GPU 槽位，写进去会静默丢失（出图看着正常但 LoRA 没生效）。

判据是与**无 swap 对照组逐位一致** —— 只测「不崩」会漏掉第 2 类静默错误。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="block swap 是 CUDA-only 机制"
)

_DIM = 64
_LAYERS = 6
_SWAP = 4


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(_DIM, _DIM, bias=False)

    def forward(self, x):
        return x + self.lin(x)


def _build_fp8_blocks(device):
    """fp8_scaled 形态：权重 fp8 + per-layer scale，走真的 patch_fp8_linears。"""
    from training.families.krea2.quant_fp8 import patch_fp8_linears

    torch.manual_seed(0)
    blocks = nn.ModuleList([_Block() for _ in range(_LAYERS)]).to(device)
    blocks.requires_grad_(False)
    scales = {}
    for i, b in enumerate(blocks):
        b.lin.weight.data = b.lin.weight.data.to(torch.float8_e4m3fn)
        scales[f"{i}.lin"] = torch.tensor(0.5, device=device)
    patch_fp8_linears(blocks, scales, device=device)
    return blocks


def _lora_state_dict():
    torch.manual_seed(7)
    sd = {}
    for i in range(_LAYERS):
        key = f"lora_unet_{i}_lin"
        sd[f"{key}.lora_down.weight"] = torch.randn(4, _DIM) * 0.05
        sd[f"{key}.lora_up.weight"] = torch.randn(_DIM, 4) * 0.05
    return sd


def _forward(blocks, x):
    h = x
    with torch.no_grad():
        for b in blocks:
            h = b(h)
    return h


def test_fp8_merge_with_swap_matches_no_swap_reference():
    """用户真机路径：swap + 跑过前向 + restore_masters + merge。

    结果必须与「无 swap 直接 merge」逐位一致，且跨多次出图稳定。
    """
    from training.block_swap import PinnedBlockSwap
    from training.families.krea2.lora_fp8_merge import merge_loras_into_fp8_model

    device = torch.device("cuda")
    x = torch.randn(2, _DIM, device=device, dtype=torch.bfloat16)
    lora = _lora_state_dict()

    # 对照组：无 swap，直接 merge
    ref = _build_fp8_blocks(device)
    merge_loras_into_fp8_model(ref, [(lora, 1.0, "t.safetensors")])
    expected = _forward(ref, x)

    # 实验组：swap → 先跑一轮（权重指向 GPU 槽）→ restore → merge
    blocks = _build_fp8_blocks(device)
    swap = PinnedBlockSwap(blocks, num_swap=_SWAP, device=device)
    swap.attach()
    _forward(blocks, x)
    swap.restore_masters()
    merge_loras_into_fp8_model(blocks, [(lora, 1.0, "t.safetensors")])

    torch.testing.assert_close(_forward(blocks, x), expected)
    # 第二次出图仍一致 —— 证明 merge 落在持久主副本而非被轮转覆盖的槽位
    torch.testing.assert_close(_forward(blocks, x), expected)


def test_fp8_merge_writes_reach_the_pinned_masters():
    """merge 的改动必须真的写进 CPU pinned 主副本（而不是某个 GPU 槽）。"""
    from training.block_swap import PinnedBlockSwap
    from training.families.krea2.lora_fp8_merge import merge_loras_into_fp8_model

    device = torch.device("cuda")
    blocks = _build_fp8_blocks(device)
    swap = PinnedBlockSwap(blocks, num_swap=_SWAP, device=device)
    swap.attach()
    _forward(blocks, torch.randn(2, _DIM, device=device, dtype=torch.bfloat16))

    swap.restore_masters()
    before = {
        rel: swap._cpu_weights[rel]["lin.weight"].clone()
        for rel in range(swap.num_swap)
    }
    merge_loras_into_fp8_model(blocks, [(_lora_state_dict(), 1.0, "t.safetensors")])

    for rel in range(swap.num_swap):
        master = swap._cpu_weights[rel]["lin.weight"]
        assert master.is_pinned(), "主副本应仍是 pinned"
        assert not torch.equal(
            master.view(torch.uint8), before[rel].view(torch.uint8)
        ), f"第 {rel} 个换出层的主副本没有被 merge 写到"
