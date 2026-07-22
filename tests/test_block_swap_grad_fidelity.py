"""block swap 的梯度保真度（真实尺寸 + 噪声底校准）。

**为什么单独一个文件、且必须用真实尺寸**：`test_block_swap.py` 里那条小张量的
checkpoint 反向测试，在权重换入换出存在竞态时**照样全绿** —— 小 block 没有
attention、计算快到竞态窗口不显形。真机 6144-dim × 28 层才暴露：梯度偏差达噪声
底的 300 倍，PPSF 的 `d` 估计直接炸掉（用户真机报告）。

**为什么不能用 assert_close 逐位比**：SDPA 反向在 CUDA/bf16 上非确定 —— 同一份
权重跑两遍梯度就差约 5e-3。所以判据是「与**对照组自身的重复性**同量级」：
先测两次无 swap 的差（噪声底），再要求 swap 的差不超过它的若干倍。

历史读数（RTX 5090，8 层 × features 6144）：
    修复前 checkpoint  噪声底 4.4e-3  swap 差 1.31    → 298× ❌
    修复后 checkpoint  噪声底 4.4e-3  swap 差 4.7e-3  →   1× ✅
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="block swap 是 CUDA-only 机制"
)

_LAYERS = 6
_SWAP = 4
_SEQ = 512 + 128
#: swap 的梯度偏差相对噪声底的容忍倍数。真实 bug 是 300× 量级，噪声本身在
#: 1–2× 抖动，10× 给足余量又能牢牢抓住回归。
_TOLERANCE = 10.0


def _make_model(device, dtype):
    """真实尺寸的 krea2 block（含 attention —— 竞态要靠它才显形）+ 模拟 LoRA。"""
    from modeling.krea2 import KREA2_CONFIG
    from modeling.krea2.krea2_modeling import SingleStreamBlock

    cfg = KREA2_CONFIG
    torch.manual_seed(0)
    blocks = nn.ModuleList([
        SingleStreamBlock(
            cfg.features, cfg.heads, cfg.multiplier, cfg.bias, cfg.kvheads,
        )
        for _ in range(_LAYERS)
    ]).to(device, dtype)
    blocks.requires_grad_(False)          # 底模 frozen
    torch.manual_seed(1)
    for b in blocks:                      # LoRA 式可训练参数（常驻，不参与 swap）
        b.lora = nn.Parameter(torch.ones(cfg.features, device=device, dtype=dtype) * 0.01)
    return blocks, cfg


def _inputs(cfg, device, dtype):
    from modeling.krea2.krea2_modeling import PositionalEncoding

    head = cfg.features // cfg.heads
    axes = (head - 12 * (head // 16), 6 * (head // 16), 6 * (head // 16))
    freqs = PositionalEncoding(axes, theta=cfg.theta)(
        torch.zeros(1, _SEQ, 3, device=device)
    )
    torch.manual_seed(9)
    x = torch.randn(1, _SEQ, cfg.features, device=device, dtype=dtype)
    vec = torch.randn(1, 6 * cfg.features, device=device, dtype=dtype)
    return x, vec, freqs


def _grads(blocks, inputs, *, use_checkpoint: bool):
    x, vec, freqs = inputs
    h = x
    for b in blocks:
        def fwd(t, blk=b):
            return blk(t, vec, freqs) * blk.lora
        h = checkpoint(fwd, h, use_reentrant=False) if use_checkpoint else fwd(h)
    h.sum().backward()
    return [b.lora.grad.clone() for b in blocks]


def _max_rel(a_list, b_list) -> float:
    return max(
        ((a.float() - b.float()).abs().max()
         / max(b.float().abs().max().item(), 1e-9)).item()
        for a, b in zip(a_list, b_list)
    )


@pytest.mark.parametrize("use_checkpoint", [True, False])
def test_swap_gradients_stay_within_nondeterminism_noise(use_checkpoint):
    """swap 的梯度偏差必须与「无 swap 跑两遍」同量级。

    回归的是：前向结束就放开槽位、而反向仍要读那批权重 → 下一次换入把正在被
    读的权重覆盖掉 → 梯度静默错乱（不报错、不 NaN，只是数值不对）。
    """
    from training.block_swap import PinnedBlockSwap

    device = torch.device("cuda")
    dtype = torch.bfloat16

    blocks_a, cfg = _make_model(device, dtype)
    inputs = _inputs(cfg, device, dtype)
    grads_a = _grads(blocks_a, inputs, use_checkpoint=use_checkpoint)

    blocks_b, _ = _make_model(device, dtype)
    grads_b = _grads(blocks_b, inputs, use_checkpoint=use_checkpoint)
    noise = _max_rel(grads_a, grads_b)      # 同一份权重跑两遍的固有抖动

    blocks_swap, _ = _make_model(device, dtype)
    swap = PinnedBlockSwap(blocks_swap, num_swap=_SWAP, device=device)
    swap.attach()
    grads_swap = _grads(blocks_swap, inputs, use_checkpoint=use_checkpoint)
    observed = _max_rel(grads_swap, grads_a)

    assert observed <= max(noise, 1e-4) * _TOLERANCE, (
        f"block swap 的梯度偏差 {observed:.2e} 超过噪声底 {noise:.2e} 的 "
        f"{_TOLERANCE}× —— 权重在反向期间被换入覆盖了"
    )
