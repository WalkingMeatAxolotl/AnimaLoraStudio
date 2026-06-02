"""lora algo 默认走 bypass_mode 的等价性 + DoRA / LoKr / LoHa 仍走 rebuild 的 guard。

背景：lycoris LoConModule 默认 forward 会 rebuild ΔW=up@down 全矩阵再多跑一次
F.linear——对普通 LoRA 是每层 ~2× FLOPs 的浪费（issue #182）。bypass_mode=True 路径
是经典 `org_forward(x) + lora_up(lora_down(x)) * scale`，等价于 sd-scripts/PEFT 的 LoRA forward。

本文件验证：
1) lora algo (LoCon) 下，bypass=True 与 bypass=False 的 forward / backward 数值等价
2) AnimaLycorisAdapter(algo='lora') 默认 bypass_mode=True
3) DoRA(weight_decompose=True) 强制 bypass_mode=False，避免 lycoris bypass 路径
   不走 wd 分支导致的静默失效
4) LoKr / LoHa 保持 bypass_mode=False（默认 rebuild，行为不变）
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from utils.lycoris_adapter import AnimaLycorisAdapter

pytest.importorskip("lycoris")


class MockDiT(nn.Module):
    """对齐 ANIMA_PRESET 的 target_name（*q_proj/*k_proj/*v_proj/*output_proj/*mlp.layer1/2）"""

    def __init__(self, d: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.output_proj = nn.Linear(d, d, bias=False)


def _bypass_modes(adapter: AnimaLycorisAdapter) -> list[bool]:
    return [bool(getattr(m, "bypass_mode", False)) for m in adapter.network.loras]


# ─── (1) LoCon 数值等价：bypass=True vs bypass=False ──────────────────────────


def _build_lora_module(bypass: bool, seed: int = 0):
    """直接造一个 LoConModule，避开 LycorisNetwork 的 preset/target 匹配"""
    from lycoris.modules.locon import LoConModule

    torch.manual_seed(seed)
    linear = nn.Linear(64, 64, bias=False)
    mod = LoConModule(
        lora_name="test",
        org_module=linear,
        multiplier=1.0,
        lora_dim=8,
        alpha=8,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        bypass_mode=bypass,
    )
    mod.apply_to()
    return linear, mod


def _copy_lora_weights(src, dst) -> None:
    """把 src 的 lora_up/down 权重塞进 dst（同 shape）"""
    dst.lora_up.weight.data.copy_(src.lora_up.weight.data)
    dst.lora_down.weight.data.copy_(src.lora_down.weight.data)


def test_locon_bypass_vs_rebuild_forward_equivalent() -> None:
    """同样权重、同样输入：bypass 路径与 rebuild 路径 forward 输出数值一致。

    LoRA paper 的 W'X = WX + BAX，bypass 是直接算右式；rebuild 是先 W'=W+BA 再算 W'X。
    fp32 下应严格一致到 ~1e-5 量级。
    """
    linear_a, mod_bypass = _build_lora_module(bypass=True, seed=0)
    linear_b, mod_rebuild = _build_lora_module(bypass=False, seed=0)
    # base linear 权重已经因为 seed=0 一致；同步 lora 部分
    _copy_lora_weights(mod_bypass, mod_rebuild)
    # 让 lora_up 不为 0（默认 init 是 0）才能真正测到 lora 路径
    with torch.no_grad():
        mod_bypass.lora_up.weight.normal_(std=0.1)
        mod_rebuild.lora_up.weight.copy_(mod_bypass.lora_up.weight)

    # 同步 base linear 权重以防 seed 之外有差异
    with torch.no_grad():
        linear_b.weight.copy_(linear_a.weight)

    mod_bypass.eval()
    mod_rebuild.eval()
    x = torch.randn(2, 16, 64)
    out_bypass = linear_a(x)
    out_rebuild = linear_b(x)
    assert torch.allclose(out_bypass, out_rebuild, atol=1e-5, rtol=1e-5)


def test_locon_bypass_vs_rebuild_backward_equivalent() -> None:
    """同样 loss：两条路径在 lora_up/lora_down 上的梯度数值一致。"""
    linear_a, mod_bypass = _build_lora_module(bypass=True, seed=1)
    linear_b, mod_rebuild = _build_lora_module(bypass=False, seed=1)
    _copy_lora_weights(mod_bypass, mod_rebuild)
    with torch.no_grad():
        mod_bypass.lora_up.weight.normal_(std=0.1)
        mod_rebuild.lora_up.weight.copy_(mod_bypass.lora_up.weight)
        linear_b.weight.copy_(linear_a.weight)

    mod_bypass.train()
    mod_rebuild.train()
    x = torch.randn(2, 16, 64, requires_grad=False)
    target = torch.randn(2, 16, 64)

    loss_bypass = (linear_a(x) - target).pow(2).mean()
    loss_rebuild = (linear_b(x) - target).pow(2).mean()
    assert torch.allclose(loss_bypass, loss_rebuild, atol=1e-5)

    loss_bypass.backward()
    loss_rebuild.backward()

    assert torch.allclose(
        mod_bypass.lora_up.weight.grad,
        mod_rebuild.lora_up.weight.grad,
        atol=1e-5, rtol=1e-5,
    )
    assert torch.allclose(
        mod_bypass.lora_down.weight.grad,
        mod_rebuild.lora_down.weight.grad,
        atol=1e-5, rtol=1e-5,
    )


# ─── (2-4) AnimaLycorisAdapter 按 algo + DoRA 自动选 bypass_mode ──────────────


def test_adapter_lora_defaults_to_bypass_mode() -> None:
    """algo='lora' 不开 DoRA → 全部模块 bypass_mode=True（issue #182 默认快路径）"""
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(algo="lora", rank=8, alpha=8)
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes, "preset 应该至少匹配一个 q/k/v/output_proj"
    assert all(modes), f"lora algo 全部模块应走 bypass，但得到 {modes}"


def test_adapter_lora_with_dora_forces_rebuild() -> None:
    """algo='lora' + lora_dora=True：DoRA 数学上必须 rebuild，guard 不能让 bypass 静默吞掉 wd 分支"""
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(
        algo="lora", rank=8, alpha=8, weight_decompose=True,
    )
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert not any(modes), f"DoRA 必须走 rebuild，但 bypass_mode={modes}"


def test_adapter_lokr_keeps_rebuild() -> None:
    """algo='lokr' 行为不变（LoKr 的 bypass 数值等价但路径相当绕，本 PR 不动）"""
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(algo="lokr", rank=8, alpha=8, factor=8)
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert not any(modes), f"lokr 应保持 rebuild，但 bypass_mode={modes}"


def test_adapter_loha_keeps_rebuild() -> None:
    """algo='loha' 行为不变（LoHa bypass 内部仍 rebuild，开了反而慢；lycoris changelog 原话）"""
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(algo="loha", rank=8, alpha=8)
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert not any(modes), f"loha 应保持 rebuild，但 bypass_mode={modes}"
