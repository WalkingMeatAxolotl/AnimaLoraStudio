"""krea2 生成显存编排（docs/todo/krea2-vram-orchestration.md D1）。

按需让位 = comfy free_memory 语义：装得下同驻（零搬运），装不下才让位。
vram_policy=None 维持旧行为——训练预览调用面绝不动 model。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import training.families.krea2 as krea2_module
from training.families.krea2 import (
    Krea2Family,
    _sampling_headroom_bytes,
    _should_offload_te,
    _should_yield_dit,
)

_GIB = 1024 ** 3


def test_sampling_headroom_scales_with_area():
    assert _sampling_headroom_bytes(1024, 1024) == int(5.0 * _GIB)
    assert _sampling_headroom_bytes(1536, 1536) == int((2.0 + 3.0 * 2.25) * _GIB)


def test_should_yield_dit_policies(monkeypatch):
    # 固定档不查询显存
    assert _should_yield_dit("performance", "cuda") is False
    assert _should_yield_dit("save_vram", "cuda") is True
    # auto：free 充裕不让位，紧张让位，查询不到（无 CUDA）保守不让
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: 17 * _GIB)
    assert _should_yield_dit("auto", "cuda") is False
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: 2 * _GIB)
    assert _should_yield_dit("auto", "cuda") is True
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: None)
    assert _should_yield_dit("auto", "cuda") is False


def test_should_offload_te_policies(monkeypatch):
    assert _should_offload_te("performance", "cuda", 1024, 1024, False) is False
    assert _should_offload_te("save_vram", "cuda", 1024, 1024, False) is True
    # DiT 刚让过位 → 必卸（free 虚高不可判）
    assert _should_offload_te("auto", "cuda", 1024, 1024, True) is True
    # auto：余量充足不卸（32GB fp8 同驻零搬运），不足卸，查不到保守卸
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: 9 * _GIB)
    assert _should_offload_te("auto", "cuda", 1024, 1024, False) is False
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: 3 * _GIB)
    assert _should_offload_te("auto", "cuda", 1024, 1024, False) is True
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: None)
    assert _should_offload_te("auto", "cuda", 1024, 1024, False) is True


class _OrchestraModel:
    def __init__(self):
        self.moves: list[str] = []

    def to(self, device):
        self.moves.append(str(device))
        return self


class _OrchestraText:
    """duck-type Krea2TextStack：记录 offload 调用与驻留状态。"""

    def __init__(self, *, cached: bool = False, resident: bool = False):
        self._cached = cached
        self.is_model_on_device = resident
        self.offload_calls = 0

    def online_conditions_cached(self, captions):
        return self._cached

    def offload_model(self):
        self.offload_calls += 1


def _run_sample(monkeypatch, *, policy, text, free_gib=None):
    family = Krea2Family()
    model = _OrchestraModel()
    calls = {}

    monkeypatch.setattr(
        krea2_module, "_cuda_free_bytes",
        lambda d: None if free_gib is None else int(free_gib * _GIB),
    )
    monkeypatch.setattr(
        krea2_module, "prepare_sampling_condition",
        lambda *a, **k: calls.setdefault("prepared", True) or "cond",
    )
    monkeypatch.setattr(
        krea2_module, "sample_image",
        lambda *a, **k: "image",
    )
    result = family.sample_image(
        model, object(), text, "a prompt",
        distilled=True, device="cpu", vram_policy=policy,
    )
    assert result == "image"
    assert calls.get("prepared")
    return model, text


def test_policy_none_keeps_legacy_behavior(monkeypatch):
    """None（训练预览等旧调用面）：model 绝不 .to()，TE 无条件 offload。"""
    model, text = _run_sample(monkeypatch, policy=None, text=_OrchestraText())
    assert model.moves == []
    assert text.offload_calls == 1


def test_auto_with_ample_vram_keeps_all_resident(monkeypatch):
    """auto + 显存充裕（32GB fp8 场景）：不让位、不卸 TE——零搬运。"""
    model, text = _run_sample(
        monkeypatch, policy="auto", text=_OrchestraText(), free_gib=15,
    )
    assert model.moves == []
    assert text.offload_calls == 0


def test_auto_tight_vram_yields_dit_and_offloads_te(monkeypatch):
    """auto + 显存紧张（16GB 场景）：DiT 让位 → 编码 → 卸 TE → DiT 回。"""
    model, text = _run_sample(
        monkeypatch, policy="auto", text=_OrchestraText(), free_gib=2,
    )
    # 让位到 cpu + 采样前搬回目标设备（测试 device="cpu"）
    assert model.moves == ["cpu", "cpu"]
    assert text.offload_calls == 1


def test_auto_lru_hit_skips_yield_entirely(monkeypatch):
    """prompt 全命中在线 LRU：编码期 TE 不上卡，让位判断整体跳过。"""
    model, text = _run_sample(
        monkeypatch, policy="auto",
        text=_OrchestraText(cached=True), free_gib=2,
    )
    assert model.moves == []


def test_auto_te_already_resident_skips_yield(monkeypatch):
    """TE 已驻 GPU（上一轮 auto 未卸）：free 已扣除 TE，不得误判让位。"""
    model, text = _run_sample(
        monkeypatch, policy="auto",
        text=_OrchestraText(resident=True), free_gib=9,
    )
    assert model.moves == []
    assert text.offload_calls == 0  # 9GB > 1024² 余量 5GB，继续同驻


# —— block swap 下的逐格让位：只搬常驻部分，换出层绝不跟着 .to() ——————————


def _swap_model():
    """四层 ModuleList 上登记一个 swap 句柄（替身即可：这里只验句柄的找回与透传）。"""
    from torch import nn

    from training.block_swap import _OWNER_ATTR

    blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
    swap = object()
    setattr(blocks, _OWNER_ATTR, swap)
    return nn.Sequential(blocks), swap


def test_find_block_swap_resolves_registered_handle():
    from torch import nn

    from training.block_swap import find_block_swap

    model, swap = _swap_model()
    assert find_block_swap(model) is swap
    # 无 swap 的普通模型 / 测试替身（无 modules()）→ None，走原 model.to()
    assert find_block_swap(nn.Sequential(nn.Linear(2, 2))) is None
    assert find_block_swap(_OrchestraModel()) is None


def test_yield_threads_swap_handle_through_both_moves(monkeypatch):
    """让位与搬回都必须带着 swap 句柄走 move_module_excluding——
    任一处退回裸 model.to() 都会把换出层主副本整份传上卡。

    （move_module_excluding 跳过被管理张量的真实行为在 CUDA 测试
    tests/test_block_swap_krea2_yield.py 里验。）
    """
    model, swap = _swap_model()
    moves = []
    monkeypatch.setattr(
        krea2_module, "move_module_excluding",
        lambda m, device, s: moves.append((m, str(device), s)),
    )
    monkeypatch.setattr(
        krea2_module, "prepare_sampling_condition", lambda *a, **k: "cond",
    )
    monkeypatch.setattr(krea2_module, "sample_image", lambda *a, **k: "image")
    result = Krea2Family().sample_image(
        model, object(), _OrchestraText(), "a prompt",
        distilled=True, device="cpu", vram_policy="save_vram",
    )
    assert result == "image"
    assert moves == [(model, "cpu", swap), (model, "cpu", swap)]


def test_yield_without_swap_passes_none(monkeypatch):
    """无 block swap：swap=None → move_module_excluding 退化为 model.to()，
    与修复前逐字节一致（_OrchestraModel 的 .to 记录见上方既有用例）。"""
    from torch import nn

    model = nn.Sequential(nn.Linear(4, 4))
    moves = []
    monkeypatch.setattr(
        krea2_module, "move_module_excluding",
        lambda m, device, s: moves.append(s),
    )
    monkeypatch.setattr(
        krea2_module, "prepare_sampling_condition", lambda *a, **k: "cond",
    )
    monkeypatch.setattr(krea2_module, "sample_image", lambda *a, **k: "image")
    Krea2Family().sample_image(
        model, object(), _OrchestraText(), "a prompt",
        distilled=True, device="cpu", vram_policy="save_vram",
    )
    assert moves == [None, None]
