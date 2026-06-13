"""save() 时 per-layer .alpha 重写覆盖测试。

回归保护：lycoris LokrModule init 写入 .alpha=scale×lora_dim；_apply_reg_dims_
改 lora_dim 后没动 self.scale / self.alpha buffer → 分层 rank 层的 .alpha 与
per-layer rank 失配 → ComfyUI 按 alpha/rank 算 scale 偏离训练几十倍 → 噪点。
save() 调 _rewrite_per_layer_alpha_ 在落盘前按当前真实 (scale, lora_dim) 重算。
"""
from __future__ import annotations

import pytest
import torch

from utils.lycoris_adapter import _rewrite_per_layer_alpha_


class _FakeLora:
    def __init__(self, name: str, scale: float, dim: int) -> None:
        self.lora_name = name
        self.scale = scale
        self.lora_dim = dim


class _FakeNet:
    def __init__(self, loras: list) -> None:
        self.loras = loras


def test_rewrite_alpha_uses_scale_times_dim() -> None:
    """每层 .alpha = scale × lora_dim，让下游 alpha/rank 还原训练 scale。"""
    net = _FakeNet([
        _FakeLora("layer_a", scale=5.6569, dim=1),    # 分层 rank=1：5.6569
        _FakeLora("layer_b", scale=5.6569, dim=8),    # 分层 rank=8：45.255
        _FakeLora("layer_c", scale=5.6569, dim=32),   # base rank：181.0
        _FakeLora("layer_d", scale=5.6569, dim=64),   # 分层 rank=64：362.0
    ])
    sd = {
        "layer_a.alpha": torch.tensor(181.0, dtype=torch.float32),  # lycoris 写入的旧值
        "layer_b.alpha": torch.tensor(181.0, dtype=torch.float32),
        "layer_c.alpha": torch.tensor(181.0, dtype=torch.float32),
        "layer_d.alpha": torch.tensor(181.0, dtype=torch.float32),
        "layer_a.lokr_w2_a": torch.zeros(2, 1),  # 非 alpha 张量不动
    }
    _rewrite_per_layer_alpha_(net, sd)
    assert sd["layer_a.alpha"].item() == pytest.approx(5.6569, abs=1e-4)
    assert sd["layer_b.alpha"].item() == pytest.approx(45.2552, abs=1e-4)
    assert sd["layer_c.alpha"].item() == pytest.approx(181.0208, abs=1e-3)
    assert sd["layer_d.alpha"].item() == pytest.approx(362.0416, abs=1e-3)
    assert sd["layer_a.lokr_w2_a"].shape == (2, 1)


def test_rewrite_alpha_noop_when_dim_equals_base() -> None:
    """未触发分层 rank 时 scale × dim 必等于 lycoris 原写入值 → no-op."""
    # lycoris init: alpha_buf = user_alpha × (dim / r_factor)；scale = user_alpha / r_factor
    # → alpha_buf == scale × dim 恒等
    net = _FakeNet([_FakeLora("x", scale=1.0, dim=32)])
    sd = {"x.alpha": torch.tensor(32.0)}
    _rewrite_per_layer_alpha_(net, sd)
    assert sd["x.alpha"].item() == pytest.approx(32.0)


def test_rewrite_alpha_handles_none_network() -> None:
    """network 未 inject 时 noop，不爆。"""
    sd = {"x.alpha": torch.tensor(99.0)}
    _rewrite_per_layer_alpha_(None, sd)
    assert sd["x.alpha"].item() == 99.0


def test_rewrite_alpha_skips_layer_without_alpha_key() -> None:
    """sd 没该层 .alpha 张量时跳过。"""
    net = _FakeNet([_FakeLora("missing", scale=1.0, dim=4)])
    sd = {"other.alpha": torch.tensor(7.0)}
    _rewrite_per_layer_alpha_(net, sd)
    assert sd["other.alpha"].item() == 7.0
    assert "missing.alpha" not in sd


def test_rewrite_alpha_preserves_dtype() -> None:
    """新 alpha 张量保留原 dtype（bf16/fp16 训练管线常见）。"""
    net = _FakeNet([_FakeLora("x", scale=5.6569, dim=1)])
    sd = {"x.alpha": torch.tensor(181.0, dtype=torch.bfloat16)}
    _rewrite_per_layer_alpha_(net, sd)
    assert sd["x.alpha"].dtype == torch.bfloat16


def test_rewrite_alpha_skips_lora_missing_attrs() -> None:
    """LoRA 对象缺 scale/lora_dim/lora_name 时跳过，不破坏 sd。"""
    class _NoName:
        scale = 1.0
        lora_dim = 4

    class _NoScale:
        lora_name = "x"
        lora_dim = 4

    net = _FakeNet([_NoName(), _NoScale()])
    sd = {"x.alpha": torch.tensor(7.0)}
    _rewrite_per_layer_alpha_(net, sd)
    assert sd["x.alpha"].item() == 7.0
