"""training/losses 单元测试（PR-D huber loss plugin）。

覆盖：
- MseLoss：跟 F.mse_loss(reduction='none') bit-for-bit 一致 + 不依赖 t + Protocol 合规
- HuberLoss：quad / linear / 边界 三段公式正确 + constant/snr/sigma 三种 schedule 的 delta 行为
- plugin registry：BUILDERS / build_loss / validate_schema_consistency 三件套
- 集成：所有 loss 在 (B, C, H, W) latent shape 上输出形状正确、有限值
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from training.losses import BUILDERS, build_loss, validate_schema_consistency
from training.losses.huber import HuberLoss
from training.losses.huber import build as build_huber
from training.losses.mse import MseLoss
from training.losses.mse import build as build_mse
from training.losses.protocol import LossProtocol


# ---------------------------------------------------------------------------
# MseLoss：跟历史 F.mse_loss bit-for-bit
# ---------------------------------------------------------------------------


def test_mse_matches_f_mse_loss_bitwise():
    """MseLoss.compute 必须等价于历史 F.mse_loss(reduction='none')。"""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 3, 4, 4)
    t = torch.rand(2)

    expected = F.mse_loss(pred, target, reduction="none")
    actual = MseLoss().compute(pred, target, t)
    torch.testing.assert_close(actual, expected)


def test_mse_ignores_t():
    """MSE 不依赖 t，传任何 t 输出相同。"""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 3, 4, 4)
    mse = MseLoss()
    out1 = mse.compute(pred, target, torch.tensor([0.1, 0.5]))
    out2 = mse.compute(pred, target, torch.tensor([0.9, 0.99]))
    torch.testing.assert_close(out1, out2)


def test_mse_build_returns_instance():
    class Args:
        pass  # mse 不需要任何字段
    loss = build_mse(Args())
    assert isinstance(loss, MseLoss)


# ---------------------------------------------------------------------------
# HuberLoss constant schedule：数学公式
# ---------------------------------------------------------------------------


def test_huber_constant_inside_quadratic_region():
    """|x| < delta 时 Huber = 0.5 * x^2。"""
    pred = torch.tensor([[0.05]])
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    huber = HuberLoss(c=0.15, schedule="constant")
    expected = torch.tensor([[0.5 * 0.05 * 0.05]])
    actual = huber.compute(pred, target, t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_huber_constant_outside_linear_region():
    """|x| > delta 时 Huber = delta * (|x| - 0.5*delta)。"""
    pred = torch.tensor([[1.0]])
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    huber = HuberLoss(c=0.15, schedule="constant")
    expected = torch.tensor([[0.15 * (1.0 - 0.5 * 0.15)]])  # 0.13875
    actual = huber.compute(pred, target, t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_huber_continuous_at_boundary():
    """|x| = delta 时 quad 和 lin 都应等于 0.5 * delta^2。"""
    delta = 0.15
    pred = torch.tensor([[delta]])
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    huber = HuberLoss(c=delta, schedule="constant")
    # torch.where(|x| < delta) 严格小于，|x|=delta 走 lin 分支
    expected = torch.tensor([[delta * (delta - 0.5 * delta)]])  # 0.5 * delta^2
    actual = huber.compute(pred, target, t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_huber_smaller_than_mse_for_large_diff():
    """大 diff 时 Huber 应明显小于 MSE（鲁棒性核心目的）。"""
    pred = torch.tensor([[5.0]])  # 比 delta=0.15 大很多
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    mse_val = MseLoss().compute(pred, target, t)
    huber_val = HuberLoss(c=0.15, schedule="constant").compute(pred, target, t)
    assert huber_val.item() < mse_val.item()  # 12.5 vs 0.74


# ---------------------------------------------------------------------------
# HuberLoss schedule
# ---------------------------------------------------------------------------


def test_huber_constant_delta_is_t_independent():
    """constant schedule 下 delta 跟 t 无关。"""
    huber = HuberLoss(c=0.15, schedule="constant")
    d_low = huber._compute_delta(torch.tensor([0.1]))
    d_high = huber._compute_delta(torch.tensor([0.9]))
    torch.testing.assert_close(d_low, d_high)
    torch.testing.assert_close(d_low, torch.tensor([0.15]))


def test_huber_snr_delta_decreases_with_t():
    """snr schedule：delta = c * sqrt((1-t)/t)，低 t 大 delta，t=0.5 时 delta=c。"""
    huber = HuberLoss(c=0.15, schedule="snr")
    delta_low = huber._compute_delta(torch.tensor([0.1]))
    delta_mid = huber._compute_delta(torch.tensor([0.5]))
    delta_high = huber._compute_delta(torch.tensor([0.9]))
    assert delta_low.item() > delta_mid.item() > delta_high.item()
    torch.testing.assert_close(delta_mid, torch.tensor([0.15]), rtol=1e-5, atol=1e-5)


def test_huber_sigma_delta_increases_with_t():
    """sigma schedule：delta = c * t/(1-t)，低 t 小 delta，跟 snr 反向。"""
    huber = HuberLoss(c=0.15, schedule="sigma")
    delta_low = huber._compute_delta(torch.tensor([0.1]))
    delta_mid = huber._compute_delta(torch.tensor([0.5]))
    delta_high = huber._compute_delta(torch.tensor([0.9]))
    assert delta_low.item() < delta_mid.item() < delta_high.item()
    torch.testing.assert_close(delta_mid, torch.tensor([0.15]), rtol=1e-5, atol=1e-5)


def test_huber_unknown_schedule_raises():
    with pytest.raises(ValueError, match="huber_schedule"):
        HuberLoss(c=0.15, schedule="nonexistent")


def test_huber_build_reads_args():
    class Args:
        huber_c = 0.3
        huber_schedule = "snr"
    h = build_huber(Args())
    assert h.c == 0.3
    assert h.schedule == "snr"


def test_huber_build_defaults_when_args_missing():
    """旧 args 没新字段时回落到默认 c=0.15 / schedule=constant。"""
    class OldArgs:
        pass
    h = build_huber(OldArgs())
    assert h.c == 0.15
    assert h.schedule == "constant"


# ---------------------------------------------------------------------------
# plugin registry
# ---------------------------------------------------------------------------


def test_builders_dict_keys():
    assert set(BUILDERS) == {"mse", "huber"}


def test_build_loss_dispatches_mse():
    class Args:
        loss_type = "mse"
    loss = build_loss(Args())
    assert isinstance(loss, MseLoss)


def test_build_loss_dispatches_huber():
    class Args:
        loss_type = "huber"
        huber_c = 0.2
        huber_schedule = "snr"
    loss = build_loss(Args())
    assert isinstance(loss, HuberLoss)
    assert loss.c == 0.2
    assert loss.schedule == "snr"


def test_build_loss_unknown_type_raises():
    class Args:
        loss_type = "ghost"
    with pytest.raises(ValueError, match="loss_type"):
        build_loss(Args())


def test_build_loss_defaults_to_mse_when_arg_missing():
    """args 没 loss_type 时回落到 mse（向后兼容）。"""
    class OldArgs:
        pass
    loss = build_loss(OldArgs())
    assert isinstance(loss, MseLoss)


def test_validate_schema_consistency_passes_on_clean_dev():
    """schema.loss_type Literal == BUILDERS keys；不抛即 pass。"""
    validate_schema_consistency()


# ---------------------------------------------------------------------------
# runtime_checkable Protocol
# ---------------------------------------------------------------------------


def test_mse_satisfies_loss_protocol():
    assert isinstance(MseLoss(), LossProtocol)


def test_huber_satisfies_loss_protocol():
    assert isinstance(HuberLoss(), LossProtocol)


# ---------------------------------------------------------------------------
# 集成：(B, C, H, W) latent shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss", [
    MseLoss(),
    HuberLoss(c=0.15, schedule="constant"),
    HuberLoss(c=0.15, schedule="snr"),
    HuberLoss(c=0.15, schedule="sigma"),
])
def test_compute_shape_and_finite_on_latent_like_input(loss):
    """所有 loss 在 latent-like (B, C, H, W) 输入上输出形状对齐 + 数值合法。"""
    torch.manual_seed(42)
    pred = torch.randn(4, 3, 8, 8)
    target = torch.randn(4, 3, 8, 8)
    t = torch.rand(4)
    out = loss.compute(pred, target, t)
    assert out.shape == pred.shape
    assert torch.isfinite(out).all()
    assert (out >= 0).all()  # huber/mse 都是非负
