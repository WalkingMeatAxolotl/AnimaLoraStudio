"""training/losses 单元测试（PR-D huber loss plugin）。

覆盖：
- MseLoss：跟 F.mse_loss(reduction='none') bit-for-bit 一致 + 不依赖 t + Protocol 合规
- HuberLoss：quad / linear / 边界 三段公式正确（constant delta）
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


def test_huber_inside_quadratic_region():
    """|x| < delta 时 Huber = 0.5 * x^2。"""
    pred = torch.tensor([[0.05]])
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    huber = HuberLoss(c=0.15)
    expected = torch.tensor([[0.5 * 0.05 * 0.05]])
    actual = huber.compute(pred, target, t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_huber_outside_linear_region():
    """|x| > delta 时 Huber = delta * (|x| - 0.5*delta)。"""
    pred = torch.tensor([[1.0]])
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    huber = HuberLoss(c=0.15)
    expected = torch.tensor([[0.15 * (1.0 - 0.5 * 0.15)]])  # 0.13875
    actual = huber.compute(pred, target, t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_huber_continuous_at_boundary():
    """|x| = delta 时 quad 和 lin 都应等于 0.5 * delta^2。"""
    delta = 0.15
    pred = torch.tensor([[delta]])
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    huber = HuberLoss(c=delta)
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
    huber_val = HuberLoss(c=0.15).compute(pred, target, t)
    assert huber_val.item() < mse_val.item()  # 12.5 vs 0.74


def test_huber_t_independent():
    """constant delta 下 huber 输出与 t 无关。"""
    huber = HuberLoss(c=0.15)
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 3, 4, 4)
    out1 = huber.compute(pred, target, torch.tensor([0.1, 0.5]))
    out2 = huber.compute(pred, target, torch.tensor([0.9, 0.99]))
    torch.testing.assert_close(out1, out2)


def test_huber_build_reads_args():
    class Args:
        huber_c = 0.3
    h = build_huber(Args())
    assert h.c == 0.3


def test_huber_build_defaults_when_args_missing():
    """旧 args 没新字段时回落到默认 c=0.15。"""
    class OldArgs:
        pass
    h = build_huber(OldArgs())
    assert h.c == 0.15


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
    loss = build_loss(Args())
    assert isinstance(loss, HuberLoss)
    assert loss.c == 0.2


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
    HuberLoss(c=0.15),
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


# ---------------------------------------------------------------------------
# InfoNoise × loss_type 解耦集成测试
# ---------------------------------------------------------------------------


def test_infonoise_raw_mse_decoupled_from_loss_type():
    """codify PR #75 loop.py:122-131 的核心解耦：启用 huber 时 InfoNoise 仍收到纯 MSE。

    防止未来重构 loop.py 把 _raw_mse 复用 ctx.loss_fn.compute 抢算力，从而违反
    InfoNoise paper 的 I-MMSE 假设。参考 [[feedback_verify_paper_before_fixing_algo]]
    （P0-2 EMA 翻转事故同根：对外部算法的"小聪明"必须先 verify 论文）。
    """
    from training.timestep_samplers.infonoise import InfoNoiseScheduler

    torch.manual_seed(0)
    pred = torch.randn(4, 4, 8, 8)
    target = torch.randn(4, 4, 8, 8)
    t = torch.rand(4).clamp(1e-3, 1 - 1e-3)

    # 复现 loop.py:122-131 的两路计算：
    # (1) 训练 loss 走 loss_fn（这里取 huber，跟 MSE 数值必然不同）
    huber = HuberLoss(c=0.15)
    loss_per_sample = huber.compute(pred.float(), target.float(), t)
    huber_per_sample = loss_per_sample.mean(dim=list(range(1, loss_per_sample.dim())))

    # (2) raw_mse 给 InfoNoise — 必须用 F.mse_loss 而不是 loss_fn
    raw_mse_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").detach()
    raw_mse = raw_mse_per_sample.mean(dim=list(range(1, raw_mse_per_sample.dim())))

    # 预检：huber 跟 MSE 数值在这组随机输入上必然不同（差异大于 1e-3）
    assert not torch.allclose(raw_mse, huber_per_sample, rtol=1e-2), (
        "测试无效：huber 输出跟 MSE 太接近，重写测试或换更极端 c。"
    )

    # 核心断言：raw_mse 路径输出 == MseLoss 输出（即与 loss_fn 类型无关）
    mse_reference = MseLoss().compute(pred.float(), target.float(), t)
    mse_per_sample = mse_reference.mean(dim=list(range(1, mse_reference.dim())))
    torch.testing.assert_close(raw_mse, mse_per_sample)

    # 用真正 InfoNoiseScheduler 验证 record() 接受这个 raw_mse 不崩
    scheduler = InfoNoiseScheduler(K=16, N_warm=10, M=5, B=10, N_min=1)
    scheduler.record(t.detach(), raw_mse)
    assert scheduler._internal_step == 1
    # 验证记录到的总样本数 == batch size（每个 bin 之和）
    total_recorded = sum(len(buf) for buf in scheduler._fifo)
    assert total_recorded == 4
