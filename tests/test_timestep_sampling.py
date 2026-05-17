"""timestep_sampling 单元测试。

主要覆盖：
- 4 个原 mode 在默认 mix_low_prob=0/timestep_schedule_shift=1.0 下行为不变（防回归）
- 2 个新 mode（mixed_uniform_low / mixed_uniform_logit）p=0/p=1/部分混合三种情形
- timestep_schedule_shift 公式数学正确 + 1.0 时恒等 + >1 推高均值
- BaselineTimestepSampler 接收+透传新参 + build() 读 args
- InfoNoise build() 读新参 + baseline 走 sample_t 时透传

随机性大的用大 batch (>=4096) + 统计性 assertion 避免 flakiness。
"""
from __future__ import annotations

import torch

from training.timestep_sampling import _apply_timestep_schedule_shift, sample_t
from training.timestep_samplers.baseline import BaselineTimestepSampler
from training.timestep_samplers.baseline import build as build_baseline
from training.timestep_samplers.infonoise import InfoNoiseScheduler
from training.timestep_samplers.infonoise import build as build_infonoise


# ---------------------------------------------------------------------------
# timestep_schedule_shift 数学
# ---------------------------------------------------------------------------


def test_timestep_schedule_shift_identity_when_one():
    t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    out = _apply_timestep_schedule_shift(t, 1.0)
    torch.testing.assert_close(out, t)


def test_timestep_schedule_shift_formula_matches_spec():
    """t' = (t * s) / (1 + (s-1)*t); s=2, t=0.5 → 1/1.5 = 2/3"""
    t = torch.tensor([0.5])
    out = _apply_timestep_schedule_shift(t, 2.0)
    torch.testing.assert_close(out, torch.tensor([2.0 / 3.0]), rtol=1e-5, atol=1e-5)


def test_timestep_schedule_shift_clamps_to_open_interval():
    """t∈(0,1) 后经 shift 也应保持 (0, 1) 开区间（clamp 到 [eps, 1-eps]）。"""
    t = torch.tensor([1e-5, 0.5, 1 - 1e-5])
    out = _apply_timestep_schedule_shift(t, 5.0)
    assert (out > 0).all() and (out < 1).all()


# ---------------------------------------------------------------------------
# 默认行为回归（4 个原 mode 在新参默认值下统计上等价于历史）
# ---------------------------------------------------------------------------


def test_legacy_logit_normal_shift_pushes_high():
    """shift=3 logit_normal 应推向高 t（mean > 0.5）。"""
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="logit_normal", shift=3.0)
    assert t.mean().item() > 0.55
    assert (t > 0).all() and (t < 1).all()


def test_legacy_uniform_mean_half():
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="uniform")
    assert abs(t.mean().item() - 0.5) < 0.05


def test_legacy_logit_normal_low_pushes_low():
    """logit_normal_low 反向 shift 应偏向低 t（mean < 0.5）。"""
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="logit_normal_low", shift=3.0)
    assert t.mean().item() < 0.45


def test_legacy_mode_outputs_in_range():
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="mode", shift=3.0)
    assert (t > 0).all() and (t < 1).all()


# ---------------------------------------------------------------------------
# mixed_uniform_low
# ---------------------------------------------------------------------------


def test_mixed_uniform_low_p_zero_is_pure_uniform():
    """p=0 时不取偏置端，分布等价于 uniform（mean≈0.5）。"""
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="mixed_uniform_low", mix_low_prob=0.0)
    assert abs(t.mean().item() - 0.5) < 0.05


def test_mixed_uniform_low_p_one_matches_logit_normal_low_distribution():
    """p=1 时所有样本走 logit_normal_low，分布偏低 t，mean 跟 pure logit_normal_low 接近。"""
    torch.manual_seed(0)
    t_mixed = sample_t(8192, "cpu", mode="mixed_uniform_low", shift=3.0, mix_low_prob=1.0)
    torch.manual_seed(0)
    t_pure = sample_t(8192, "cpu", mode="logit_normal_low", shift=3.0)
    # 两者均值都应该 < 0.5 且接近
    assert t_mixed.mean().item() < 0.45
    assert t_pure.mean().item() < 0.45
    assert abs(t_mixed.mean().item() - t_pure.mean().item()) < 0.03


def test_mixed_uniform_low_partial_interpolates():
    """p=0.3 时均值应该介于 pure uniform (0.5) 和 pure low 之间。"""
    torch.manual_seed(0)
    t_mixed = sample_t(8192, "cpu", mode="mixed_uniform_low", shift=3.0, mix_low_prob=0.3)
    torch.manual_seed(0)
    t_pure_low = sample_t(8192, "cpu", mode="logit_normal_low", shift=3.0)
    low_mean = t_pure_low.mean().item()
    # 介于 pure low (e.g. 0.25) 和 uniform (0.5) 之间
    assert low_mean < t_mixed.mean().item() < 0.5


# ---------------------------------------------------------------------------
# mixed_uniform_logit
# ---------------------------------------------------------------------------


def test_mixed_uniform_logit_p_one_matches_logit_normal_distribution():
    """p=1 时分布跟 pure logit_normal（shift=3）接近，mean 都 > 0.5。"""
    torch.manual_seed(0)
    t_mixed = sample_t(8192, "cpu", mode="mixed_uniform_logit", shift=3.0, mix_low_prob=1.0)
    torch.manual_seed(0)
    t_pure = sample_t(8192, "cpu", mode="logit_normal", shift=3.0)
    assert t_mixed.mean().item() > 0.55
    assert abs(t_mixed.mean().item() - t_pure.mean().item()) < 0.03


def test_mixed_uniform_logit_partial_pulls_toward_high():
    """shift=3 偏置端推高 t；p=0.3 应让均值略 > 0.5。"""
    torch.manual_seed(0)
    t = sample_t(8192, "cpu", mode="mixed_uniform_logit", shift=3.0, mix_low_prob=0.3)
    assert t.mean().item() > 0.5


# ---------------------------------------------------------------------------
# timestep_schedule_shift 跟分布混合
# ---------------------------------------------------------------------------


def test_timestep_schedule_shift_high_value_raises_mean():
    """timestep_schedule_shift > 1 推高 mean。"""
    torch.manual_seed(0)
    t_base = sample_t(4096, "cpu", mode="uniform", timestep_schedule_shift=1.0)
    torch.manual_seed(0)
    t_high = sample_t(4096, "cpu", mode="uniform", timestep_schedule_shift=3.0)
    assert t_high.mean().item() > t_base.mean().item() + 0.05


def test_timestep_schedule_shift_applies_to_mixed_mode():
    """mixed_uniform_low + timestep_schedule_shift=2 输出应推高（覆盖偏低端原本的均值）。"""
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="mixed_uniform_low", shift=3.0, mix_low_prob=1.0, timestep_schedule_shift=3.0)
    # 即使是 mixed_uniform_low (本应偏低)，timestep_schedule_shift=3 也应把均值推回 > 0.4
    assert t.mean().item() > 0.4


# ---------------------------------------------------------------------------
# BaselineTimestepSampler 集成
# ---------------------------------------------------------------------------


def test_baseline_sampler_accepts_new_params():
    s = BaselineTimestepSampler(
        mode="mixed_uniform_low",
        shift=3.0,
        mix_low_prob=0.5,
        timestep_schedule_shift=1.5,
    )
    t = s.sample(64, "cpu")
    assert t.shape == (64,)
    assert (t > 0).all() and (t < 1).all()
    status = s.status()
    assert status["mode"] == "mixed_uniform_low"
    assert status["mix_low_prob"] == 0.5
    assert status["timestep_schedule_shift"] == 1.5


def test_baseline_build_reads_new_args():
    """build() 从 args 读取新参。"""
    class Args:
        timestep_sampling = "mixed_uniform_low"
        timestep_shift = 2.0
        timestep_mix_low_prob = 0.25
        timestep_schedule_shift = 1.12

    s = build_baseline(Args(), total_steps=1000)
    assert s.mode == "mixed_uniform_low"
    assert s.mix_low_prob == 0.25
    assert s.timestep_schedule_shift == 1.12


def test_baseline_build_backward_compatible_with_missing_new_args():
    """旧 args namespace 缺新参时，build() 应回落到默认值（getattr 默认）。"""
    class OldArgs:
        timestep_sampling = "logit_normal"
        timestep_shift = 3.0
        # 缺 timestep_mix_low_prob / timestep_schedule_shift

    s = build_baseline(OldArgs(), total_steps=1000)
    assert s.mix_low_prob == 0.0
    assert s.timestep_schedule_shift == 1.0


def test_baseline_sampler_default_args_unchanged():
    """无显式传参时跟历史 BaselineTimestepSampler() 行为等价（mix_low_prob=0, timestep_schedule_shift=1）。"""
    s = BaselineTimestepSampler()
    assert s.mix_low_prob == 0.0
    assert s.timestep_schedule_shift == 1.0
    t = s.sample(128, "cpu")
    # 默认 mode=logit_normal shift=3 → mean > 0.5
    assert t.mean().item() > 0.5


# ---------------------------------------------------------------------------
# InfoNoise 兼容
# ---------------------------------------------------------------------------


def test_infonoise_build_reads_new_args():
    class Args:
        timestep_sampling = "mixed_uniform_low"
        timestep_shift = 3.0
        timestep_mix_low_prob = 0.2
        timestep_schedule_shift = 1.5
        infonoise_K = 32
        infonoise_N_warm = 0  # auto
        infonoise_M = 50
        infonoise_B = 128
        infonoise_beta = 0.9
        infonoise_N_min = 10

    s = build_infonoise(Args(), total_steps=500)
    assert s.baseline_mode == "mixed_uniform_low"
    assert s.baseline_mix_low_prob == 0.2
    assert s.baseline_timestep_schedule_shift == 1.5


def test_infonoise_warmup_sample_uses_new_params():
    """CDF 未就绪时 sample 走 _sample_baseline，应该正确透传新参（不崩溃，输出合法 t）。"""
    s = InfoNoiseScheduler(
        K=32, N_warm=100, M=10, B=10, N_min=1,
        baseline_mode="mixed_uniform_low",
        baseline_shift=3.0,
        baseline_mix_low_prob=0.5,
        baseline_timestep_schedule_shift=1.0,
    )
    assert s._cdf_values is None
    t = s.sample(128, "cpu")
    assert t.shape == (128,)
    assert (t > 0).all() and (t < 1).all()


def test_infonoise_default_baseline_params_unchanged():
    """InfoNoiseScheduler 无显式 baseline_mix_low_prob/timestep_schedule_shift 时回落到默认。"""
    s = InfoNoiseScheduler(K=32, N_warm=100, M=10, B=10, N_min=1)
    assert s.baseline_mix_low_prob == 0.0
    assert s.baseline_timestep_schedule_shift == 1.0
