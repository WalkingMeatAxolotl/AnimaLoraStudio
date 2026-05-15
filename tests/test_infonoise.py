"""InfoNoise 单元测试（P2-3）。

InfoNoise 是 PR #63 引入的核心 timestep 采样算法。这次重构（PR #65）一度
把 EMA 公式按"直觉"翻反了，正是因为没有任何测试 codify 论文原意。

测试覆盖：
- EMA 公式 = 论文 Algorithm 1（β 乘新值）
- _refresh 三个早退分支正确记 status
- 退化时 sample() 走 baseline，sample() 输出 t 在 (0,1)
- baseline_mode 四种模式都能跑（P1-3）
- 冷启动 trip wire：CDF 没就绪时 maybe_refresh 发一次 logger.warning
- N_warm = 0 自动按 total_steps × 20% 计算（最少 200）
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from training.infonoise import InfoNoiseScheduler, build_info_noise


# ---------------------------------------------------------------------------
# EMA 公式（防 P0-2 类回归）
# ---------------------------------------------------------------------------


def test_ema_formula_matches_paper_algorithm1():
    """论文 arxiv 2602.18647 Algorithm 1 第 11 行：mse^k ← (1-β)·mse^k + β·ℓ̄_k

    β 乘的是**新值**，不是历史。β=0.5 + 历史 0 + 新值 10 → 5。
    若公式反了（β*old + (1-β)*new），β=0.5 给出同样结果，区分不开 —— 用 β=0.9 测。
    """
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1, beta=0.9)
    # 设置初始历史
    s._mse_ema = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    # 给每个 bin 填一个值 10
    for k in range(4):
        s._fifo[k].append(10.0)
        s._n_count[k] = 1
    s._refresh()
    # 论文公式: new_ema = (1-0.9)*1 + 0.9*10 = 0.1 + 9.0 = 9.1
    # 反向公式: new_ema = 0.9*1 + 0.1*10 = 0.9 + 1.0 = 1.9
    np.testing.assert_allclose(s._mse_ema, 9.1, rtol=1e-6)


def test_ema_beta_high_means_high_responsiveness():
    """β=0.99 意味着新值占 99% 权重，强响应；β=0.01 历史占 99%，强平滑。"""
    s_responsive = InfoNoiseScheduler(K=2, N_warm=1, M=1, B=2, N_min=1, beta=0.99)
    s_responsive._mse_ema = np.array([0.0, 0.0], dtype=np.float64)
    for k in range(2):
        s_responsive._fifo[k].append(100.0)
        s_responsive._n_count[k] = 1
    s_responsive._refresh()
    assert s_responsive._mse_ema[0] >= 99.0  # 几乎全新值

    s_smooth = InfoNoiseScheduler(K=2, N_warm=1, M=1, B=2, N_min=1, beta=0.01)
    s_smooth._mse_ema = np.array([0.0, 0.0], dtype=np.float64)
    for k in range(2):
        s_smooth._fifo[k].append(100.0)
        s_smooth._n_count[k] = 1
    s_smooth._refresh()
    assert s_smooth._mse_ema[0] <= 2.0  # 几乎全历史（仍是 0）


# ---------------------------------------------------------------------------
# _refresh 三个早退分支 status 标记（P1-1）
# ---------------------------------------------------------------------------


def test_refresh_status_mse_collapsed():
    """所有 bin 的 EMA 都 ~ 0 → r_hat 最大值 < 1e-30 → 标 mse_collapsed。"""
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1, beta=0.5)
    for k in range(4):
        s._fifo[k].append(0.0)
        s._n_count[k] = 1
    s._refresh()
    assert s._last_refresh_status == "mse_collapsed"
    assert s._cdf_values is None
    assert s._refresh_degraded_count == 1


def test_refresh_status_ok_path():
    """正常 mse 应该走完整条 pipeline，标 ok 并写 _cdf_values。"""
    s = InfoNoiseScheduler(K=8, N_warm=1, M=1, B=2, N_min=1, beta=0.5)
    # 中间几个 bin 给较大 mse，模拟有信息窗口
    for k in range(8):
        val = 10.0 if 2 <= k <= 5 else 0.01
        s._fifo[k].append(val)
        s._n_count[k] = 1
    s._refresh()
    assert s._last_refresh_status == "ok"
    assert s._cdf_values is not None
    # CDF 必须单调非降，从 0 到 1
    cdf = s._cdf_values
    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0
    assert np.all(np.diff(cdf) >= -1e-12)


# ---------------------------------------------------------------------------
# sample() 行为
# ---------------------------------------------------------------------------


def test_sample_falls_back_to_baseline_when_cdf_not_ready():
    s = InfoNoiseScheduler(K=4, N_warm=10, M=5, B=2, N_min=1)
    # CDF 未就绪，sample 必须走 baseline 不报错
    t = s.sample(4, "cpu")
    assert t.shape == (4,)
    assert (t > 0).all() and (t < 1).all()


def test_sample_after_refresh_in_unit_range():
    s = InfoNoiseScheduler(K=8, N_warm=1, M=1, B=2, N_min=1)
    for k in range(8):
        val = 10.0 if 2 <= k <= 5 else 0.01
        s._fifo[k].append(val)
        s._n_count[k] = 1
    s._refresh()
    assert s._cdf_values is not None
    t = s.sample(16, "cpu")
    assert (t > 0).all() and (t < 1).all()


@pytest.mark.parametrize("mode", ["logit_normal", "uniform", "logit_normal_low", "mode"])
def test_baseline_mode_works_for_all_options(mode):
    """P1-3 回归：warmup 必须复用 sample_t 四种模式都能跑。"""
    s = InfoNoiseScheduler(K=4, N_warm=10, M=5, B=2, N_min=1, baseline_mode=mode)
    t = s._sample_baseline(8, "cpu")
    assert t.shape == (8,)
    assert (t > 0).all() and (t < 1).all()


# ---------------------------------------------------------------------------
# 冷启动 trip wire（P1-1）
# ---------------------------------------------------------------------------


def test_cold_start_warning_emits_once(caplog):
    """warmup 过 + bin 样本充足 + _refresh 早退 → 一次性 logger.warning。"""
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1)
    # 全 0 mse → mse_collapsed
    for k in range(4):
        for _ in range(2):
            s._fifo[k].append(0.0)
            s._n_count[k] = 2
    s._internal_step = 100  # 装作已过 warmup

    with caplog.at_level(logging.WARNING, logger="training.infonoise"):
        s.maybe_refresh(global_step=1)
        s.maybe_refresh(global_step=2)
        s.maybe_refresh(global_step=3)
    # 三次 refresh 但 warning 只发一次
    warnings = [r for r in caplog.records if "InfoNoise" in r.message]
    assert len(warnings) == 1
    assert "mse_collapsed" in warnings[0].message


def test_status_dict_shape():
    s = InfoNoiseScheduler(K=4, N_warm=10, M=5, B=2, N_min=1)
    status = s.status()
    assert set(status.keys()) == {
        "cdf_ready",
        "last_refresh_status",
        "refresh_attempts",
        "refresh_degraded_count",
        "internal_step",
    }
    assert status["cdf_ready"] is False
    assert status["last_refresh_status"] == "not_refreshed_yet"


# ---------------------------------------------------------------------------
# build_info_noise factory
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_build_info_noise_disabled_returns_none():
    args = _FakeArgs(infonoise_enabled=False)
    assert build_info_noise(args, total_steps=1000) is None


def test_build_info_noise_n_warm_auto_20_pct():
    args = _FakeArgs(infonoise_enabled=True, infonoise_N_warm=0, timestep_sampling="logit_normal", timestep_shift=3.0)
    s = build_info_noise(args, total_steps=10_000)
    assert s is not None
    # 10000 × 20% = 2000
    assert s.N_warm == 2000


def test_build_info_noise_n_warm_min_200():
    args = _FakeArgs(infonoise_enabled=True, infonoise_N_warm=0, timestep_sampling="logit_normal", timestep_shift=3.0)
    s = build_info_noise(args, total_steps=100)
    assert s is not None
    # 100 × 20% = 20 < 200，应用 max(200, ...) 后是 200
    assert s.N_warm == 200


def test_build_info_noise_passes_baseline_mode():
    """P1-3 集成：build_info_noise 必须把 args.timestep_sampling 传给 scheduler。"""
    args = _FakeArgs(
        infonoise_enabled=True, infonoise_N_warm=500,
        timestep_sampling="uniform", timestep_shift=3.0,
    )
    s = build_info_noise(args, total_steps=10_000)
    assert s is not None
    assert s.baseline_mode == "uniform"
