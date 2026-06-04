"""InfoNoise 单元测试。

InfoNoise 是 PR #63 引入的核心 timestep 采样算法。这些测试 codify
关键设计选择（EMA 方向、warmup 单位、gate pivot 默认值）防回归。

测试覆盖：
- EMA 设计选择（β 乘新值，FIFO + EMA 两层平滑）
- _refresh 早退分支正确记 status
- 退化时 sample() 走 baseline，sample() 输出 t 在 (0,1)
- baseline_mode 四种模式都能跑
- 冷启动 trip wire：CDF 没就绪时 maybe_refresh 发一次 logger.warning
- N_warm 按 optimizer step 计（防 grad_accum>1 时 warmup 提前结束）
- gate pivot 默认 c=0.15（防回归到 σ_min）
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from training.timestep_samplers import build_timestep_sampler
from training.timestep_samplers.infonoise import InfoNoiseScheduler
from training.timestep_samplers.infonoise import build as build_info_noise


# ---------------------------------------------------------------------------
# EMA 设计选择（防回归）
# ---------------------------------------------------------------------------


def test_ema_responsiveness_codifies_design_choice():
    """EMA 公式 new = (1-β)·old + β·l_bar；β 乘新值是 InfoNoise 设计选择。

    论文 §3.1 描述 "smoothed binwise estimate" 但未给字面公式；方向由实现固定。
    用 β=0.9 + 历史 1 + 新值 10 → 9.1 区分公式方向（β=0.5 给同样结果，区分不开）。
    """
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1, beta=0.9)
    # 设置初始历史
    s._mse_ema = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    # 给每个 bin 填一个值 10
    for k in range(4):
        s._fifo[k].append(10.0)
        s._n_count[k] = 1
    s._refresh()
    # 设计选择: new_ema = (1-0.9)*1 + 0.9*10 = 0.1 + 9.0 = 9.1
    # 反向：    new_ema = 0.9*1 + 0.1*10 = 0.9 + 1.0 = 1.9
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

    with caplog.at_level(logging.WARNING, logger="training.timestep_samplers.infonoise"):
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
        "kind",
        "cdf_ready",
        "last_refresh_status",
        "refresh_attempts",
        "refresh_degraded_count",
        "internal_step",
    }
    assert status["kind"] == "infonoise"
    assert status["cdf_ready"] is False
    assert status["last_refresh_status"] == "not_refreshed_yet"


# ---------------------------------------------------------------------------
# build_info_noise factory
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_build_timestep_sampler_disabled_returns_baseline():
    """P2-1 plugin registry：未启用 InfoNoise 时统一走 baseline（非 None）。"""
    args = _FakeArgs(
        infonoise_enabled=False,
        timestep_sampling="logit_normal",
        timestep_shift=3.0,
    )
    sampler = build_timestep_sampler(args, total_steps=1000)
    assert sampler is not None
    assert sampler.status()["kind"] == "baseline"


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


# ---------------------------------------------------------------------------
# Warmup gate 用 optimizer step（防 grad_accum>1 提前结束 — P0-1）
# ---------------------------------------------------------------------------


def test_n_warm_uses_optimizer_step_not_micro_batch():
    """maybe_refresh 内 warmup 判定用 global_step (optimizer step)。

    grad_accum>1 项目下，micro-batch 计数（_internal_step）增长比 global_step 快
    grad_accum 倍。如果 warmup 用 _internal_step 判定，grad_accum=4 下配置
    N_warm=100 实际只跑 25 个 optimizer step 就切自适应（早 4× 结束）。
    """
    s = InfoNoiseScheduler(K=4, N_warm=100, M=10, B=4, N_min=1, beta=0.9)
    # 模拟 grad_accum=4 跑 50 optimizer steps：200 个 micro-batch 喂 record
    for _ in range(200):
        s.record(torch.tensor([0.5] * 4), torch.tensor([1.0] * 4))
    # global_step=50 < N_warm=100 → maybe_refresh 必须早退（不管 _internal_step 已经 200）
    s.maybe_refresh(global_step=50)
    assert s._cdf_values is None
    # 防回归：明确断言 _internal_step 跟 global_step 是两个量
    assert s._internal_step == 200
    assert s._internal_step > 100  # 旧逻辑会让这里 ≥ N_warm 然后 refresh，新逻辑不会


# ---------------------------------------------------------------------------
# Gate pivot 默认 c=0.15（防回归到 σ_min — P0-5）
# ---------------------------------------------------------------------------


def test_default_gate_pivot_uses_paper_c_value():
    """gate_pivot_c 默认 0.15（论文 §5 CIFAR 报告值）。"""
    s = InfoNoiseScheduler(K=8, N_warm=1, M=1, B=2, N_min=1)
    assert s.gate_pivot_c == 0.15


def test_dynamic_gate_pivot_when_set_to_zero():
    """gate_pivot_c=0 走 dynamic last_above 路径（论文 Eq 87 字面实现）。"""
    s = InfoNoiseScheduler(K=8, N_warm=1, M=1, B=2, N_min=1, gate_pivot_c=0.0)
    # 注入中段 information mass 形状
    for k in range(8):
        val = 10.0 if 2 <= k <= 5 else 0.01
        s._fifo[k].append(val)
        s._n_count[k] = 1
    s._refresh()
    assert s._last_refresh_status == "ok"
    assert s._cdf_values is not None


def test_p_onset_rejected_when_invalid():
    """p_onset ∈ (0,1) 是 gate 早退分支不可达的前提；__init__ 应 fail-fast。"""
    with pytest.raises(ValueError, match="p_onset"):
        InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1, p_onset=1.5)


# ---------------------------------------------------------------------------
# state_dict version 兼容（防 σ³→σ² 公式切换的老 ckpt 漂移 — P0-4）
# ---------------------------------------------------------------------------


def test_state_dict_version_mismatch_triggers_cold_start():
    """老 ckpt（v1: σ³ 公式）resume 到新代码（v2: σ²）走冷启动，不偷偷读老 mse_ema。"""
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1)
    # 装一个 v1 state（无 __version__ 字段，等价 v1）
    old_state = {
        "K": 4, "B": 2,
        "fifo": [[1.0], [2.0], [3.0], [4.0]],
        "mse_ema": np.array([1.0, 2.0, 3.0, 4.0]),
        "n_count": np.array([1, 1, 1, 1], dtype=np.int32),
        "cdf_values": np.linspace(0, 1, 5),
        "internal_step": 100,
        "last_refresh_status": "ok",
        "refresh_attempts": 5,
        "refresh_degraded_count": 0,
        "warned_cold_start": False,
    }
    s.load_state_dict(old_state)
    # v1 → v2 → 冷启动：mse_ema 保持 zeros，cdf_values 保持 None
    np.testing.assert_array_equal(s._mse_ema, np.zeros(4))
    assert s._cdf_values is None
    assert s._internal_step == 0


def test_state_dict_round_trip_v2():
    """v2 state_dict 正确 round-trip。"""
    s1 = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1)
    for k in range(4):
        s1._fifo[k].append(float(k))
    s1._mse_ema = np.array([1.0, 2.0, 3.0, 4.0])
    state = s1.state_dict()
    assert state["__version__"] == 2

    s2 = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1)
    s2.load_state_dict(state)
    np.testing.assert_array_equal(s2._mse_ema, s1._mse_ema)
    assert list(s2._fifo[2]) == [2.0]
