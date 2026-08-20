"""TimestepSamplerProtocol pause/resume 支持（ADR 0006）。

主流 trainer（kohya / diffusers / SimpleTuner）的 resume 都不保自适应噪声采样器
内部状态 —— 它们没有自适应采样器。InfoNoise 是我们独有的，resume 丢 CDF / EMA / FIFO
等于 N_warm（默认 ~total_steps × 20%）warmup 白跑。

覆盖：
- baseline 默认实现：state_dict 返回 {}，load_state_dict 静默 no-op（含意外字段）
- InfoNoise 冷启动 roundtrip（CDF=None 路径）
- InfoNoise warmup 后 CDF 就绪 roundtrip + 所有字段对齐
- 同 torch RNG 下 resume 前后 sample 输出一致（核心保证）
- K / B shape mismatch 不抛，告警后保冷启动
- resume 后能继续 record / refresh（状态不污染）
- save_training_state / load_training_state 端到端集成
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# conftest.py 已把 REPO_ROOT 和 runtime/ 加到 sys.path
from training.state import load_training_state, save_training_state
from training.timestep_samplers import baseline as baseline_mod
from training.timestep_samplers import infonoise as infonoise_mod


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _drive_to_cdf_ready(sched, n_steps=120, seed=0):
    """喂 record + maybe_refresh 把 InfoNoise CDF 推到 ready。

    关键：t 在 t-space uniform 会让 log_sigma 集中中段，端 bin（极小/极大 σ）
    长期 _n_count<N_min → maybe_refresh 早退 `skipped_bins_not_full`，CDF 永远不就绪。
    走 log_sigma uniform 采样保证 K 个 bin 都被覆盖。
    """
    rng = np.random.default_rng(seed)
    log_sigma_lo = float(np.log(0.001 / 0.999))   # ≈ -6.91
    log_sigma_hi = float(np.log(0.999 / 0.001))   # ≈ +6.91
    for step in range(n_steps):
        bs = 8  # 比 K 大，期望每 step 多 bin 命中
        log_sigma = rng.uniform(log_sigma_lo, log_sigma_hi, size=bs)
        sigma = np.exp(log_sigma)
        t = (sigma / (1.0 + sigma)).astype(np.float32)
        # MSE 随 σ 单调（大噪声预测更难）→ 形成 schedule 信号
        mse = (0.1 + 2.0 * np.log1p(sigma) + rng.normal(0, 0.05, size=bs)).astype(np.float32)
        sched.record(torch.from_numpy(t), torch.from_numpy(mse))
        sched.maybe_refresh(global_step=step)


def _new_info_sched(**kw):
    """小尺寸 InfoNoise，N_warm 设为很小让测试快"""
    defaults = dict(K=8, N_warm=20, B=16, M=5, N_min=2)
    defaults.update(kw)
    return infonoise_mod.InfoNoiseScheduler(**defaults)


# ---------------------------------------------------------------------------
# baseline：Protocol 默认 no-op 实现
# ---------------------------------------------------------------------------

def test_baseline_state_dict_is_empty():
    s = baseline_mod.BaselineTimestepSampler(mode="logit_normal", shift=3.0)
    assert s.state_dict() == {}


def test_baseline_load_state_dict_accepts_anything():
    """default no-op：即便给了垃圾字段也不应抛 —— Protocol 契约"""
    s = baseline_mod.BaselineTimestepSampler(mode="logit_normal", shift=3.0)
    s.load_state_dict({})
    s.load_state_dict({"garbage": [1, 2, 3], "unexpected": "field"})  # 不应抛


# ---------------------------------------------------------------------------
# InfoNoise：state_dict / load_state_dict roundtrip
# ---------------------------------------------------------------------------

def test_infonoise_state_dict_roundtrip_cold():
    """构造好立即 state_dict → load 到另一个实例，字段对齐（CDF=None 路径）"""
    s1 = _new_info_sched()
    sd = s1.state_dict()
    assert sd["cdf_values"] is None  # 冷启动无 CDF
    assert sd["internal_step"] == 0

    s2 = _new_info_sched()
    s2.load_state_dict(sd)
    assert s2._internal_step == s1._internal_step
    assert s2._cdf_values is None
    assert np.array_equal(s2._mse_ema, s1._mse_ema)
    assert np.array_equal(s2._n_count, s1._n_count)
    for buf1, buf2 in zip(s1._fifo, s2._fifo):
        assert list(buf1) == list(buf2)
        assert buf2.maxlen == s2.B


def test_infonoise_state_dict_roundtrip_warm_cdf_ready():
    """跑出 CDF 后保存 → 恢复，CDF / FIFO / EMA / 计数全字段对齐"""
    s1 = _new_info_sched()
    _drive_to_cdf_ready(s1, n_steps=80)
    assert s1._cdf_values is not None, "测试 setup 没把 CDF 推 ready"
    assert s1._last_refresh_status == "ok"

    sd = s1.state_dict()
    s2 = _new_info_sched()
    s2.load_state_dict(sd)

    assert s2._internal_step == s1._internal_step
    assert s2._last_refresh_status == s1._last_refresh_status
    assert s2._refresh_attempts == s1._refresh_attempts
    assert s2._refresh_degraded_count == s1._refresh_degraded_count
    assert s2._warned_cold_start == s1._warned_cold_start
    np.testing.assert_allclose(s2._mse_ema, s1._mse_ema, rtol=1e-12, atol=1e-12)
    assert np.array_equal(s2._n_count, s1._n_count)
    np.testing.assert_allclose(s2._cdf_values, s1._cdf_values, rtol=1e-12, atol=1e-12)

    for i, (buf1, buf2) in enumerate(zip(s1._fifo, s2._fifo)):
        assert list(buf1) == list(buf2), f"fifo[{i}] 内容不一致"
        assert buf2.maxlen == s2.B, f"fifo[{i}] maxlen 不一致"


def test_infonoise_sample_deterministic_after_resume():
    """核心保证：同 torch RNG 下 resume 前后下一次 sample 输出**一致**。

    InfoNoise 的 sample 走 torch.rand → np.interp(u, cdf_edges, log_sigma_edges) →
    映射回 t。CDF 是关键状态，丢了等于走 baseline，分布完全不同。
    """
    s1 = _new_info_sched()
    _drive_to_cdf_ready(s1, n_steps=80)
    assert s1._cdf_values is not None

    torch.manual_seed(7)
    expected = s1.sample(8, device="cpu")

    sd = s1.state_dict()
    s2 = _new_info_sched()
    s2.load_state_dict(sd)

    torch.manual_seed(7)
    actual = s2.sample(8, device="cpu")
    assert torch.allclose(actual, expected, atol=1e-6), (
        f"resume 后采样分布漂移: expected={expected}, actual={actual}"
    )


def test_infonoise_sample_cold_vs_loaded_differs():
    """反向证明：如果不 load_state_dict，s2 走 baseline，与 s1 输出**不同** —— 证明
    test_infonoise_sample_deterministic_after_resume 的等价不是巧合"""
    s1 = _new_info_sched()
    _drive_to_cdf_ready(s1, n_steps=80)
    assert s1._cdf_values is not None

    torch.manual_seed(7)
    with_cdf = s1.sample(8, device="cpu")

    s2 = _new_info_sched()  # 冷启动，cdf None → 走 baseline
    assert s2._cdf_values is None
    torch.manual_seed(7)
    cold = s2.sample(8, device="cpu")

    # 至少一个元素不同 —— 不要求全不同（极端情况下 baseline 和 CDF 采样可能某个 t 偶合）
    assert not torch.allclose(with_cdf, cold, atol=1e-3), (
        "冷启动 sample 和 CDF-ready sample 输出意外相同 —— 测试 setup 没正确推 CDF？"
    )


# ---------------------------------------------------------------------------
# 错误恢复：shape mismatch 不抛
# ---------------------------------------------------------------------------

def test_infonoise_load_state_dict_K_mismatch_falls_back_cold(caplog):
    """K 改了（用户 resume 时不小心改了 infonoise_K hyperparameter）→ warning + 冷启动"""
    s_save = _new_info_sched(K=8)
    _drive_to_cdf_ready(s_save, n_steps=80)
    sd = s_save.state_dict()

    s_load = _new_info_sched(K=16)  # K 不一致
    with caplog.at_level("WARNING"):
        s_load.load_state_dict(sd)
    assert any("saved bin layout" in r.message for r in caplog.records)
    # 加载失败但不抛 —— 状态保持初始
    assert s_load._internal_step == 0
    assert s_load._cdf_values is None


def test_infonoise_load_state_dict_B_mismatch_falls_back_cold():
    s_save = _new_info_sched(B=16)
    _drive_to_cdf_ready(s_save, n_steps=80)
    sd = s_save.state_dict()

    s_load = _new_info_sched(B=64)
    s_load.load_state_dict(sd)
    assert s_load._internal_step == 0
    assert s_load._cdf_values is None


# ---------------------------------------------------------------------------
# 状态不污染：resume 后能继续推进
# ---------------------------------------------------------------------------

def test_infonoise_continues_recording_after_resume():
    """resume 后 record / maybe_refresh 应能继续推进 internal_step + 触发新 refresh"""
    s1 = _new_info_sched()
    _drive_to_cdf_ready(s1, n_steps=60)
    sd = s1.state_dict()

    s2 = _new_info_sched()
    s2.load_state_dict(sd)
    pre_step = s2._internal_step
    pre_attempts = s2._refresh_attempts

    _drive_to_cdf_ready(s2, n_steps=60, seed=1)
    assert s2._internal_step > pre_step
    assert s2._refresh_attempts > pre_attempts


def test_infonoise_fifo_maxlen_preserved_after_resume():
    """FIFO 在 resume 后仍尊重 maxlen=B —— deque 重建时 maxlen 必须传"""
    s1 = _new_info_sched(B=4)
    # 喂超过 B 个 record 到某 bin 触发 maxlen drop
    for _ in range(10):
        t = torch.tensor([0.5], dtype=torch.float32)
        mse = torch.tensor([0.1], dtype=torch.float32)
        s1.record(t, mse)
    # 某个 bin 应有 maxlen=4 个元素
    full_bins = [i for i, buf in enumerate(s1._fifo) if len(buf) == 4]
    assert full_bins, "测试 setup：没有 bin 被填满"

    sd = s1.state_dict()
    s2 = _new_info_sched(B=4)
    s2.load_state_dict(sd)
    for i in full_bins:
        assert s2._fifo[i].maxlen == 4
        # 再 append 一个验证 drop 行为
        s2._fifo[i].append(999.0)
        assert len(s2._fifo[i]) == 4
        assert 999.0 in s2._fifo[i]


# ---------------------------------------------------------------------------
# state.py 端到端集成
# ---------------------------------------------------------------------------

class _TinyInjector(nn.Module):
    """模拟 LycorisAdapter — save_training_state 只需要 .state_dict() 和 .load_state_dict()"""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.lin(x)


def _make_tiny_optimizer():
    m = _TinyInjector()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # 跑一步让 optimizer state 非空
    loss = m(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return m, opt


def test_save_load_training_state_persists_timestep_sampler(tmp_path):
    """save_training_state(..., timestep_sampler=sched) → load 时把 state 灌回新 sched"""
    injector, opt = _make_tiny_optimizer()
    sched = _new_info_sched()
    _drive_to_cdf_ready(sched, n_steps=80)
    assert sched._cdf_values is not None

    state_path = tmp_path / "state.pt"
    save_training_state(
        state_path, injector, opt, epoch=1, global_step=42,
        timestep_sampler=sched,
    )

    # 模拟 resume：新建 sched + injector + opt 从同一类型，load_training_state 灌回
    injector2 = _TinyInjector()
    opt2 = torch.optim.AdamW(injector2.parameters(), lr=1e-3)
    sched2 = _new_info_sched()
    assert sched2._cdf_values is None  # 加载前是冷启动

    epoch, step, _, _ = load_training_state(
        state_path, injector2, opt2, timestep_sampler=sched2,
    )
    assert (epoch, step) == (1, 42)
    assert sched2._cdf_values is not None, "InfoNoise CDF 没从 ckpt 恢复"
    np.testing.assert_allclose(sched2._cdf_values, sched._cdf_values, rtol=1e-12)


def test_save_skips_baseline_sampler_state(tmp_path):
    """baseline.state_dict() == {} → ckpt 里不应有 timestep_sampler_state key，省空间"""
    injector, opt = _make_tiny_optimizer()
    sched = baseline_mod.BaselineTimestepSampler(mode="logit_normal", shift=3.0)
    state_path = tmp_path / "state.pt"
    save_training_state(
        state_path, injector, opt, epoch=0, global_step=0,
        timestep_sampler=sched,
    )
    raw = torch.load(state_path, map_location="cpu", weights_only=False)
    assert "timestep_sampler_state" not in raw


def test_load_state_without_sampler_state_does_not_call_load(tmp_path):
    """ckpt 没有 timestep_sampler_state（旧 ckpt 或 baseline 时）→ sampler 不被调用，
    不应抛 / 不该被改 —— 后向兼容"""
    injector, opt = _make_tiny_optimizer()
    state_path = tmp_path / "state.pt"
    save_training_state(state_path, injector, opt, epoch=0, global_step=0)  # 不传 sampler

    injector2 = _TinyInjector()
    opt2 = torch.optim.AdamW(injector2.parameters(), lr=1e-3)
    sched2 = _new_info_sched()
    # 装填一些 state 到 sched2 — load 不应清掉它
    _drive_to_cdf_ready(sched2, n_steps=50)
    cdf_before = sched2._cdf_values.copy() if sched2._cdf_values is not None else None
    internal_before = sched2._internal_step

    load_training_state(state_path, injector2, opt2, timestep_sampler=sched2)

    # ckpt 没存 sampler → sched2 状态不该被动
    assert sched2._internal_step == internal_before
    if cdf_before is not None:
        np.testing.assert_array_equal(sched2._cdf_values, cdf_before)


def test_load_corrupted_sampler_state_logs_and_continues(tmp_path, caplog):
    """ckpt 里的 sampler state 损坏（如手动改 ckpt 删字段）→ load 应 warning 不崩"""
    injector, opt = _make_tiny_optimizer()
    sched = _new_info_sched()
    _drive_to_cdf_ready(sched, n_steps=80)
    state_path = tmp_path / "state.pt"
    save_training_state(
        state_path, injector, opt, epoch=0, global_step=0,
        timestep_sampler=sched,
    )

    # 手动破坏 ckpt
    raw = torch.load(state_path, map_location="cpu", weights_only=False)
    del raw["timestep_sampler_state"]["fifo"]  # 缺关键字段 → load_state_dict KeyError
    torch.save(raw, state_path)

    injector2 = _TinyInjector()
    opt2 = torch.optim.AdamW(injector2.parameters(), lr=1e-3)
    sched2 = _new_info_sched()
    with caplog.at_level("WARNING"):
        load_training_state(state_path, injector2, opt2, timestep_sampler=sched2)
    assert any("Timestep sampler state failed to restore" in r.message for r in caplog.records)
    # 训练状态本身仍正常返回
    assert sched2._cdf_values is None  # 没装上，保持冷启动
