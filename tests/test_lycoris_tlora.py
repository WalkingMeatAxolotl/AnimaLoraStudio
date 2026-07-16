"""T-LoRA timestep mask 方向 + sample 清 mask + bypass 不变式 单测。

锁定与官方 ControlGenAI/T-LoRA (arxiv 2507.05964, https://github.com/ControlGenAI/T-LoRA)
公式对齐：
- t=0 (clean, FLUX 约定 noisy = (1-t)*x + t*eps) → 满 rank
- t=1 (max noise)                              → min_rank
- alpha=1.0 等价 FLUX 线性 schedule
- algo='tlora' 永远不走 bypass_mode（否则 make_weight patch 失效）

为了绕开 lycoris-lora 不同版本 preset / fnmatch 接口差异（test_lycoris_bypass.py
也踩同样的坑），mask schedule 单元测试不走 AnimaLycorisAdapter.inject() — 直接
构造 adapter 内部状态测 `_set_tlora_mask` / `clear_timestep_mask` 的纯算法部分。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("lycoris")

from utils.lycoris_adapter import AnimaLycorisAdapter  # noqa: E402
from training.families.anima.preset import ANIMA_PRESET


def _bare_tlora_adapter(rank: int, min_rank: int, alpha: float) -> AnimaLycorisAdapter:
    """构造 AnimaLycorisAdapter 但不调 inject() — 手动 setup mask buffer +
    单个虚拟 lora 模块占位，让 `_set_tlora_mask` 跑得通。"""
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, 
        algo="tlora",
        rank=rank,
        alpha=float(rank),
        tlora_min_rank=min_rank,
        tlora_alpha_rank_scale=alpha,
    )
    # 占位 lora 模块：只需要支持 setattr `_anima_tlora_mask`。空 nn.Module 够用。
    placeholder = nn.Module()
    adapter._tlora_modules = [placeholder]
    return adapter


# ── test 1: mask 方向与官方 ControlGenAI/T-LoRA 一致 ───────────────────────


def test_tlora_mask_clean_timestep_full_rank() -> None:
    """t=0 (FLUX clean) → mask 全 1 (满 rank)。与官方 (max_t - t)/max_t = 1 对齐。"""
    rank, min_rank = 32, 4
    adapter = _bare_tlora_adapter(rank=rank, min_rank=min_rank, alpha=1.0)
    adapter._set_tlora_mask(torch.tensor([0.0]))
    mask = adapter._tlora_mask
    assert mask is not None
    assert int(mask.sum().item()) == rank, (
        f"t=0 应全 rank ({rank})，实际 active={int(mask.sum().item())}"
    )


def test_tlora_mask_max_noise_timestep_min_rank() -> None:
    """t=1 (FLUX max noise) → 只有前 min_rank 个 active。"""
    rank, min_rank = 32, 4
    adapter = _bare_tlora_adapter(rank=rank, min_rank=min_rank, alpha=1.0)
    adapter._set_tlora_mask(torch.tensor([1.0]))
    mask = adapter._tlora_mask
    assert mask is not None
    assert int(mask.sum().item()) == min_rank, (
        f"t=1 应 min_rank ({min_rank})，实际 active={int(mask.sum().item())}"
    )
    # principal sig_type：前 min_rank 个 = 1，其余 = 0
    expected = torch.cat(
        [torch.ones(min_rank), torch.zeros(rank - min_rank)]
    ).to(mask.device)
    assert torch.equal(mask, expected)


def test_tlora_mask_midpoint_linear() -> None:
    """t=0.5, alpha=1.0 → 线性插值 active = min_rank + 0.5*(rank-min_rank)。"""
    rank, min_rank = 32, 4
    adapter = _bare_tlora_adapter(rank=rank, min_rank=min_rank, alpha=1.0)
    adapter._set_tlora_mask(torch.tensor([0.5]))
    # frac = (1 - 0.5)^1 = 0.5; active = 4 + 0.5*28 = 18.0; floor(18) = 18
    expected = min_rank + 0.5 * (rank - min_rank)
    mask = adapter._tlora_mask
    assert int(mask.sum().item()) == int(expected), (
        f"t=0.5, alpha=1 应 active={int(expected)}，实际={int(mask.sum().item())}"
    )


def test_tlora_mask_alpha_power_steepens_schedule() -> None:
    """alpha>1：高噪声端比线性更小（schedule 更陡向 low-rank 倾）。

    t=0.5, alpha=2 → frac = (1-0.5)^2 = 0.25 → active = 4 + 0.25*28 = 11
    vs alpha=1 同 t 下 = 18，alpha=2 更小。
    """
    rank, min_rank = 32, 4
    adapter = _bare_tlora_adapter(rank=rank, min_rank=min_rank, alpha=2.0)
    adapter._set_tlora_mask(torch.tensor([0.5]))
    expected = min_rank + (1 - 0.5) ** 2 * (rank - min_rank)
    mask = adapter._tlora_mask
    assert int(mask.sum().item()) == int(expected), (
        f"t=0.5, alpha=2 应 active={int(expected)}，实际={int(mask.sum().item())}"
    )


def test_tlora_mask_monotone_t_to_active_rank() -> None:
    """单调性：t 升高 → active rank 不增加（与官方 noisy→低 rank 方向锁死，
    防止公式被误改回 PR 原来的反向）。"""
    adapter = _bare_tlora_adapter(rank=32, min_rank=4, alpha=1.0)
    prev = 33  # > rank
    for t in torch.linspace(0.0, 1.0, 11):
        adapter._set_tlora_mask(t.reshape(1))
        cur = int(adapter._tlora_mask.sum().item())
        assert cur <= prev, f"非单调：t={float(t):.2f} active={cur} > prev={prev}"
        prev = cur


# ── test 2: clear_timestep_mask 把 mask 还原成全 1 ────────────────────────


def test_tlora_clear_mask_restores_full_rank() -> None:
    """sample 阶段调 clear_timestep_mask → mask 全 1。
    与官方推理 mask=None fallback 出全 1 一致。"""
    rank, min_rank = 32, 4
    adapter = _bare_tlora_adapter(rank=rank, min_rank=min_rank, alpha=1.0)
    # 先制造一个 partial mask（t=0.7）
    adapter._set_tlora_mask(torch.tensor([0.7]))
    assert int(adapter._tlora_mask.sum().item()) < rank, "前置：mask 应该 partial"
    # 清除 → 全 1
    adapter.clear_timestep_mask()
    assert int(adapter._tlora_mask.sum().item()) == rank, (
        f"clear 后应该全 rank ({rank})，"
        f"实际 active={int(adapter._tlora_mask.sum().item())}"
    )
    # 后续 _set_tlora_mask 仍能正常工作（下一步训练立即重写）
    adapter._set_tlora_mask(torch.tensor([1.0]))
    assert int(adapter._tlora_mask.sum().item()) == min_rank


# ── test 3: sample_runner 入口调 clear_timestep_mask ────────────────────


def test_run_sample_clears_tlora_mask() -> None:
    """sample_runner.run_sample 入口必须调 injector.clear_timestep_mask（如有）。
    用 stub injector 验证 hook 被调；不实际跑模型。"""
    from unittest.mock import MagicMock
    from runtime.training import sample_runner as sr
    from runtime.training.families.anima import ANIMA_SPEC
    from runtime.training.sample_runner import run_sample

    clear_called = {"n": 0}

    class _StubInjector:
        def clear_timestep_mask(self) -> None:
            clear_called["n"] += 1

    ctx = MagicMock()
    ctx.injector = _StubInjector()
    ctx.args = MagicMock(
        resolution=64,
        sample_width=0,
        sample_height=0,
        sample_cfg_scale=1.0,
        sample_negative_prompt="",
        sample_seed=0,
        sample_infer_steps=1,
        sample_sampler_name="er_sde",
        sample_scheduler="simple",
    )
    ctx.optimizer = MagicMock()
    ctx.model = MagicMock()
    ctx.wandb_monitor = MagicMock(log_samples=False)
    ctx.monitor_server = None
    ctx.dtype = torch.float32
    ctx.device = torch.device("cpu")
    # 采样经 ctx.family 派发：spec 要真的（run_sample 拿 align_px / 采样默认值
    # 做算术），出图本身由 MagicMock ctx 自动 stub 掉，不真跑模型。
    ctx.family.spec = ANIMA_SPEC

    # stub 掉 optimizer_eval_mode
    from contextlib import contextmanager

    @contextmanager
    def _noop_ctx(*a, **kw):
        yield

    orig_eval = sr.optimizer_eval_mode
    sr.optimizer_eval_mode = _noop_ctx  # type: ignore[assignment]
    ctx.emit = MagicMock()

    try:
        run_sample(ctx, prompt="x", sample_path=MagicMock())
    finally:
        sr.optimizer_eval_mode = orig_eval  # type: ignore[assignment]

    assert clear_called["n"] == 1, (
        "run_sample 必须调 injector.clear_timestep_mask() 让 sample 走满 rank"
    )


def test_run_sample_no_clear_method_does_not_crash() -> None:
    """non-tlora adapter (lokr / lora / loha) 没有 clear_timestep_mask 方法，
    run_sample 必须安全跳过，不能 AttributeError。"""
    from unittest.mock import MagicMock
    from runtime.training import sample_runner as sr
    from runtime.training.families.anima import ANIMA_SPEC
    from runtime.training.sample_runner import run_sample

    class _NonTloraInjector:
        # 故意没有 clear_timestep_mask 方法
        pass

    ctx = MagicMock()
    ctx.injector = _NonTloraInjector()
    ctx.args = MagicMock(
        resolution=64,
        sample_width=0,
        sample_height=0,
        sample_cfg_scale=1.0,
        sample_negative_prompt="",
        sample_seed=0,
        sample_infer_steps=1,
        sample_sampler_name="er_sde",
        sample_scheduler="simple",
    )
    ctx.optimizer = MagicMock()
    ctx.model = MagicMock()
    ctx.wandb_monitor = MagicMock(log_samples=False)
    ctx.monitor_server = None
    ctx.dtype = torch.float32
    ctx.device = torch.device("cpu")
    ctx.family.spec = ANIMA_SPEC

    from contextlib import contextmanager

    @contextmanager
    def _noop_ctx(*a, **kw):
        yield

    orig_eval = sr.optimizer_eval_mode
    sr.optimizer_eval_mode = _noop_ctx  # type: ignore[assignment]
    ctx.emit = MagicMock()

    try:
        # 不应 raise
        run_sample(ctx, prompt="x", sample_path=MagicMock())
    finally:
        sr.optimizer_eval_mode = orig_eval  # type: ignore[assignment]


# ── test 4: bypass_mode invariant ──────────────────────────────────────────
#
# 设计 invariant：AnimaLycorisAdapter.inject() 里 line 111-112 只在
#   self.algo == "lora" and not self.weight_decompose
# 时设置 bypass_mode=True；algo == "tlora" 永远不进此分支，必走 rebuild
# (make_weight) 路径，让 _install_tlora_masks 的 mask patch 真正生效。
#
# 这里不依赖完整 inject (lycoris preset 与本机版本不兼容)，直接 inspect
# AnimaLycorisAdapter.inject 源码确认 tlora 不会被设 bypass_mode=True。


def test_tlora_inject_never_sets_bypass_mode() -> None:
    """inject() 里给 lycoris 的 extra dict 设 bypass_mode=True 这件事只在
    algo='lora' 分支发生；algo='tlora' 不进，必走 make_weight rebuild 路径让
    _install_tlora_masks 的 mask patch 生效。

    防止后续维护误把 tlora 也加进 bypass 路径让 mask 静默失效。"""
    import re
    import inspect
    src = inspect.getsource(AnimaLycorisAdapter.inject)
    # 找形如 `extra["bypass_mode"] = ...` 或 `extra['bypass_mode'] = ...` 的赋值
    pattern = re.compile(r'extra\[["\']bypass_mode["\']\]\s*=', re.MULTILINE)
    matches = list(pattern.finditer(src))
    assert matches, (
        "inject 源码里找不到 extra['bypass_mode'] 赋值；"
        "源码结构变了，需重写本 invariant 测试或重新评估 bypass_mode 默认行为。"
    )
    for m in matches:
        # 该赋值之前 ~250 字符内必须出现 'self.algo == "lora"' 守卫（带或不带空格）
        # 且必须不在跟 'tlora' 相关的条件里
        ctx_start = max(0, m.start() - 250)
        ctx = src[ctx_start:m.start()]
        has_lora_guard = bool(re.search(r'self\.algo\s*==\s*["\']lora["\']', ctx))
        mentions_tlora_nearby = "tlora" in ctx.lower()
        assert has_lora_guard and not mentions_tlora_nearby, (
            f"extra['bypass_mode'] 赋值 (位置 {m.start()}) 上方 250 字符内未见到"
            f" `self.algo == 'lora'` 独立守卫；可能让 tlora 也走 bypass 路径让"
            f" make_weight mask patch 静默失效。"
            f" 上下文：\n{ctx[-200:]}"
        )
