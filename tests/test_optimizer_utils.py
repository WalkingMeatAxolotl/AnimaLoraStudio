"""Optimizer utils 测试 — 重点覆盖 PPSF 接入。

- optimizer_eval_mode context manager 对 PPSF / 非 PPSF 行为
- create_prodigy_plus_schedulefree 工厂在依赖缺失时友好报错
- 工厂 lr 强制 1.0 + betas 默认覆盖逻辑
"""
from __future__ import annotations

import builtins
import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from utils.optimizer_utils import (
    Automagic,
    Lion,
    create_optimizer,
    create_prodigy_plus_schedulefree,
    get_optimizer_monitor_metrics,
    optimizer_eval_mode,
)


# ---------------------------------------------------------------------------
# optimizer_eval_mode
# ---------------------------------------------------------------------------


def test_eval_mode_noop_for_plain_adamw() -> None:
    """AdamW 没有 .eval/.train 方法，ctx 静默 no-op，不抛错。"""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # AdamW 没有 .train / .eval 方法 — ctx 应该静默走过
    with optimizer_eval_mode(optim):
        # 还能正常调用 — 没有 side effect
        assert optim.param_groups[0]["lr"] == 1e-3


def test_eval_mode_calls_eval_and_train_on_schedulefree_like() -> None:
    """PPSF-like 优化器 ctx 进入调 .eval()，退出调 .train()。"""
    fake_opt = MagicMock(spec=["eval", "train"])
    with optimizer_eval_mode(fake_opt):
        fake_opt.eval.assert_called_once_with()
        fake_opt.train.assert_not_called()
    fake_opt.train.assert_called_once_with()


def test_eval_mode_restores_train_on_exception() -> None:
    """ctx 内部抛异常时也要保证切回 .train() —— 否则训练权重永远停在 averaged 状态。"""
    fake_opt = MagicMock(spec=["eval", "train"])

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with optimizer_eval_mode(fake_opt):
            raise Boom()

    fake_opt.eval.assert_called_once_with()
    fake_opt.train.assert_called_once_with()


def test_eval_mode_skips_if_only_partial_methods() -> None:
    """只有 .eval 没有 .train（或反过来）的优化器 — 视为非 PPSF，no-op。
    防止误调单边方法把内部状态搞坏。"""
    fake_opt = MagicMock(spec=["eval"])  # 只有 eval 没 train
    with optimizer_eval_mode(fake_opt):
        pass
    fake_opt.eval.assert_not_called()


# ---------------------------------------------------------------------------
# get_optimizer_monitor_metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lion
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Automagic
# ---------------------------------------------------------------------------


def test_create_automagic_optimizer_updates_parameters() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    # 起点 lr 给到 max_lr，单步变化才能跨过 torch.allclose 默认 tolerance；
    # 默认 lr=1e-6 单步 ~1e-7 变化在 atol=1e-8+rtol=1e-5 内会被判 "close"。
    optim = create_optimizer(
        "automagic",
        model.parameters(),
        learning_rate=1e-3,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=1e-5,
        weight_decay=0.0,
    )

    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optim.step()

    assert isinstance(optim, Automagic)
    assert not torch.allclose(model.weight, torch.ones_like(model.weight))
    assert "lr_mask" in optim.state[model.weight]
    assert optim.get_avg_learning_rate() >= optim.min_lr


def test_automagic_monitor_metrics_use_dynamic_lr() -> None:
    model = nn.Linear(2, 1)
    optim = create_optimizer("automagic", model.parameters(), learning_rate=1e-6)
    for p in model.parameters():
        optim.initialize_state(p)
    metrics = get_optimizer_monitor_metrics(optim)
    assert metrics["lr"] == pytest.approx(1e-6)
    assert metrics["actual_lr"] == pytest.approx(1e-6)


def test_automagic_sign_agreement_increases_lr() -> None:
    """Same-sign gradients across two steps → sign_agreement>0 → lr_mask bumps up."""
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optim = create_optimizer(
        "automagic",
        model.parameters(),
        learning_rate=1e-5,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=5e-5,  # 大于 8-bit 量化粒度，保证可观测
        weight_decay=0.0,
    )

    # 第一步：建立 last_polarity
    model(torch.ones(1, 2)).sum().backward()
    optim.step()
    lr_after_step1 = optim.get_avg_learning_rate()
    optim.zero_grad()

    # 第二步：同号梯度 → sign_agreement>0 → lr 应升
    model(torch.ones(1, 2)).sum().backward()
    optim.step()
    lr_after_step2 = optim.get_avg_learning_rate()

    assert lr_after_step2 > lr_after_step1, (
        f"sign-agreement 同向时 lr_mask 应上调，但 {lr_after_step1} → {lr_after_step2}"
    )


def test_automagic_sign_flip_decreases_lr() -> None:
    """Sign-flipped gradients between steps → sign_agreement<0 → lr_mask bumps down."""
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optim = create_optimizer(
        "automagic",
        model.parameters(),
        learning_rate=5e-4,  # 起点高一点，留下降空间
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=5e-5,
        weight_decay=0.0,
    )

    # 第一步：正梯度
    model(torch.ones(1, 2)).sum().backward()
    optim.step()
    lr_after_step1 = optim.get_avg_learning_rate()
    optim.zero_grad()

    # 第二步：负梯度（翻 sign）
    (-model(torch.ones(1, 2)).sum()).backward()
    optim.step()
    lr_after_step2 = optim.get_avg_learning_rate()

    assert lr_after_step2 < lr_after_step1, (
        f"sign 翻转时 lr_mask 应下调，但 {lr_after_step1} → {lr_after_step2}"
    )


def test_automagic_does_not_register_grad_accum_hook() -> None:
    """对齐上游 diffusion-pipe：默认不开 stochastic-rounding grad accum hook，
    避免和 AMP GradScaler.unscale_ / clip_grad_norm_ 静默冲突。"""
    model = nn.Linear(2, 2).to(torch.bfloat16)
    optim = create_optimizer("automagic", model.parameters(), learning_rate=1e-6)

    assert optim.is_stochastic_rounding_accumulation is False, (
        "Automagic must not enable post-accumulate-grad hook by default"
    )
    # 确认 backward 后 p.grad 不被 hook 删掉
    out = model(torch.ones(1, 2, dtype=torch.bfloat16)).sum()
    out.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, (
                "p.grad was unexpectedly deleted; hook must not be active"
            )


def test_automagic_bf16_step_runs_finite() -> None:
    """bf16 训练单步：Kahan summation 路径 + lr_mask 正常更新，loss/p 保持有限。"""
    model = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
    optim = create_optimizer(
        "automagic",
        model.parameters(),
        learning_rate=1e-5,
        min_lr=1e-7,
        max_lr=1e-3,
        weight_decay=0.0,
    )
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    loss = model(x).sum()
    loss.backward()
    optim.step()

    for p in model.parameters():
        assert torch.isfinite(p).all(), "bf16 Kahan path produced non-finite p"
        # Kahan 路径必须建出 shift state
        assert "shift" in optim.state[p]
        assert optim.state[p]["shift"].dtype == torch.bfloat16


def test_automagic_state_dict_roundtrip() -> None:
    """state_dict / load_state_dict 保留 lr_mask（Auto8bitTensor）+ 数值一致。"""
    model1 = nn.Linear(4, 4, bias=False)
    optim1 = create_optimizer(
        "automagic", model1.parameters(),
        learning_rate=1e-5, min_lr=1e-7, max_lr=1e-3, lr_bump=5e-5,
    )
    # 跑两步建立 state
    for _ in range(2):
        model1(torch.randn(2, 4)).sum().backward()
        optim1.step()
        optim1.zero_grad()
    sd = optim1.state_dict()

    # 用第二个 optim 加载
    model2 = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.copy_(p1)
    optim2 = create_optimizer(
        "automagic", model2.parameters(),
        learning_rate=1e-5, min_lr=1e-7, max_lr=1e-3, lr_bump=5e-5,
    )
    optim2.load_state_dict(sd)

    # lr_mask 的 quantized + scale 都得对得上
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        s1 = optim1.state[p1]
        s2 = optim2.state[p2]
        assert "lr_mask" in s2
        assert torch.equal(s1["lr_mask"].quantized, s2["lr_mask"].quantized)
        assert s1["lr_mask"].scale == s2["lr_mask"].scale


def test_create_lion_optimizer_updates_parameters() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optim = create_optimizer(
        "lion",
        model.parameters(),
        learning_rate=0.1,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )

    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optim.step()

    assert isinstance(optim, Lion)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 0.9))
    assert "exp_avg" in optim.state[model.weight]


def test_create_lion_rejects_invalid_betas() -> None:
    model = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="Invalid beta1"):
        create_optimizer("lion", model.parameters(), learning_rate=1e-4, betas=(1.0, 0.99))


def test_monitor_metrics_uses_plain_lr_for_adamw() -> None:
    """AdamW-style optimizers keep the historical monitor lr unchanged."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    assert get_optimizer_monitor_metrics(optim) == {"lr": 1e-4}


def test_monitor_metrics_reports_prodigy_effective_lr_from_d() -> None:
    """Prodigy/PPSF expose base lr=1; monitor should show d-adjusted LR."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1.0)
    optim.param_groups[0]["d"] = 2e-4

    metrics = get_optimizer_monitor_metrics(optim)

    assert metrics["lr"] == 2e-4
    assert metrics["actual_lr"] == 2e-4
    assert metrics["base_lr"] == 1.0
    assert metrics["d"] == 2e-4


def test_monitor_metrics_uses_ppsf_effective_lr_multiplier() -> None:
    """PPSF v2 recommends logging d * effective_lr."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1.0)
    optim.param_groups[0]["d"] = 2e-4
    optim.param_groups[0]["effective_lr"] = 0.25

    metrics = get_optimizer_monitor_metrics(optim)

    assert metrics["lr"] == 5e-5
    assert metrics["actual_lr"] == 5e-5
    assert metrics["base_lr"] == 1.0
    assert metrics["effective_lr"] == 0.25


def test_monitor_metrics_uses_ppsf_shared_d_when_split_groups_mean() -> None:
    """PPSF split_groups_mean uses shared_d for the dynamic learning rate."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1.0)
    optim.param_groups[0]["d"] = 2e-4
    optim.param_groups[0]["shared_d"] = 5e-5
    optim.param_groups[0]["split_groups"] = True
    optim.param_groups[0]["split_groups_mean"] = True

    metrics = get_optimizer_monitor_metrics(optim)

    assert metrics["lr"] == 5e-5
    assert metrics["actual_lr"] == 5e-5
    assert metrics["d"] == 5e-5


# ---------------------------------------------------------------------------
# create_prodigy_plus_schedulefree
# ---------------------------------------------------------------------------


def _has_ppsf() -> bool:
    try:
        importlib.import_module("prodigyplus")
        return True
    except ImportError:
        return False


def test_create_ppsf_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """没装 PPSF 时报错信息要包含安装提示，而不是裸 ImportError。

    用 builtins.__import__ 强制让 `from prodigyplus import ...` 抛 ImportError，
    不依赖运行环境是否真装了 PPSF（CI / dev / 本地 venv 都能跑）。
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "prodigyplus" or name.startswith("prodigyplus."):
            raise ImportError("simulated: no prodigyplus")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "prodigyplus", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    model = nn.Linear(4, 4)
    with pytest.raises(ImportError, match="prodigy-plus-schedule-free"):
        create_prodigy_plus_schedulefree(model.parameters(), lr=1.0)


@pytest.mark.skipif(not _has_ppsf(), reason="PPSF 未安装")
def test_create_ppsf_forces_lr_to_one(caplog: pytest.LogCaptureFixture) -> None:
    """传非 1.0 的 lr 时强制覆盖并 WARN（PPSF 要求 lr=1.0）。"""
    import logging
    model = nn.Linear(4, 4)
    with caplog.at_level(logging.WARNING, logger="utils.optimizer_utils"):
        optim = create_prodigy_plus_schedulefree(model.parameters(), lr=1e-4)
    assert any(
        "lr=1.0" in r.getMessage() and r.levelno >= logging.WARNING
        for r in caplog.records
    ), f"expected WARNING with 'lr=1.0', got: {[r.getMessage() for r in caplog.records]}"
    assert optim.param_groups[0]["lr"] == 1.0


@pytest.mark.skipif(not _has_ppsf(), reason="PPSF 未安装")
def test_create_ppsf_overrides_pytorch_default_betas() -> None:
    """上层 create_optimizer 默认 betas=(0.9, 0.999) 时，工厂内部覆盖为 PPSF 推荐 (0.9, 0.99)。
    用户显式传别的值就尊重。"""
    model = nn.Linear(4, 4)
    # 不显式传 betas — 应被工厂覆盖
    optim = create_prodigy_plus_schedulefree(model.parameters(), lr=1.0, betas=(0.9, 0.999))
    assert optim.param_groups[0]["betas"] == (0.9, 0.99)

    # 显式传则尊重
    model2 = nn.Linear(4, 4)
    optim2 = create_prodigy_plus_schedulefree(model2.parameters(), lr=1.0, betas=(0.95, 0.98))
    assert optim2.param_groups[0]["betas"] == (0.95, 0.98)


@pytest.mark.skipif(not _has_ppsf(), reason="PPSF 未安装")
def test_create_ppsf_exposes_train_eval_methods() -> None:
    """实例化后必须有 .train / .eval 方法 — 否则 optimizer_eval_mode 永远 no-op。"""
    model = nn.Linear(4, 4)
    optim = create_prodigy_plus_schedulefree(model.parameters(), lr=1.0)
    assert hasattr(optim, "train") and callable(optim.train)
    assert hasattr(optim, "eval") and callable(optim.eval)
