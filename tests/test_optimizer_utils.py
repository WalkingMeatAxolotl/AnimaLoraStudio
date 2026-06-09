"""Optimizer utils 测试 — 覆盖 PPSF 接入 + Automagic v1/v2。

- optimizer_eval_mode context manager 对 PPSF / 非 PPSF 行为
- create_prodigy_plus_schedulefree 工厂在依赖缺失时友好报错
- 工厂 lr 强制 1.0 + betas 默认覆盖逻辑
- Automagic v1: step、bf16 Kahan (fp32 shift)、state_dict roundtrip
- Automagic v2: fused backward hook、scalar lr
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
    Automagic2,
    create_automagic,
    create_automagic_v2,
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


# ---------------------------------------------------------------------------
# Automagic v1
# ---------------------------------------------------------------------------


def test_create_automagic_optimizer_updates_parameters() -> None:
    """Automagic v1 基本 step 能跑通 — 参数发生变化。"""
    torch.manual_seed(42)
    model = nn.Linear(4, 4, bias=False)
    p_before = model.weight.data.clone()

    optim = create_automagic(model.parameters(), lr=1e-4)
    # 模拟 5 步训练
    for _ in range(5):
        out = model(torch.randn(2, 4))
        loss = out.sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    assert not torch.allclose(model.weight.data, p_before), "5 步后参数应该有变化"


def test_automagic_bf16_shift_is_fp32() -> None:
    """核心修复验证：bf16 参数的 Kahan shift buffer 必须是 fp32。"""
    p = nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    optim = Automagic([p], lr=1e-4)

    # 手动触发一步让 state 初始化
    p.grad = torch.randn_like(p)
    optim.step()

    state = optim.state[p]
    assert "shift" in state, "bf16 参数应该创建 shift buffer"
    assert state["shift"].dtype == torch.float32, (
        f"shift 应为 fp32，实际 {state['shift'].dtype}"
    )


def test_automagic_state_dict_roundtrip() -> None:
    """lr_mask (Auto8bitTensor) 序列化/反序列化后数值不丢失。"""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(16, 16))
    optim = Automagic([p], lr=1e-4)

    # 跑几步让 lr_mask 积累非零值
    for _ in range(10):
        p.grad = torch.randn_like(p)
        optim.step()
        p.grad = None

    sd = optim.state_dict()

    # 新建实例并 load
    p2 = nn.Parameter(torch.randn(16, 16))
    optim2 = Automagic([p2], lr=1e-4)
    optim2.load_state_dict(sd)

    # 比较 lr_mask 数值
    orig_lr = optim.state[p]["lr_mask"].dequantize()
    loaded_lr = optim2.state[p2]["lr_mask"].dequantize()
    assert torch.allclose(orig_lr, loaded_lr, atol=1e-6), "lr_mask roundtrip 应保持数值一致"


# ---------------------------------------------------------------------------
# Automagic v2
# ---------------------------------------------------------------------------


def test_automagic_v2_scalar_lr_updates() -> None:
    """Automagic v2 使用 fused backward hook，验证参数确实更新。"""
    torch.manual_seed(7)
    model = nn.Linear(8, 4, bias=False)
    p_before = model.weight.data.clone()

    optim = create_automagic_v2(model.parameters(), lr=1e-3)

    # v2 的 hook 在 backward 时自动更新参数
    for _ in range(5):
        out = model(torch.randn(2, 8))
        loss = out.sum()
        loss.backward()
        # v2 不需要手动 step（hook 内完成），但调用也无害
        optim.zero_grad()

    assert not torch.allclose(model.weight.data, p_before), "v2 backward hook 应导致参数变化"


def test_automagic_v2_get_avg_learning_rate() -> None:
    """v2 实例应暴露 get_avg_learning_rate 方法。"""
    model = nn.Linear(4, 4)
    optim = create_automagic_v2(model.parameters(), lr=1e-3)
    assert hasattr(optim, "get_avg_learning_rate")
    avg_lr = optim.get_avg_learning_rate()
    assert isinstance(avg_lr, float) or isinstance(avg_lr, torch.Tensor)


# ---------------------------------------------------------------------------
# get_optimizer_monitor_metrics — Automagic duck typing
# ---------------------------------------------------------------------------


def test_automagic_monitor_metrics_uses_get_avg_learning_rate() -> None:
    """get_optimizer_monitor_metrics 优先走 get_avg_learning_rate 鸭子类型。"""
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optim = create_automagic(model.parameters(), lr=1e-4)

    # 跑几步让 lr 有值
    for _ in range(3):
        out = model(torch.randn(2, 4))
        out.sum().backward()
        optim.step()
        optim.zero_grad()

    metrics = get_optimizer_monitor_metrics(optim)
    assert "lr" in metrics
    assert "actual_lr" in metrics
    assert metrics["lr"] > 0


# ---------------------------------------------------------------------------
# create_optimizer 工厂分派
# ---------------------------------------------------------------------------


def test_create_optimizer_dispatches_automagic() -> None:
    """create_optimizer(optimizer_type='automagic') 返回 Automagic 实例。"""
    model = nn.Linear(4, 4)
    optim = create_optimizer("automagic", model.parameters(), learning_rate=1e-4)
    assert isinstance(optim, Automagic)


def test_create_optimizer_dispatches_automagic_v2() -> None:
    """create_optimizer(optimizer_type='automagic_v2') 返回 Automagic2 实例。"""
    model = nn.Linear(4, 4)
    optim = create_optimizer("automagic_v2", model.parameters(), learning_rate=1e-3)
    assert isinstance(optim, Automagic2)
