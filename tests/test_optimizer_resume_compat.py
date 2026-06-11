"""Automagic v1/v2 + Lion 断点续训兼容性测试。

走真实 save_training_state → load_training_state 链路（torch.save/load 序列化），
覆盖三类优化器各自的 resume 风险点：

- Automagic v1：lr_mask (Auto8bitTensor) 序列化为 plain dict、load 后 int8/bool
  state 搬回 param device、bf16 Kahan shift dtype 稳定（PyTorch load 会把 float
  state cast 到 param dtype —— shift 与 param 同 dtype 所以 roundtrip 不漂移）。
- Automagic v2：scalar lr / 二阶矩在 load fixup 后恢复 fp32（PyTorch 在 bf16
  param 下会把它们降成 bf16）；fused backward hook 在 resume 后继续工作。
- Lion：标准 PyTorch state（exp_avg），无自定义钩子，验证数值保留 + 续步可跑。

注意：pause snapshot freeze（bootstrap）保证 UI 路径下 resume 不会换 optimizer
类型 / variant；跨 variant 手动 resume（CLI 改 yaml）不在支持范围，v2 加载 v1
state 会在首次 update 时 KeyError fail-fast 而非静默错误。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from utils.optimizer_utils import Automagic, Automagic2, Lion


class _StubInjector:
    """state.py 只要求 state_dict() / load_state_dict(sd, strict=False)。"""

    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"stub.weight": torch.zeros(1)}

    def load_state_dict(self, sd, strict=True):
        self.loaded = sd
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def _roundtrip(tmp_path: Path, optimizer, model_factory, optimizer_factory):
    """save → 新 model/optimizer → load，返回 (新 optimizer, 新 model)。"""
    from training.state import load_training_state, save_training_state

    ckpt = tmp_path / "state.pt"
    save_training_state(
        ckpt, _StubInjector(), optimizer, epoch=3, global_step=42,
        loss_history=[1.0, 0.5],
    )

    model2 = model_factory()
    optim2 = optimizer_factory(model2)
    epoch, global_step, loss_history, _monitor = load_training_state(
        ckpt, _StubInjector(), optim2,
    )
    assert (epoch, global_step) == (3, 42)
    assert loss_history == [1.0, 0.5]
    return optim2, model2


# ---------------------------------------------------------------------------
# Automagic v1
# ---------------------------------------------------------------------------


def test_automagic_v1_resume_roundtrip_fp32(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False)
    optim = Automagic(model.parameters(), lr=1e-5, lr_bump=5e-5)
    for _ in range(3):
        model(torch.randn(2, 8)).sum().backward()
        optim.step()
        optim.zero_grad()
    lr_before = optim.get_avg_learning_rate()
    p1 = next(iter(model.parameters()))
    mask_before = optim.state[p1]["lr_mask"].dequantize()

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False),
        lambda m: Automagic(m.parameters(), lr=1e-5, lr_bump=5e-5),
    )
    p2 = next(iter(model2.parameters()))
    st = optim2.state[p2]

    # lr_mask 数值精确保留（int8 quantized + scale 序列化为 plain dict）
    assert torch.allclose(st["lr_mask"].dequantize(), mask_before, atol=1e-9)
    assert optim2.get_avg_learning_rate() == pytest.approx(lr_before)
    # 续步可跑：lr 轨迹从恢复值继续而不是重置
    model2(torch.randn(2, 8)).sum().backward()
    optim2.step()
    assert torch.isfinite(p2).all()


def test_automagic_v1_resume_bf16_kahan_shift_stable(tmp_path: Path) -> None:
    """bf16 训练 resume：shift 与 param 同 dtype，roundtrip 后不漂移；
    bool last_polarity / int8 lr_mask 搬回 param device（CPU 下退化为 no-op，
    但 dtype 断言仍然有效）。"""
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    optim = Automagic(model.parameters(), lr=1e-5)
    model(torch.randn(2, 8, dtype=torch.bfloat16)).sum().backward()
    optim.step()
    optim.zero_grad()

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False).to(torch.bfloat16),
        lambda m: Automagic(m.parameters(), lr=1e-5),
    )
    p2 = next(iter(model2.parameters()))
    st = optim2.state[p2]
    assert st["shift"].dtype == p2.dtype, "Kahan shift dtype 在 resume 后必须稳定"
    assert st["lr_mask"].quantized.dtype == torch.int8
    # 续步 Kahan 路径可跑
    model2(torch.randn(2, 8, dtype=torch.bfloat16)).sum().backward()
    optim2.step()
    assert torch.isfinite(p2).all()


# ---------------------------------------------------------------------------
# Automagic v2
# ---------------------------------------------------------------------------


def test_automagic_v2_resume_scalar_lr_fp32_and_hook_alive(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False)
    optim = Automagic2(model.parameters(), lr=1e-6, lr_bump=1e-6)
    for _ in range(3):
        model(torch.randn(2, 8)).sum().backward()  # hook 内完成 update
        optim.zero_grad()
    p1 = next(iter(model.parameters()))
    lr_before = float(optim.state[p1]["lr"])
    step_before = optim.state[p1]["step"]

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False),
        lambda m: Automagic2(m.parameters(), lr=1e-6, lr_bump=1e-6),
    )
    p2 = next(iter(model2.parameters()))
    st = optim2.state[p2]
    assert st["lr"].dtype == torch.float32
    assert float(st["lr"]) == pytest.approx(lr_before)
    assert st["step"] == step_before

    # resume 后 fused hook 继续工作：backward 即更新参数并清 grad
    before = p2.detach().clone()
    model2(torch.randn(2, 8)).sum().backward()
    assert p2.grad is None, "fused hook 应在 backward 中消费 grad"
    assert not torch.equal(p2.detach(), before), "resume 后 hook 应继续更新参数"
    assert st["step"] == step_before + 1, "step 应从恢复值继续递增"


def test_automagic_v2_resume_bf16_second_moment_fp32(tmp_path: Path) -> None:
    """bf16 param 下 PyTorch load 会把 fp32 state 降成 bf16，
    Automagic2.load_state_dict fixup 必须恢复 fp32。"""
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    optim = Automagic2(model.parameters(), lr=1e-6)
    model(torch.randn(2, 8, dtype=torch.bfloat16)).sum().backward()
    optim.zero_grad()

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False).to(torch.bfloat16),
        lambda m: Automagic2(m.parameters(), lr=1e-6),
    )
    p2 = next(iter(model2.parameters()))
    st = optim2.state[p2]
    assert st["lr"].dtype == torch.float32
    assert st["exp_avg_sq_row"].dtype == torch.float32
    assert st["exp_avg_sq_col"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Lion
# ---------------------------------------------------------------------------


def test_lion_resume_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False)
    optim = Lion(model.parameters(), lr=1e-5)
    for _ in range(3):
        model(torch.randn(2, 8)).sum().backward()
        optim.step()
        optim.zero_grad()
    p1 = next(iter(model.parameters()))
    exp_avg_before = optim.state[p1]["exp_avg"].clone()

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False),
        lambda m: Lion(m.parameters(), lr=1e-5),
    )
    p2 = next(iter(model2.parameters()))
    assert torch.allclose(optim2.state[p2]["exp_avg"], exp_avg_before)
    assert optim2.param_groups[0]["lr"] == 1e-5
    # 续步可跑
    model2(torch.randn(2, 8)).sum().backward()
    optim2.step()
    assert torch.isfinite(p2).all()
