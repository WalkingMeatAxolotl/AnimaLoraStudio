"""_display_total_steps：monitor 进度条总步数的每-epoch 动态修正。

背景：navit 打包器每 epoch reshuffle 后包数会变，optimizer_phase 启动时算的
total_steps 是 epoch-0 快照 → 监控进度条分母全程陈旧、结束时 step ≠ total_steps。
修正 = 每 epoch 开始按「已走步数 + 当前包数 × 剩余 epochs」重估；非 navit 恒等。
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from runtime.training.loop import _display_total_steps  # noqa: E402


def test_constant_dl_len_is_identity() -> None:
    # 非 navit：dl_len 恒定 → 每个 epoch 的估计都等于启动时 total_steps
    spe = 63  # ceil(126 / (bs=2 → dl_len=63) / ga=1)
    total = spe * 40
    for epoch in range(40):
        est = _display_total_steps(
            global_step=epoch * spe, dl_len=63, grad_accum=1,
            total_epochs=40, epoch=epoch, max_steps=0,
        )
        assert est == total


def test_grad_accum_ceil_matches_optimizer_phase() -> None:
    # ceil 语义与 optimizer_phase 的 steps_per_epoch = ceil(dl_len/ga) 一致
    est = _display_total_steps(
        global_step=0, dl_len=10, grad_accum=4, total_epochs=3, epoch=0, max_steps=0,
    )
    assert est == 3 * 3  # ceil(10/4)=3

def test_navit_pack_drift_converges() -> None:
    # navit：epoch-0 估 126 包 × 3 epochs = 378；epoch-1 实际包数变 120
    # → 估计变 126 + 120×2 = 366；最后一个 epoch 精确。
    e0 = _display_total_steps(0, 126, 1, 3, 0, 0)
    assert e0 == 378
    e1 = _display_total_steps(126, 120, 1, 3, 1, 0)
    assert e1 == 126 + 120 * 2
    e2 = _display_total_steps(246, 124, 1, 3, 2, 0)
    assert e2 == 246 + 124  # 最后 epoch：已走 + 本 epoch 实际包数


def test_max_steps_caps_estimate() -> None:
    est = _display_total_steps(0, 126, 1, 40, 0, max_steps=1000)
    assert est == 1000


def test_no_len_or_no_epochs_returns_none() -> None:
    assert _display_total_steps(0, None, 1, 40, 0, 0) is None
    assert _display_total_steps(0, 126, 1, 0, 0, 0) is None
