"""loss_weighting 单元测试。

主要覆盖：
- detail_inv_t 默认 clamp 范围 [1, 5] —— 防 PR 把默认行为改坏
- detail_inv_t_min/max 自定义边界生效（PR：参数化原本写死的上下限）
- min > max 时 swap 后正常工作，不崩溃
- 新参数只影响 detail_inv_t，不影响 min_snr / cosmap / none

其他 scheme（min_snr 公式 / cosmap 公式 / weight_cap_ratio）不在本 PR 范围内重测。
"""
from __future__ import annotations

import torch

from training.loss_weighting import compute_loss_weight


# ---------------------------------------------------------------------------
# detail_inv_t 默认行为回归
# ---------------------------------------------------------------------------


def test_detail_inv_t_default_matches_legacy_clamp():
    """默认参数下，clamp 行为应等价于历史 hardcode [1, 5]：
        t=0.5  → 1/t=2     → 未 clamp → w=2
        t=0.1  → 1/t=10    → 触上界 → w=5
        t=0.9  → 1/t≈1.11  → 未 clamp
        t=0.999→ 1/t≈1.001 → 未 clamp（默认下界 1.0 在 t∈(0,1) 下几乎不生效）
    """
    t = torch.tensor([0.5, 0.1, 0.9, 0.999])
    w = compute_loss_weight(t, scheme="detail_inv_t")
    expected = torch.tensor([2.0, 5.0, 1.0 / 0.9, 1.0 / 0.999])
    torch.testing.assert_close(w, expected, rtol=1e-4, atol=1e-4)


def test_detail_inv_t_extreme_low_t_clamps_to_max():
    """t 趋近 0 时 1/t 爆炸，必须被 max 截住（默认 5）。"""
    t = torch.tensor([1e-5, 1e-3, 0.0])  # 0.0 会被 eps 处理
    w = compute_loss_weight(t, scheme="detail_inv_t")
    assert torch.all(w <= 5.0 + 1e-6)
    assert torch.all(w == 5.0)


# ---------------------------------------------------------------------------
# 自定义 min/max 生效
# ---------------------------------------------------------------------------


def test_detail_inv_t_custom_max_hazy_profile():
    """雾蒙蒙画风用 [1, 3]：t=0.1 应该 clamp 到 3 而不是 5。"""
    t = torch.tensor([0.5, 0.1, 0.9])
    w = compute_loss_weight(t, scheme="detail_inv_t", detail_inv_t_min=1.0, detail_inv_t_max=3.0)
    expected = torch.tensor([2.0, 3.0, 1.0 / 0.9])
    torch.testing.assert_close(w, expected, rtol=1e-4, atol=1e-4)


def test_detail_inv_t_custom_max_aggressive_profile():
    """激进风用 [1, 8]：t=0.1 应该 clamp 到 8。"""
    t = torch.tensor([0.5, 0.1, 0.05])
    w = compute_loss_weight(t, scheme="detail_inv_t", detail_inv_t_min=1.0, detail_inv_t_max=8.0)
    # t=0.05 → 1/t=20 → clamp 到 8
    expected = torch.tensor([2.0, 8.0, 8.0])
    torch.testing.assert_close(w, expected, rtol=1e-4, atol=1e-4)


def test_detail_inv_t_custom_min_lifts_high_t():
    """提高下限到 1.5：t=0.9 (1/t≈1.11) 应被抬到 1.5。"""
    t = torch.tensor([0.9, 0.999])
    w = compute_loss_weight(t, scheme="detail_inv_t", detail_inv_t_min=1.5, detail_inv_t_max=5.0)
    assert torch.all(w >= 1.5 - 1e-6)


# ---------------------------------------------------------------------------
# 边界 / 健壮性
# ---------------------------------------------------------------------------


def test_detail_inv_t_min_above_max_is_swapped():
    """误配 min > max 时函数内部 swap，不崩溃且产出有意义结果。"""
    t = torch.tensor([0.1, 0.5, 0.9])
    w_swap = compute_loss_weight(t, scheme="detail_inv_t", detail_inv_t_min=5.0, detail_inv_t_max=1.0)
    w_normal = compute_loss_weight(t, scheme="detail_inv_t", detail_inv_t_min=1.0, detail_inv_t_max=5.0)
    torch.testing.assert_close(w_swap, w_normal)


def test_detail_inv_t_min_equals_max_collapses_to_constant():
    """min == max 时 clamp 把所有权重压成一个常数。"""
    t = torch.tensor([0.05, 0.5, 0.95])
    w = compute_loss_weight(t, scheme="detail_inv_t", detail_inv_t_min=2.5, detail_inv_t_max=2.5)
    expected = torch.full_like(t, 2.5)
    torch.testing.assert_close(w, expected)


# ---------------------------------------------------------------------------
# 不影响其他 scheme
# ---------------------------------------------------------------------------


def test_detail_inv_t_bounds_do_not_affect_min_snr():
    t = torch.tensor([0.1, 0.5, 0.9])
    w_a = compute_loss_weight(t, scheme="min_snr", detail_inv_t_min=99.0, detail_inv_t_max=0.5)
    w_b = compute_loss_weight(t, scheme="min_snr")
    torch.testing.assert_close(w_a, w_b)


def test_detail_inv_t_bounds_do_not_affect_cosmap():
    t = torch.tensor([0.1, 0.5, 0.9])
    w_a = compute_loss_weight(t, scheme="cosmap", detail_inv_t_min=99.0, detail_inv_t_max=0.5)
    w_b = compute_loss_weight(t, scheme="cosmap")
    torch.testing.assert_close(w_a, w_b)


def test_detail_inv_t_bounds_do_not_affect_none():
    t = torch.tensor([0.1, 0.5, 0.9])
    w = compute_loss_weight(t, scheme="none", detail_inv_t_min=99.0, detail_inv_t_max=0.5)
    torch.testing.assert_close(w, torch.ones_like(t))
