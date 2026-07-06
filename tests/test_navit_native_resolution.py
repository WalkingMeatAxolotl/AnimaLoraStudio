"""NaViT 原生定尺寸（navit_native_resolution）单测。

覆盖：
  1) plan_native_fit_image(floor)：零 padding、16px 整倍数、尺寸 ≤ 源、
     token 数与 latent patch 网格一致。
  2) 原生 floor 定尺寸 + 预算打包的数据链路不变量：异构 token 数能被
     pack_indices_by_budget 在预算内无重叠覆盖。
  3) config：navit_native_resolution 默认 False 且能经 TrainingConfig 正确开启。

注：本地版的 scan_max_image_side 测试引用 anima_train（上游不存在），已剪除。
"""
from __future__ import annotations

from training.dataset import pack_indices_by_budget, plan_native_fit_image


# 一组异构原生尺寸（含非 16 整倍数的）
_SIZES = [(1000, 1500), (1536, 512), (777, 777), (2048, 768), (640, 1664)]


def test_floor_alignment_invariants():
    for w, h in _SIZES:
        plan = plan_native_fit_image(w, h, align_mode="floor", max_tokens=10 ** 9)
        # 16px 整倍数（VAE 下采样 8 × patch 2）
        assert plan.width % 16 == 0
        assert plan.height % 16 == 0
        # floor 只会裁小、不放大
        assert plan.width <= w
        assert plan.height <= h
        # 每边裁掉的像素 < 16（仅去掉不足一个对齐单元的余数）
        assert w - plan.width < 16
        assert h - plan.height < 16
        # 零 gray padding（navit 缓存路径无 mask 的前提）：floor 下规划尺寸 ≤ 源
        assert min(plan.source_width, plan.width) == plan.width
        assert min(plan.source_height, plan.height) == plan.height
        # token 数与 latent patch 网格一致
        assert plan.token_count == (plan.width // 16) * (plan.height // 16)


def test_heterogeneous_packs_within_budget_and_cover():
    counts = [
        plan_native_fit_image(w, h, align_mode="floor", max_tokens=10 ** 9).token_count
        for (w, h) in _SIZES
    ]
    budget = max(counts) * 2  # 保证每图都能进包、且能拼多张
    packs = pack_indices_by_budget(counts, budget, list(range(len(counts))))
    flat = sorted(i for p in packs for i in p)
    assert flat == list(range(len(counts)))
    for p in packs:
        assert sum(counts[i] for i in p) <= budget


def test_pad_mode_ceil_aligns():
    """pad（ceil）模式向上对齐，保留所有源像素。"""
    plan = plan_native_fit_image(1000, 1500, align_mode="pad", max_tokens=10 ** 9)
    assert plan.width >= 1000
    assert plan.height >= 1500
    assert plan.width % 16 == 0
    assert plan.height % 16 == 0
    assert plan.was_padded is True


def test_over_budget_raises():
    import pytest
    with pytest.raises(ValueError):
        plan_native_fit_image(4096, 4096, max_tokens=100)


# --------------------------------------------------------- config switch


def test_config_default_off():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.navit_native_resolution is False


def test_config_can_enable():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig(navit_native_resolution=True)
    assert cfg.navit_native_resolution is True
