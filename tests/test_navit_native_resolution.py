"""NaViT 原生定尺寸（navit_native_resolution）单测。

覆盖：
  1) plan_native_fit_image：floor 对齐 16px、零 padding、token 数 = 网格积。
  2) 超预算 downscale：等比缩到 ≤ token 预算、≤ RoPE 单边上限，仍 16 对齐。
  3) over_budget=fail：超限直接 raise。
  4) **接线测试（关键）**：ImageDataset(native_resolution=True) 的定尺寸真的走原生 floor-16，
     与 ARB 桶路径产出不同尺寸——这正是上一版 PR 缺失、导致"参数是空的"的那条。
  5) config：TrainingConfig 默认 navit_native_resolution=False、可开、native 需 navit_packing。

CPU-only、无 GPU / 无 VAE：CI（Linux 无 GPU）可跑。
"""
from __future__ import annotations

import pytest

from training.dataset import (
    BucketManager,
    ImageDataset,
    NativeFitImagePlan,
    plan_native_fit_image,
)


# 一组异构原生尺寸（含非 16 整倍数的）
_SIZES = [(1000, 1500), (1536, 512), (777, 777), (2048, 768), (640, 1664)]


# --------------------------------------------------------- plan：floor 不变量
def test_floor_alignment_invariants():
    for w, h in _SIZES:
        plan = plan_native_fit_image(w, h, align=16)  # 预算不限
        assert isinstance(plan, NativeFitImagePlan)
        assert plan.width % 16 == 0 and plan.height % 16 == 0
        assert plan.width <= w and plan.height <= h          # floor 只裁不放大
        assert w - plan.width < 16 and h - plan.height < 16  # 每边只裁掉 <16 的余数
        assert plan.token_count == (plan.width // 16) * (plan.height // 16)
        assert plan.was_downscaled is False


# --------------------------------------------------------- plan：超预算 downscale
def test_over_budget_downscale_fits_and_aligned():
    # 1000x1500 floor→992x1488 = 62x93 = 5766 tokens；预算 1024 → 必须 downscale
    budget = 1024
    plan = plan_native_fit_image(1000, 1500, max_tokens=budget, over_budget="downscale")
    assert plan.was_downscaled is True
    assert plan.width % 16 == 0 and plan.height % 16 == 0
    assert plan.token_count <= budget
    # 宽高比大致保持（缩放 + floor 的偏差不超过一个对齐单元的比例）
    assert plan.height > plan.width  # 竖图仍竖


def test_over_budget_downscale_extreme_aspect_within_budget():
    # 极端长宽比：某轴被 max(1,·) 顶起时应把另一轴压回，仍不超预算
    budget = 64
    plan = plan_native_fit_image(4096, 128, max_tokens=budget, over_budget="downscale")
    assert plan.token_count <= budget
    assert plan.width % 16 == 0 and plan.height % 16 == 0


def test_rope_side_cap_downscales():
    # 极扁图单边超 RoPE 上限：应被压到 ≤ max_side_tokens
    max_side = 32
    plan = plan_native_fit_image(
        8000, 512, max_tokens=0, max_side_tokens=max_side, over_budget="downscale"
    )
    assert plan.token_w <= max_side and plan.token_h <= max_side


def test_over_budget_fail_raises():
    with pytest.raises(ValueError):
        plan_native_fit_image(4096, 4096, max_tokens=1024, over_budget="fail")


# --------------------------------------------------------- 接线测试（关键）
def _make_dataset_dir(tmp_path, sizes):
    """在 tmp_path 造若干带 .txt caption 的 PNG，返回目录路径。"""
    from PIL import Image
    for i, (w, h) in enumerate(sizes):
        Image.new("RGB", (w, h), (i * 7 % 256, 0, 0)).save(tmp_path / f"img{i}.png")
        (tmp_path / f"img{i}.txt").write_text("1girl, solo", encoding="utf-8")
    return str(tmp_path)


def test_dataset_native_sizing_bypasses_buckets(tmp_path):
    """native_resolution=True 时定尺寸走原生 floor-16，且与 ARB 桶路径尺寸不同。"""
    data_dir = _make_dataset_dir(tmp_path, [(1000, 1500), (777, 777)])

    native = ImageDataset(data_dir, resolution=1024, bucket_mgr=None,
                          native_resolution=True, native_token_budget=1_000_000)
    # 直接问定尺寸口径：原生 floor-16
    assert native._target_size_for(1000, 1500) == (992, 1488)   # 1000//16*16, 1500//16*16
    assert native._target_size_for(777, 777) == (768, 768)
    # 预扫的 bucket_for_index 也应是原生尺寸（非 None、16 对齐、≤源）
    for size in native.bucket_for_index:
        assert size is not None
        tw, th = size
        assert tw % 16 == 0 and th % 16 == 0

    # 对照：ARB 桶路径（native 关）对同样的图给出桶尺寸，与原生不同
    bucketed = ImageDataset(data_dir, resolution=1024,
                            bucket_mgr=BucketManager(1024, aspect_ratio_limit=2.0))
    assert bucketed.native_resolution is False
    assert bucketed._target_size_for(1000, 1500) != native._target_size_for(1000, 1500)


def test_dataset_native_downscale_wiring(tmp_path):
    """native + 小 token 预算：ImageDataset 定尺寸真的把超预算图 downscale 到 fit。"""
    data_dir = _make_dataset_dir(tmp_path, [(2048, 2048)])  # floor→128x128 token=16384
    budget = 4096
    ds = ImageDataset(data_dir, resolution=1024, bucket_mgr=None,
                      native_resolution=True, native_token_budget=budget,
                      native_over_budget="downscale")
    tw, th = ds._target_size_for(2048, 2048)
    assert (tw // 16) * (th // 16) <= budget


# --------------------------------------------------------- config switch
def test_config_default_off():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.navit_native_resolution is False
    assert cfg.navit_native_over_budget == "downscale"


def test_config_can_enable_with_packing():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig(
        navit_packing=True, cache_latents=True, navit_token_budget=16384,
        navit_native_resolution=True,
    )
    assert cfg.navit_native_resolution is True


def test_config_native_requires_packing():
    from studio.domain import TrainingConfig
    with pytest.raises(ValueError):
        TrainingConfig(navit_native_resolution=True)  # navit_packing 默认 False → 拒
