"""NaViT 多尺度阶梯（navit_multiscale）单测。

覆盖：
  1) plan_multiscale_copy：只降不升采样、16px 整倍数、token 数 ≤ 目标档、
     等比（长宽比保持）、极端长宽比下不超预算。
  2) CachedLatentDataset._get_npz_path：副本 sidecar 命名（.ms<T>.npz）与原生互不覆盖。
  3) collate_fn_navit_pack：逐图 ms 标志透传。
  4) config：默认关闭、TrainingConfig 可开启。

注：本地版的 ImageDataset 展开测试引用 FiT 专属参数（fit_packed / fit_max_tokens /
fit_align_mode / navit_ms_token_ladder），上游无 FiT，已剪除。
"""
from __future__ import annotations

import pytest

from training.dataset import (
    collate_fn_navit_pack,
    plan_multiscale_copy,
    plan_native_fit_image,
)


# ----------------------------------------------------- plan_multiscale_copy


def test_basic_downscale_invariants():
    # 用户场景：2814x4456 原生 ≈ 48650 token，缩到 ≤4096 token 档
    plan = plan_multiscale_copy(2814, 4456, 4096)
    assert plan is not None
    assert plan.width % 16 == 0
    assert plan.height % 16 == 0
    assert plan.token_count <= 4096
    assert plan.token_count == (plan.width // 16) * (plan.height // 16)
    # 等比：副本长宽比与源一致（16px 量化容差内）
    src_ar = 2814 / 4456
    ms_ar = plan.width / plan.height
    assert abs(ms_ar - src_ar) / src_ar < 0.05
    # 有效区 = 整张（navit 缓存路径 mask 恒全 1 的前提）
    assert plan.source_width == plan.width
    assert plan.source_height == plan.height
    assert plan.was_resized is True
    assert plan.was_padded is False


def test_never_upscales_or_duplicates_native():
    # 源 token 数 == 目标档 → 不产出副本（避免与原生重复）
    assert plan_multiscale_copy(1024, 1024, 4096) is None
    # 源比目标档还小 → 绝不上采样
    assert plan_multiscale_copy(512, 512, 4096) is None
    # 非法输入
    assert plan_multiscale_copy(0, 512, 4096) is None
    assert plan_multiscale_copy(512, 512, 0) is None


def test_strictly_larger_source_gets_copy():
    # 1040x1040 → floor 后 65*65=4225 > 4096 → 应产出 ≤4096 的副本
    plan = plan_multiscale_copy(1040, 1040, 4096)
    assert plan is not None
    assert plan.token_count <= 4096


def test_extreme_aspect_ratio_stays_within_budget():
    # 极扁图：8000x160 → 源 500*10=5000 token > 目标 4096
    plan = plan_multiscale_copy(8000, 160, 4096)
    assert plan is not None
    assert plan.token_count <= 4096
    assert plan.width % 16 == 0
    assert plan.height % 16 == 0
    assert plan.token_h >= 1
    assert plan.token_w >= 1


def test_ladder_tokens_below_native():
    native = plan_native_fit_image(2814, 4456, align_mode="floor", max_tokens=10 ** 9)
    plan = plan_multiscale_copy(2814, 4456, 16384)
    assert plan.token_count < native.token_count


# --------------------------------------------------- npz sidecar naming


def test_ms_copy_gets_own_sidecar():
    from training.dataset import CachedLatentDataset
    p_native = CachedLatentDataset._get_npz_path(None, "D:/ds/pic.png")
    p_ms = CachedLatentDataset._get_npz_path(None, "D:/ds/pic.png", ms_tokens=4096)
    assert p_native.name == "pic.npz"
    assert p_ms.name == "pic.ms4096.npz"
    assert p_native != p_ms


def test_bare_path_backward_compatible():
    from training.dataset import CachedLatentDataset
    p = CachedLatentDataset._get_npz_path(None, "D:/ds/pic.png")
    assert p.name == "pic.npz"


# --------------------------------------------- collate ms flags passthrough


def test_collate_flags_passthrough():
    torch = pytest.importorskip("torch")
    lat = torch.zeros(16, 1, 8, 8)
    batch = [
        {"latent": lat, "caption": "a", "image": "a.png", "ms_tokens_target": 0},
        {"latent": lat, "caption": "b", "image": "b.png", "ms_tokens_target": 4096},
        {"latent": lat, "caption": "c", "image": "c.png"},  # 旧条目无键 → False
    ]
    out = collate_fn_navit_pack(batch)
    assert out["navit_ms_flags"] == [False, True, False]
    assert len(out["navit_latents"]) == 3


# --------------------------------------------------------- config switch


def test_config_default_off():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.navit_multiscale is False
    assert cfg.navit_multiscale_token_ladder == "4096"
    assert cfg.navit_multiscale_loss_weight == 1.0


def test_config_can_enable():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig(
        navit_multiscale=True,
        navit_multiscale_token_ladder="4096,16384",
        navit_multiscale_loss_weight=0.5,
    )
    assert cfg.navit_multiscale is True
    assert cfg.navit_multiscale_token_ladder == "4096,16384"
    assert cfg.navit_multiscale_loss_weight == 0.5
