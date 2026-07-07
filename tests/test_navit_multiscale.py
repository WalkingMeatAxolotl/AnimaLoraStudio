"""NaViT 多尺度阶梯（navit_multiscale）单测。

覆盖：
  1) plan_multiscale_copy：只降不升（src ≤ 档 → None）、副本 token ≤ 档、16 对齐。
  2) **接线测试**：ImageDataset(native + ladder) 真的把原生大图展开成副本样本
     （带 ms_tokens_target、bucket_for_index 同步、副本尺寸 ≤ 档），原生条目不动。
  3) collate_fn_navit_pack 把 ms_weight 折进 per-image loss_weight（× 正则集 loss_weight）；
     无 reg 且无降权时不发 loss_weight（行为中立）。
  4) config：默认 off、multiscale 需 native、需非空 ladder、可开。

CPU-only、无 GPU / 无 VAE。
"""
from __future__ import annotations

import pytest
import torch

from training.dataset import (
    ImageDataset,
    collate_fn_navit_pack,
    plan_multiscale_copy,
)


# --------------------------------------------------------- plan_multiscale_copy
def test_no_upscale_returns_none():
    # 源 floor 后 token 数 ≤ 目标档 → 不产副本
    assert plan_multiscale_copy(512, 512, 4096) is None   # 32x32=1024 ≤ 4096
    assert plan_multiscale_copy(1024, 1024, 4096) is None  # 64x64=4096 == 4096（同档不产）


def test_downscaled_copy_within_target_and_aligned():
    copy = plan_multiscale_copy(2048, 2048, 4096)  # 128x128=16384 → 副本 ≤4096
    assert copy is not None
    assert copy.token_count <= 4096
    assert copy.width % 16 == 0 and copy.height % 16 == 0
    assert copy.was_downscaled is True


# --------------------------------------------------------- 展开接线测试
def _make_dataset_dir(tmp_path, sizes):
    from PIL import Image
    for i, (w, h) in enumerate(sizes):
        Image.new("RGB", (w, h), (0, i * 9 % 256, 0)).save(tmp_path / f"m{i}.png")
        (tmp_path / f"m{i}.txt").write_text("1girl", encoding="utf-8")
    return str(tmp_path)


def test_dataset_expands_multiscale_copies(tmp_path):
    # 一张 2048² 大图（原生 128x128=16384 token）+ 一张 512² 小图（32x32=1024 ≤ 4096，不展开）
    data_dir = _make_dataset_dir(tmp_path, [(2048, 2048), (512, 512)])
    ds = ImageDataset(
        data_dir, resolution=1024, bucket_mgr=None,
        native_resolution=True, native_token_budget=1_000_000,
        native_ms_token_ladder=[4096],
    )
    ms_samples = [s for s in ds.samples if int(s.get("ms_tokens_target", 0) or 0) > 0]
    # 只有大图被展开：恰好 1 个副本
    assert len(ms_samples) == 1
    assert ms_samples[0]["ms_tokens_target"] == 4096
    # samples 与 bucket_for_index 仍等长（同步 extend）
    assert len(ds.bucket_for_index) == len(ds.samples)
    # 副本尺寸的 token 数 ≤ 档
    tw, th = ds.bucket_for_index[-1]
    assert (tw // 16) * (th // 16) <= 4096
    # 原生大图条目尺寸不变（仍 2048x2048 floor）
    assert (2048, 2048) in [tuple(s) for s in ds.bucket_for_index if s is not None]


def test_target_size_for_ms_copy(tmp_path):
    data_dir = _make_dataset_dir(tmp_path, [(2048, 2048)])
    ds = ImageDataset(data_dir, resolution=1024, bucket_mgr=None,
                      native_resolution=True, native_token_budget=1_000_000,
                      native_ms_token_ladder=[4096])
    native = ds._target_size_for(2048, 2048)                       # 原生
    copy = ds._target_size_for(2048, 2048, ms_tokens_target=4096)  # 副本
    assert (copy[0] // 16) * (copy[1] // 16) <= 4096
    assert copy != native  # 副本比原生小


# --------------------------------------------------------- collate 权重折叠
def _fake_item(ms_weight=None, loss_weight=None, is_reg=False):
    item = {"latent": torch.zeros(4, 1, 8, 8), "caption": "", "image": "x"}
    if ms_weight is not None:
        item["ms_weight"] = ms_weight
    if loss_weight is not None:
        item["loss_weight"] = loss_weight
        item["is_reg"] = is_reg
    return item


def test_collate_folds_ms_weight():
    batch = [_fake_item(ms_weight=0.5), _fake_item()]  # 副本 0.5 / 原生（无 ms_weight）
    out = collate_fn_navit_pack(batch)
    assert "loss_weight" in out
    assert torch.allclose(out["loss_weight"], torch.tensor([0.5, 1.0]))


def test_collate_composes_reg_and_ms():
    # 正则集副本：reg loss_weight 0.7 × ms_weight 0.5 = 0.35
    batch = [_fake_item(ms_weight=0.5, loss_weight=0.7, is_reg=True), _fake_item(loss_weight=1.0)]
    out = collate_fn_navit_pack(batch)
    assert torch.allclose(out["loss_weight"], torch.tensor([0.35, 1.0]))


def test_collate_neutral_without_weights():
    # 无 reg、无 ms 降权 → 不发 loss_weight（与改动前逐字节等价）
    out = collate_fn_navit_pack([_fake_item(), _fake_item()])
    assert "loss_weight" not in out


# --------------------------------------------------------- config
def test_config_default_off():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.navit_multiscale is False
    assert cfg.navit_multiscale_loss_weight == 1.0


def test_config_multiscale_requires_native():
    from studio.domain import TrainingConfig
    with pytest.raises(ValueError):
        TrainingConfig(navit_packing=True, cache_latents=True, navit_token_budget=16384,
                       navit_multiscale=True, navit_multiscale_token_ladder="4096")  # 缺 native


def test_config_multiscale_requires_ladder():
    from studio.domain import TrainingConfig
    with pytest.raises(ValueError):
        TrainingConfig(navit_packing=True, cache_latents=True, navit_token_budget=16384,
                       navit_native_resolution=True, navit_multiscale=True)  # 缺 ladder


def test_config_multiscale_can_enable():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig(
        navit_packing=True, cache_latents=True, navit_token_budget=16384,
        navit_native_resolution=True, navit_multiscale=True,
        navit_multiscale_token_ladder="4096", navit_multiscale_loss_weight=0.5,
    )
    assert cfg.navit_multiscale is True
    assert cfg.navit_multiscale_loss_weight == 0.5
