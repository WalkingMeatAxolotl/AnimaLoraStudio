"""compute_navit_pack_estimate：NaViT 打包模式的 steps/epoch 预估数据源。

打包走真 NavitPackBatchSampler（非算法拷贝），扫描规则与 compute_bucket_histogram
同源。回归背景：训练页步数预估用「样本 ÷ batch_size」在 navit 下失真（batch_size
不参与分批），1536px 数据集预估 2520 实际 5040。
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _img(d: Path, names, size=(64, 64), caption=True) -> None:
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", size).save(d / f"{n}.png")
        if caption:
            (d / f"{n}.txt").write_text("1girl", encoding="utf-8")


def test_native_one_image_per_pack(tmp_path: Path) -> None:
    # 本 bug 场景：单图 token > budget/2 → 任何两图都装不下 → 包数 = 样本数
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["a", "b", "c"])  # 64×64 → 4×4 = 16 token
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=24,
    )
    assert out["samples"] == 3
    assert out["packs_per_epoch"] == 3
    assert out["token_min"] == out["token_max"] == 16


def test_native_packs_fill_budget(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["a", "b", "c"])  # 16 token each
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=48,
    )
    assert out["packs_per_epoch"] == 1
    assert out["avg_images_per_pack"] == 3.0
    assert out["sizes"] == [{"w": 64, "h": 64, "count": 3}]


def test_repeat_expands_samples(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "5_data", ["a", "b"])
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=16,
    )
    assert out["samples"] == 10  # 2 图 × repeat 5
    assert out["packs_per_epoch"] == 10  # budget 恰好单图 → 每包 1


def test_arb_bucket_tokens_without_native(tmp_path: Path) -> None:
    # 非 native：token 按 ARB 桶尺寸 (w//16)*(h//16)，与 latent 形状推导同口径
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["a", "b"], size=(1024, 1024))
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=False, token_budget=8192,
    )
    # 1024×1024 桶 → 64×64 = 4096 token；8192 预算装 2 张
    assert out["token_min"] == out["token_max"] == 4096
    assert out["samples"] == 2
    assert out["packs_per_epoch"] == 1
    assert out["sizes"] == []  # 非 native 不出尺寸直方图（桶直方图已有）


def test_native_downscale_over_budget(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["big"], size=(256, 256))  # 16×16 = 256 token
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=64,
    )
    assert out["downscaled"] == 1
    assert out["token_max"] <= 64
    assert out["packs_per_epoch"] == 1


def test_uncaptioned_images_skipped(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["a"])
    _img(tmp_path / "1_data", ["b"], caption=False)
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=16,
    )
    assert out["samples"] == 1


def test_reg_dir_joins_the_pool(tmp_path: Path) -> None:
    # reg 集与 main 拼进同一打包池（MergedDataset 语义）
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "train" / "1_data", ["a"])
    _img(tmp_path / "reg" / "1_prior", ["r1", "r2"])
    out = compute_navit_pack_estimate(
        [tmp_path / "train", tmp_path / "reg"], [1024],
        native_resolution=True, token_budget=16,
    )
    assert out["samples"] == 3
    assert out["packs_per_epoch"] == 3


def test_empty_dataset(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    out = compute_navit_pack_estimate(
        [tmp_path / "nope"], [1024], native_resolution=True, token_budget=16,
    )
    assert out["packs_per_epoch"] == 0
    assert out["samples"] == 0
