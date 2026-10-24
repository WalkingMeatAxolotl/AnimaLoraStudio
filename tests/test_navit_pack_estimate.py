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


@pytest.mark.parametrize("prefer_json", [True, False])
def test_uncaptioned_images_included(tmp_path: Path, prefer_json: bool) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["a"])
    _img(tmp_path / "1_data", ["b"], caption=False)
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=16,
        prefer_json=prefer_json,
    )
    assert out["samples"] == 2


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


# ── 文本 token 计入（MOSS：文本与图像同在序列）────────────────────────────


def test_text_tokens_join_budget_with_real_captions(tmp_path: Path) -> None:
    """caption 经真 ImageDataset 解析；每张图 token = 图像 + 文本，与训练打包同口径。"""
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["a", "b", "c"])  # 64×64 → 16 图像 token，caption "1girl"
    image_only = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=48,
    )
    with_text = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=48,
        text_token_count=lambda caption: len(caption),  # "1girl" → 5
    )
    assert image_only["text_tokens"] is False and with_text["text_tokens"] is True
    assert with_text["downscaled"] == 0  # 预算 48 − 文本 5 ≥ 16，不触发 downscale
    assert (image_only["token_max"], with_text["token_max"]) == (16, 21)
    assert (image_only["packs_per_epoch"], with_text["packs_per_epoch"]) == (1, 2)


def test_native_reserves_text_and_reg_caption_override(tmp_path: Path) -> None:
    """native 先从预算扣掉该图文本 token（镜像 ImageDataset._native_image_budget）；
    正则集按 caption_overrides 用 reg_caption 计数。"""
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "train" / "1_data", ["a"], size=(256, 256))  # 原生 256 token
    _img(tmp_path / "reg" / "1_prior", ["r"], size=(256, 256))
    seen = []

    def count(caption: str) -> int:
        seen.append(caption)
        return 40 if caption == "reg prompt" else 10

    out = compute_navit_pack_estimate(
        [tmp_path / "train", tmp_path / "reg"], [1024],
        native_resolution=True, token_budget=200,
        text_token_count=count, caption_overrides=[None, "reg prompt"],
    )
    assert set(seen) == {"1girl", "reg prompt"}
    assert out["token_max"] <= 200  # 图像 downscale 时已给文本留位
    assert out["downscaled"] == 2


def test_text_counter_only_for_families_with_hook(monkeypatch) -> None:
    pytest.importorskip("torch")
    from studio.services.projects import versions
    from runtime.training.families import get_family

    monkeypatch.setattr(versions, "_NAVIT_TEXT_COUNTERS", {})
    assert versions.navit_text_token_counter("anima", "some/te") is None  # 文本走 cross-attn
    assert versions.navit_text_token_counter("moss", "") is None  # 未设 TE 路径

    moss = get_family("moss")
    calls = []

    def failing_load(*_a, **_kw):
        calls.append("load")
        raise OSError("tokenizer missing")

    monkeypatch.setattr(type(moss), "load_text", lambda self, *a, **kw: failing_load())
    assert versions.navit_text_token_counter("moss", "te/path") is None
    assert versions.navit_text_token_counter("moss", "te/path") is None
    assert calls == ["load", "load"]  # 失败不缓存（TE 下载后再请求能拿到）

    fake_text = type("T", (), {"token_count": lambda self, caption: len(caption) + 1})()
    monkeypatch.setattr(type(moss), "load_text", lambda self, *a, **kw: (calls.append("ok"), fake_text)[1])
    counter = versions.navit_text_token_counter("moss", "te/path")
    assert counter("abc") == 4
    assert versions.navit_text_token_counter("moss", "te/path") is counter  # 成功后复用
    assert calls.count("ok") == 1
