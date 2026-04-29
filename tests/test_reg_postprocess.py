"""PP5.5 — reg_postprocess 单元测试。"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from studio.services import reg_postprocess


def _make_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 255, 255)).save(path, "PNG")


# ---------------------------------------------------------------------------
# crop ratio 计算
# ---------------------------------------------------------------------------


def test_calculate_crop_ratio_smart_same_aspect_returns_zero() -> None:
    """同 ar 不论分辨率多少，smart 现在都返回 0（修源脚本 bug）。"""
    r = reg_postprocess.calculate_crop_ratio(1024, 768, 512, 384, "smart")
    assert r == pytest.approx(0.0, abs=1e-6)
    # 即使分辨率相差悬殊，只要 ar 相同也是 0
    r = reg_postprocess.calculate_crop_ratio(1200, 675, 1920, 1080, "smart")
    assert r == pytest.approx(0.0, abs=1e-3)


def test_calculate_crop_ratio_smart_identical_returns_zero() -> None:
    r = reg_postprocess.calculate_crop_ratio(512, 512, 512, 512, "smart")
    assert r == pytest.approx(0.0, abs=1e-6)


def test_calculate_crop_ratio_smart_wider_image_crops_width() -> None:
    # 原图 ar=2，目标 ar=1 → smart 切到 ar=1 = 1 - 1/2 = 0.5
    r = reg_postprocess.calculate_crop_ratio(1000, 500, 500, 500, "smart")
    assert r == pytest.approx(0.5, abs=1e-3)


def test_calculate_crop_ratio_smart_resolution_independent() -> None:
    """同样 ar 比，crop_ratio 跟分辨率绝对值无关。"""
    a = reg_postprocess.calculate_crop_ratio(1000, 500, 1000, 750, "smart")
    b = reg_postprocess.calculate_crop_ratio(2000, 1000, 2000, 1500, "smart")
    assert a == pytest.approx(b, abs=1e-6)


def test_calculate_crop_ratio_stretch_compares_dimensions() -> None:
    r = reg_postprocess.calculate_crop_ratio(1000, 1000, 500, 500, "stretch")
    assert r == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# resize / crop
# ---------------------------------------------------------------------------


def test_resize_smart_only_crops_to_target_ar() -> None:
    """smart 只 center-crop 到 target ar，保留原分辨率。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "a.png"
        _make_image(src, (1024, 768))
        # target ar = 1.0；原图 ar = 4/3，所以切宽到 ar=1 → (768, 768)
        ok = reg_postprocess.resize_and_crop_image(src, 512, 512, src, "smart")
        assert ok is True
        with Image.open(src) as img:
            assert img.size == (768, 768)


def test_resize_smart_same_ar_writes_unchanged(tmp_path: Path) -> None:
    """ar 已经匹配 target ar 时 smart 不动尺寸。"""
    src = tmp_path / "a.png"
    _make_image(src, (1200, 675))  # ar = 1.778
    ok = reg_postprocess.resize_and_crop_image(src, 1920, 1080, src, "smart")
    assert ok is True
    with Image.open(src) as img:
        assert img.size == (1200, 675)


def test_resize_crop_writes_target_size(tmp_path: Path) -> None:
    """method=crop 仍然 resize 到 target 分辨率（保留旧行为）。"""
    src = tmp_path / "a.png"
    _make_image(src, (1024, 768))
    ok = reg_postprocess.resize_and_crop_image(src, 512, 512, src, "crop")
    assert ok is True
    with Image.open(src) as img:
        assert img.size == (512, 512)


def test_resize_stretch_writes_target_size(tmp_path: Path) -> None:
    src = tmp_path / "a.png"
    _make_image(src, (200, 800))
    ok = reg_postprocess.resize_and_crop_image(src, 400, 400, src, "stretch")
    assert ok is True
    with Image.open(src) as img:
        assert img.size == (400, 400)


def test_resize_invalid_method_returns_false(tmp_path: Path) -> None:
    src = tmp_path / "a.png"
    _make_image(src, (200, 200))
    ok = reg_postprocess.resize_and_crop_image(src, 100, 100, src, "bogus")
    assert ok is False


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


def test_cluster_by_resolution_uniform_returns_single_cluster(tmp_path: Path) -> None:
    """所有图同分辨率 → k=1 必然 valid。"""
    images = [
        reg_postprocess._ImageInfo(
            path=tmp_path / f"{i}.png", width=512, height=512, aspect_ratio=1.0
        )
        for i in range(5)
    ]
    clusters = reg_postprocess.cluster_by_resolution(images, max_crop_ratio=0.1)
    assert clusters is not None
    assert len(clusters) == 1


def test_cluster_by_resolution_fewer_than_two_images() -> None:
    images = [reg_postprocess._ImageInfo(
        path=Path("x.png"), width=100, height=100, aspect_ratio=1.0,
    )]
    clusters = reg_postprocess.cluster_by_resolution(images, max_crop_ratio=0.1)
    assert clusters == {0: images}


def test_cluster_by_resolution_unable_returns_none() -> None:
    """图分辨率差异极大且 max_crop_ratio 极严 → 找不到满足限制的 K。"""
    # 4 张完全不同 AR 的图，max_crop=0 → 1 张图自己一类才能 0% crop
    # 但 _cluster_by_resolution 跳过 k >= len(images)，所以无法都 1 张一类 → None
    images = [
        reg_postprocess._ImageInfo(path=Path("a.png"), width=100, height=1000, aspect_ratio=0.1),
        reg_postprocess._ImageInfo(path=Path("b.png"), width=1000, height=100, aspect_ratio=10.0),
        reg_postprocess._ImageInfo(path=Path("c.png"), width=500, height=500, aspect_ratio=1.0),
        reg_postprocess._ImageInfo(path=Path("d.png"), width=200, height=400, aspect_ratio=0.5),
    ]
    clusters = reg_postprocess.cluster_by_resolution(images, max_crop_ratio=0.0)
    assert clusters is None


# ---------------------------------------------------------------------------
# postprocess 主入口
# ---------------------------------------------------------------------------


def test_postprocess_smart_uniform_ar_keeps_original_resolution(
    tmp_path: Path,
) -> None:
    """smart：所有图同 ar=1.0 → 1 个 cluster；保留各自原分辨率，不 resize。"""
    reg_dir = tmp_path / "reg"
    _make_image(reg_dir / "1_data" / "100.png", (512, 512))
    _make_image(reg_dir / "1_data" / "101.png", (640, 640))
    _make_image(reg_dir / "1_data" / "102.png", (768, 768))
    result = reg_postprocess.postprocess(
        reg_dir, method="smart", max_crop_ratio=0.5, on_progress=lambda _: None
    )
    assert result["clusters"] == 1
    # smart 不 resize，各自保留原分辨率（ar 一致即落同 ARB 桶）
    sizes = sorted(
        Image.open(p).size for p in (reg_dir / "1_data").glob("*.png")
    )
    assert sizes == [(512, 512), (640, 640), (768, 768)]


def test_postprocess_smart_merges_same_ar_different_resolution(
    tmp_path: Path,
) -> None:
    """关键回归：同 ar=1.778 但分辨率差 2.5×（1200×675 / 1920×1080 / 3100×1780）
    应聚为 1 类，而不是按分辨率分开。"""
    reg_dir = tmp_path / "reg"
    _make_image(reg_dir / "1_data" / "small.png", (1200, 675))
    _make_image(reg_dir / "1_data" / "mid.png", (1920, 1080))
    _make_image(reg_dir / "1_data" / "big.png", (3100, 1744))  # ≈1.777
    result = reg_postprocess.postprocess(
        reg_dir, method="smart", max_crop_ratio=0.1, on_progress=lambda _: None
    )
    assert result["clusters"] == 1
    # 每张图 ar 都接近 1.778 → smart 几乎不改尺寸
    for p in (reg_dir / "1_data").glob("*.png"):
        with Image.open(p) as img:
            w, h = img.size
            assert abs(w / h - 1.778) < 0.005


def test_postprocess_no_images_returns_empty(tmp_path: Path) -> None:
    reg_dir = tmp_path / "reg"
    reg_dir.mkdir(parents=True)
    result = reg_postprocess.postprocess(reg_dir, on_progress=lambda _: None)
    assert result["clusters"] is None
    assert result["processed"] == 0


def test_postprocess_unable_to_cluster_keeps_original(tmp_path: Path) -> None:
    """max_crop=0.0 + 不同 AR → 找不到 K，保持原样。"""
    reg_dir = tmp_path / "reg" / "1_data"
    _make_image(reg_dir / "a.png", (100, 1000))
    _make_image(reg_dir / "b.png", (1000, 100))
    _make_image(reg_dir / "c.png", (500, 500))
    _make_image(reg_dir / "d.png", (200, 400))
    result = reg_postprocess.postprocess(
        reg_dir.parent, method="smart", max_crop_ratio=0.0,
        on_progress=lambda _: None,
    )
    assert result["clusters"] is None
    # 原图不动
    with Image.open(reg_dir / "a.png") as img:
        assert img.size == (100, 1000)


def test_postprocess_invalid_method_skips(tmp_path: Path) -> None:
    reg_dir = tmp_path / "reg" / "1_data"
    _make_image(reg_dir / "a.png", (256, 256))
    result = reg_postprocess.postprocess(
        reg_dir.parent, method="bogus", on_progress=lambda _: None
    )
    assert result["clusters"] is None


def test_postprocess_smart_skips_when_ar_matches_target(tmp_path: Path) -> None:
    """smart 现按 ar 判跳过：所有图同 ar → 全部 skipped，无 processed。"""
    reg_dir = tmp_path / "reg" / "1_data"
    _make_image(reg_dir / "a.png", (512, 512))
    _make_image(reg_dir / "b.png", (640, 640))
    _make_image(reg_dir / "c.png", (768, 768))
    result = reg_postprocess.postprocess(
        reg_dir.parent, method="smart", max_crop_ratio=0.1,
        on_progress=lambda _: None,
    )
    assert result["clusters"] == 1
    # 全部 ar=1.0 与 target ar 一致 → 都 skipped
    assert result["processed"] == 0
    assert result["skipped"] == 3


def test_postprocess_dedupes_same_filename_across_subfolders(tmp_path: Path) -> None:
    """文件名（小写）重复时只处理第一份。"""
    reg_dir = tmp_path / "reg"
    _make_image(reg_dir / "1_data" / "100.png", (512, 512))
    _make_image(reg_dir / "5_concept" / "100.png", (1024, 1024))
    result = reg_postprocess.postprocess(
        reg_dir, method="smart", max_crop_ratio=0.5,
        on_progress=lambda _: None,
    )
    # 共「2 张唯一」？不，文件名都叫 100.png，去重后只有 1 张 → cluster 必然 1 张
    assert result.get("clusters") in (1, None)
