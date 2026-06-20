"""compute_bucket_histogram：后端用真 BucketManager 算训练集桶分布（桶预览数据源）。

复用 runtime 的 BucketManager + _parse_folder_meta，保证与实际训练逐桶一致。
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _sq(d: Path, names, size=(1024, 1024)) -> None:
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", size).save(d / f"{n}.png")


def test_single_resolution_repeat_counts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    _sq(tmp_path / "5_data", ["a", "b"])
    out = compute_bucket_histogram(tmp_path, [1024], 2.0)
    assert len(out) == 1 and out[0]["reso"] == 1024
    assert sum(b["count"] for b in out[0]["buckets"]) == 10  # 2 图 × repeat 5
    sq = next(b for b in out[0]["buckets"] if b["w"] == 1024 and b["h"] == 1024)
    assert sq["count"] == 10


def test_px_folder_override_resolution(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    _sq(tmp_path / "512px_2_hi", ["a"])
    out = compute_bucket_histogram(tmp_path, [1024], 2.0)
    assert [g["reso"] for g in out] == [512]  # px 覆盖 → 512 档，不在 1024
    assert sum(b["count"] for b in out[0]["buckets"]) == 2  # 1 图 × repeat 2


def test_resolution_list_fans_out(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    _sq(tmp_path / "data", ["a"])
    out = compute_bucket_histogram(tmp_path, [512, 768, 1024], 2.0)
    assert sorted(g["reso"] for g in out) == [512, 768, 1024]
    for g in out:
        assert sum(b["count"] for b in g["buckets"]) == 1  # 每档各 1 张


def test_empty_train_dir(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    assert compute_bucket_histogram(tmp_path / "nope", [1024], 2.0) == []
