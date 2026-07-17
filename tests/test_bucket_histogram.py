"""compute_bucket_histogram：后端用真 BucketManager 算训练集桶分布（桶预览数据源）。

复用 runtime 的 BucketManager + _parse_folder_meta，扫描规则镜像 ImageDataset._scan
（递归 rglob + 根目录散图 + 只计有 caption 的图），保证与实际训练逐桶一致。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _sq(d: Path, names, size=(1024, 1024), caption=True) -> None:
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", size).save(d / f"{n}.png")
        if caption:
            (d / f"{n}.txt").write_text("1girl", encoding="utf-8")


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


def test_only_captioned_images_counted(tmp_path: Path) -> None:
    # 镜像 trainer：无 caption 的图被丢弃，不计入直方图（否则预览 ≠ 实际训练）。
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    _sq(tmp_path / "1_data", ["a", "b"])               # 有 caption
    _sq(tmp_path / "1_data", ["c"], caption=False)     # 无 caption
    out = compute_bucket_histogram(tmp_path, [1024], 2.0)
    assert sum(b["count"] for b in out[0]["buckets"]) == 2  # c 不计


def test_dataset_importable_without_runtime_on_sys_path() -> None:
    """studio server 的 sys.path 只有仓库根（没有 runtime/）——dataset.py 及其
    导入链必须在 `runtime.training.*` 命名下可导入，否则 bucket-distribution
    endpoint 500。conftest 会把 runtime/ 注入 sys.path，进程内测试测不出来，
    必须开干净子进程。回归：多模型 PR-1 (#405) 引入的 `from training.*` 绝对
    导入曾砸坏这条链（修复 = families 子树改相对导入）。"""
    pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parent.parent
    code = (
        "import sys; "
        "sys.path[:] = [p for p in sys.path if 'runtime' not in p.lower()]; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "from runtime.training.dataset import BucketManager, ImageDataset; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "ok" in proc.stdout


def test_root_and_nested_images_counted(tmp_path: Path) -> None:
    # 根目录散图（repeat=1）+ 子文件夹深层图（rglob）都要计，跟 _scan 一致。
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    _sq(tmp_path, ["root"])                              # 根目录散图
    _sq(tmp_path / "2_data" / "nested", ["deep"])        # 子文件夹深层
    out = compute_bucket_histogram(tmp_path, [1024], 2.0)
    total = sum(b["count"] for g in out for b in g["buckets"])
    assert total == 1 + 2  # root×1 + deep×repeat2
