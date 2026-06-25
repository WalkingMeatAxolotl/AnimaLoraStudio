"""Held-out validation split (pre-training)."""
from __future__ import annotations

from pathlib import Path

from studio.services import eval_validation


def _make_dataset(version_dir: Path, n: int, folder: str = "1_data") -> None:
    train = version_dir / "train" / folder
    train.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (train / f"img{i:02d}.png").write_bytes(b"png")
        (train / f"img{i:02d}.txt").write_text(f"caption {i}", encoding="utf-8")


def _val_names(version_dir: Path) -> set[str]:
    return {img.name for _folder, img in eval_validation.iter_images(version_dir / "validation")}


def test_split_moves_ratio_with_captions_mirroring_folders(tmp_path: Path) -> None:
    vdir = tmp_path / "v"
    _make_dataset(vdir, 10)

    summary = eval_validation.ensure_validation_split(vdir, ratio=0.2, seed=42)

    assert summary == {"train": 8, "validation": 2, "moved": 2, "target": 2}
    assert eval_validation.count_images(vdir / "train") == 8
    assert eval_validation.count_images(vdir / "validation") == 2
    # caption sidecars travel with the image; folder structure mirrored
    for _folder, img in eval_validation.iter_images(vdir / "validation"):
        assert img.parent.name == "1_data"
        assert img.with_suffix(".txt").exists()
        # moved out of train/
        assert not (vdir / "train" / "1_data" / img.name).exists()


def test_split_tops_up_to_target_then_noop(tmp_path: Path) -> None:
    vdir = tmp_path / "v"
    _make_dataset(vdir, 10)

    first = eval_validation.ensure_validation_split(vdir, ratio=0.3, seed=1)
    assert first["moved"] == 3 and first["validation"] == 3

    # 再跑同比例：已达目标 → 不再移动
    again = eval_validation.ensure_validation_split(vdir, ratio=0.3, seed=1)
    assert again["moved"] == 0 and again["validation"] == 3
    assert eval_validation.count_images(vdir / "train") == 7


def test_split_never_moves_back_when_ratio_lowered(tmp_path: Path) -> None:
    vdir = tmp_path / "v"
    _make_dataset(vdir, 10)
    eval_validation.ensure_validation_split(vdir, ratio=0.3, seed=1)  # val=3

    lowered = eval_validation.ensure_validation_split(vdir, ratio=0.1, seed=1)
    assert lowered["moved"] == 0 and lowered["validation"] == 3  # 不移回


def test_ratio_zero_is_noop(tmp_path: Path) -> None:
    vdir = tmp_path / "v"
    _make_dataset(vdir, 10)
    summary = eval_validation.ensure_validation_split(vdir, ratio=0.0, seed=1)
    assert summary["moved"] == 0
    assert eval_validation.count_images(vdir / "validation") == 0


def test_split_seed_is_reproducible(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_dataset(a, 10)
    _make_dataset(b, 10)

    eval_validation.ensure_validation_split(a, ratio=0.4, seed=7)
    eval_validation.ensure_validation_split(b, ratio=0.4, seed=7)
    assert _val_names(a) == _val_names(b)


def test_manual_validation_images_count_toward_target(tmp_path: Path) -> None:
    vdir = tmp_path / "v"
    _make_dataset(vdir, 10)
    # 用户手动往 validation/ 放了 2 张
    manual = vdir / "validation" / "1_data"
    manual.mkdir(parents=True, exist_ok=True)
    (manual / "manual0.png").write_bytes(b"png")
    (manual / "manual1.png").write_bytes(b"png")

    # total=12, ratio 0.25 → target=3；已有 2 → 只补 1
    summary = eval_validation.ensure_validation_split(vdir, ratio=0.25, seed=3)
    assert summary["target"] == 3
    assert summary["moved"] == 1
    assert summary["validation"] == 3
