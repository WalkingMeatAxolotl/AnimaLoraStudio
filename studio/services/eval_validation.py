"""Held-out validation set for LoRA eval.

The validation set lives beside the training set under the version dir
(`versions/{label}/validation/`) and mirrors `train/`'s folder layout. It is the
held-out reference for post-training metrics: images here are NOT trained on, so
CLIP-I / DINO-I against them measure generalization rather than memorization.

`ensure_validation_split` runs once before training (supervisor `_spawn_task`):
it tops the validation set up to `ratio` of the whole dataset by *moving* images
(and their caption sidecars) out of `train/`. It never moves images back, and is
a no-op once the target is met — so re-training / resume is stable, and images a
user dropped into `validation/` by hand count toward the target.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from .dataset.scan import IMAGE_EXTS

logger = logging.getLogger(__name__)

# Caption sidecars travel with their image when an image is moved to validation/.
CAPTION_SUFFIXES = (".json", ".txt")


def validation_dir(version_dir: Path) -> Path:
    return version_dir / "validation"


def train_dir(version_dir: Path) -> Path:
    return version_dir / "train"


def iter_images(root: Path):
    """Yield (folder_name, image_path) for images under ``root/<folder>/``."""
    if not root.exists():
        return
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for image in sorted(folder.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                yield folder.name, image


def caption_sidecars(image: Path) -> list[Path]:
    return [p for suf in CAPTION_SUFFIXES if (p := image.with_suffix(suf)).exists()]


def count_images(root: Path) -> int:
    return sum(1 for _ in iter_images(root))


def ensure_validation_split(
    version_dir: Path, *, ratio: float, seed: int
) -> dict[str, Any]:
    """Top the validation set up to ``round(ratio × total)`` by moving from train/.

    ``total`` is the whole pool (train + validation), so the ratio stays stable
    across runs and counts any manually-added validation images. Returns a
    summary ``{train, validation, moved, target}``. ``moved=0`` when validation
    already meets the target — never moves images back.
    """
    train = train_dir(version_dir)
    val = validation_dir(version_dir)
    train_imgs = list(iter_images(train))
    val_count = count_images(val)
    total = len(train_imgs) + val_count
    if ratio <= 0 or total == 0:
        return {"train": len(train_imgs), "validation": val_count, "moved": 0, "target": 0}

    target = round(ratio * total)
    need = min(target - val_count, len(train_imgs))
    if need <= 0:
        return {
            "train": len(train_imgs),
            "validation": val_count,
            "moved": 0,
            "target": target,
        }

    rng = random.Random(seed)
    picks = rng.sample(train_imgs, need)
    moved = 0
    for folder_name, image in picks:
        dest_folder = val / folder_name
        dest_folder.mkdir(parents=True, exist_ok=True)
        for src in [image, *caption_sidecars(image)]:
            dest = dest_folder / src.name
            if dest.exists():
                continue
            src.replace(dest)
        moved += 1

    return {
        "train": len(train_imgs) - moved,
        "validation": val_count + moved,
        "moved": moved,
        "target": target,
    }


def split_for_task(conn, task: dict[str, Any], cfg_path: Path) -> dict[str, Any] | None:
    """Resolve a training task's version and run the split if it opted in.

    Reads the frozen task config from ``cfg_path``. Returns the split summary, or
    ``None`` when validation eval is disabled, the ratio is 0, or the task is not
    bound to a project/version.
    """
    import yaml

    try:
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(cfg, dict) or not cfg.get("eval_validation_enabled"):
        return None
    try:
        ratio = float(cfg.get("eval_validation_split_ratio") or 0.0)
    except (TypeError, ValueError):
        ratio = 0.0
    if ratio <= 0:
        return None

    project_id = int(task.get("project_id") or 0)
    version_id = int(task.get("version_id") or 0)
    if not project_id or not version_id:
        return None

    from studio.services.projects import projects, versions

    project = projects.get_project(conn, project_id)
    version = versions.get_version(conn, version_id)
    if not project or not version or int(version["project_id"]) != project_id:
        return None

    vdir = versions.version_dir(project_id, str(project["slug"]), str(version["label"]))
    try:
        seed = int(cfg.get("eval_validation_split_seed") or 0)
    except (TypeError, ValueError):
        seed = 0
    return ensure_validation_split(vdir, ratio=ratio, seed=seed)
