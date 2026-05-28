"""数据集目录扫描：识别 Kohya 风格 N_xxx 前缀、统计样本数和 caption 类型。

不做缓存：每次端点调用都重新扫一遍。dataset 目录通常 < 几千张图，扫描很快。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# 全链路图片格式白名单 —— 上传 / 下载 / curation / tag / reg / 训练都引用这个集合。
# 保持与 anima_train.py:EXTS 同步（trainer 是独立脚本，不 import studio）。
# 删了 .jxl：PIL 12 没注册 .jxl，需要 pillow-jxl-plugin 才能解，booru 生态见不到。
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
KOHYA_PREFIX = re.compile(r"^(\d+)_(.+)$")


def parse_repeat(folder_name: str) -> tuple[int, str]:
    """`5_concept` → (5, 'concept')；无前缀返回 (1, name)。"""
    m = KOHYA_PREFIX.match(folder_name)
    if m:
        return int(m.group(1)), m.group(2)
    return 1, folder_name


def caption_kind(image_path: Path) -> str:
    """同名 .json > .txt > 'none'。"""
    if image_path.with_suffix(".json").exists():
        return "json"
    if image_path.with_suffix(".txt").exists():
        return "txt"
    return "none"


def scan_folder(folder: Path, sample_limit: int = 4) -> dict[str, Any]:
    """统计单个文件夹的样本与 caption 分布。"""
    repeat, label = parse_repeat(folder.name)
    counts = {"json": 0, "txt": 0, "none": 0}
    samples: list[str] = []
    image_count = 0

    if folder.is_dir():
        for entry in sorted(folder.iterdir()):
            if entry.suffix.lower() not in IMAGE_EXTS:
                continue
            image_count += 1
            counts[caption_kind(entry)] += 1
            if len(samples) < sample_limit:
                samples.append(entry.name)

    return {
        "name": folder.name,
        "label": label,
        "repeat": repeat,
        "image_count": image_count,
        "caption_types": counts,
        "samples": samples,
        "path": str(folder),
    }


def scan_dataset_root(root: Path) -> dict[str, Any]:
    """扫描 dataset 根目录，返回每个子目录的统计；根目录散图也算一个虚拟项。"""
    if not root.exists() or not root.is_dir():
        return {"root": str(root), "exists": False, "folders": []}

    folders: list[dict[str, Any]] = []
    # 子目录
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            folders.append(scan_folder(entry))

    # 根目录直接放的图（无前缀子目录的图算 repeat=1）
    root_loose = scan_folder(root, sample_limit=4)
    if root_loose["image_count"] > 0:
        root_loose["name"] = "(根目录)"
        root_loose["label"] = "(loose)"
        root_loose["repeat"] = 1
        folders.insert(0, root_loose)

    total_images = sum(f["image_count"] for f in folders)
    weighted = sum(f["image_count"] * f["repeat"] for f in folders)
    return {
        "root": str(root),
        "exists": True,
        "folders": folders,
        "total_images": total_images,
        "weighted_steps_per_epoch": weighted,
    }
