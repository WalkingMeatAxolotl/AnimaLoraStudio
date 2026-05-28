"""reg dataset 分析 / 评分原语（PR-3.9 从 builder.py 1108 行抽出）。

只做"看 + 算"，不动盘 / 不写 meta / 不串主循环。builder.py 的主流程
（_build_for_subfolder / _build_inner / build）从本模块调进来组合出"贪心搜
+ 评分挑图 + 落盘"的端到端逻辑。

公开（builder.py 主流程用）：
    analyze_dataset_structure   扫 train_dir 出 tag 频率 / 分辨率 / 长宽比统计
    collect_source_image_ids    扫源数据集 post_id（避免与 train 撞图）
    collect_existing_reg_per_subfolder  扫已存在 reg / 增量模式用
    calculate_tag_similarity    评分：负 MSE（tag 频率向量距离）
    calculate_resolution_similarity  评分：aspect 0.6 + resolution 0.4
    calculate_missing_tags      算"还差什么 tag" 的优先级队列
    check_aspect_ratio          单图长宽比是否在过滤器允许范围
    find_best_match             一批 posts 里挑最高分（综合 tag + 分辨率）

半公开（builder.py + tests 用）：
    _normalize_tags             小写 + 空格→_ + 去重保序
    analyze_tags_in_file        读单图 caption + 归一化
    _search_with_filters        search_posts + 本地黑名单 / id 排除过滤
    _IMAGE_EXT_NODOT            datasets.IMAGE_EXTS 去点版（与 booru file_ext 比对）

所有阈值 / 公式与源脚本 regex_dataset_builder.py 一致：
- 标签相似度 sigmoid 0.1 系数；分辨率打分 aspect 0.6 + resolution 0.4
- 最终分数 = tag_score + resolution_score * 0.1（resolution 作 tie-breaker）
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import requests
from PIL import Image

from ...datasets import IMAGE_EXTS
from ..booru import api as booru_api, pool as booru_pool
from .. import tagedit


ProgressFn = Callable[[str], None]

# IMAGE_EXTS 在 datasets.py 用 ".xxx" 形式；这里需要不带点的形式（与 file_ext 比对）
_IMAGE_EXT_NODOT = {e.lstrip(".") for e in IMAGE_EXTS}


# ---------------------------------------------------------------------------
# tag analysis
# ---------------------------------------------------------------------------


def _normalize_tags(raw: list[str]) -> list[str]:
    """小写 + 空格→下划线 + 去重保序。"""
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        n = t.lower().strip().replace(" ", "_")
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def analyze_tags_in_file(image_path: Path) -> list[str]:
    """读图片对应 caption（.txt 或 .json），返回标准化 tag 列表。

    复用 `tagedit.read_tags`，再做大小写 / 空格 / 去重标准化。
    """
    raw = tagedit.read_tags(image_path)
    return _normalize_tags(raw)


def analyze_dataset_structure(
    dataset_path: Path, on_progress: ProgressFn = print
) -> dict[str, Any]:
    """扫子文件夹 + 根目录，统计 tag 频率 / 分辨率 / 长宽比。

    返回结构与源脚本一致：
    ```
    {
        "subfolders": {name: {"images": [...], "tag_freq": Counter, "image_count": int}},
        "total_images": int,
        "global_tag_freq": Counter,
        "global_tag_weights": {tag: count/total_images},
        "resolutions": [(w, h), ...],
        "aspect_ratios": [...],
        "median_resolution": (w, h) | None,
        "median_aspect_ratio": float | None,
        "resolution_std": (sw, sh) | None,
    }
    ```
    """
    structure: dict[str, Any] = {
        "subfolders": {},
        "total_images": 0,
        "global_tag_freq": Counter(),
        "global_tag_weights": {},
        "resolutions": [],
        "aspect_ratios": [],
    }

    def _scan_folder(folder: Path, key: str) -> None:
        data = {"images": [], "tag_freq": Counter(), "image_count": 0}
        for img in sorted(folder.iterdir()):
            if not img.is_file():
                continue
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            tags = analyze_tags_in_file(img)
            if not tags:
                continue
            w, h, ar = None, None, None
            try:
                with Image.open(img) as im:
                    w, h = im.size
                    if h > 0:
                        ar = w / h
            except Exception as exc:
                on_progress(f"    警告：无法读取图片尺寸 {img.name}: {exc}")
            data["images"].append({
                "image": img.name,
                "tags": tags,
                "width": w,
                "height": h,
                "aspect_ratio": ar,
            })
            data["tag_freq"].update(tags)
            structure["global_tag_freq"].update(tags)
            data["image_count"] += 1
            structure["total_images"] += 1
            if w and h:
                structure["resolutions"].append((w, h))
                if ar:
                    structure["aspect_ratios"].append(ar)
        if data["image_count"] > 0:
            structure["subfolders"][key] = data
            on_progress(
                f"  [{key or '<root>'}] {data['image_count']} 张图片，"
                f"{len(data['tag_freq'])} tag 种类"
            )

    # 根目录直接图
    has_root_imgs = any(
        f.is_file() and f.suffix.lower() in IMAGE_EXTS
        for f in dataset_path.iterdir()
    )
    if has_root_imgs:
        _scan_folder(dataset_path, "")

    # 子文件夹
    for sub in sorted(dataset_path.iterdir()):
        if sub.is_dir():
            _scan_folder(sub, sub.name)

    # 全局权重
    if structure["total_images"] > 0:
        for tag, count in structure["global_tag_freq"].items():
            structure["global_tag_weights"][tag] = (
                count / structure["total_images"]
            )

    # 分辨率统计（中位数 + 标准差）
    if structure["resolutions"]:
        res_arr = np.array(structure["resolutions"])
        structure["median_resolution"] = (
            int(np.median(res_arr[:, 0])),
            int(np.median(res_arr[:, 1])),
        )
        structure["resolution_std"] = (
            float(np.std(res_arr[:, 0])),
            float(np.std(res_arr[:, 1])),
        )
        ar_arr = np.array(structure["aspect_ratios"])
        structure["median_aspect_ratio"] = float(np.median(ar_arr))
    else:
        structure["median_resolution"] = None
        structure["resolution_std"] = None
        structure["median_aspect_ratio"] = None

    return structure


def collect_source_image_ids(source_path: Path) -> set[str]:
    """递归收集源数据集所有图片的文件 stem（= post_id）。

    源脚本约定：booru 下载的文件名是 `{post_id}.{ext}`，所以 stem 就是 ID。
    """
    ids: set[str] = set()
    for img in source_path.rglob("*"):
        if not img.is_file():
            continue
        if img.suffix.lower().lstrip(".") not in _IMAGE_EXT_NODOT:
            continue
        ids.add(img.stem)
    return ids


def collect_existing_reg_per_subfolder(
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """PP5.1 — 扫已存在 reg 图，按子文件夹聚合 (ids, tags, count)。

    返回 {subfolder_name: {"ids": set[str], "tags": list[list[str]], "count": int}}
    subfolder_name == "" 表示 output_dir 根。
    """
    out: dict[str, dict[str, Any]] = {}
    if not output_dir.exists():
        return out

    def _ensure(key: str) -> dict[str, Any]:
        if key not in out:
            out[key] = {"ids": set(), "tags": [], "count": 0}
        return out[key]

    for img in output_dir.rglob("*"):
        if not img.is_file():
            continue
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(output_dir)
        # 子文件夹 = 路径第一段（如果存在）
        sub_key = rel.parts[0] if len(rel.parts) > 1 else ""
        bucket = _ensure(sub_key)
        bucket["ids"].add(img.stem)
        bucket["tags"].append(analyze_tags_in_file(img))
        bucket["count"] += 1
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def calculate_tag_similarity(
    target_weights: dict[str, float],
    candidate_tags: list[str],
    current_weights: dict[str, float],
    target_count: int,
) -> float:
    """与源脚本一致：负 MSE。"""
    new_weights = dict(current_weights)
    for tag in candidate_tags:
        new_weights[tag] = new_weights.get(tag, 0) + (1 / target_count)
    score = 0.0
    all_tags = set(target_weights.keys()) | set(candidate_tags)
    for tag in all_tags:
        score += (target_weights.get(tag, 0) - new_weights.get(tag, 0)) ** 2
    return -score


def calculate_resolution_similarity(
    post_w: int,
    post_h: int,
    target_resolution: tuple[int, int],
    target_aspect_ratio: float,
    resolution_std: Optional[tuple[float, float]] = None,
) -> float:
    """与源脚本一致：长宽比 0.6 + 分辨率 0.4。"""
    if not post_w or not post_h or not target_resolution or not target_aspect_ratio:
        return 0.0
    post_ar = post_w / post_h if post_h > 0 else 1.0
    aspect_diff = abs(post_ar - target_aspect_ratio) / max(target_aspect_ratio, 0.001)
    aspect_score = 1.0 / (1.0 + aspect_diff * 10)

    tw, th = target_resolution
    if resolution_std and resolution_std[0] > 0 and resolution_std[1] > 0:
        width_score = 1.0 / (1.0 + abs(post_w - tw) / (resolution_std[0] * 2))
        height_score = 1.0 / (1.0 + abs(post_h - th) / (resolution_std[1] * 2))
    else:
        width_score = 1.0 / (1.0 + abs(post_w - tw) / max(tw, 1) * 10)
        height_score = 1.0 / (1.0 + abs(post_h - th) / max(th, 1) * 10)

    resolution_score = (width_score + height_score) / 2
    return aspect_score * 0.6 + resolution_score * 0.4


def calculate_missing_tags(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    blacklist_tags: set[str],
    failed_tags: set[str],
) -> list[tuple[str, float]]:
    missing: list[tuple[str, float]] = []
    for tag, tw in target_weights.items():
        if tag in blacklist_tags or tag in failed_tags:
            continue
        diff = tw - current_weights.get(tag, 0.0)
        if diff > 0:
            missing.append((tag, diff))
    missing.sort(key=lambda x: x[1], reverse=True)
    return missing


def check_aspect_ratio(
    w: Optional[int],
    h: Optional[int],
    *,
    enabled: bool,
    min_ar: float,
    max_ar: float,
) -> bool:
    if not enabled:
        return True
    if not w or not h or h == 0:
        return False
    ar = w / h
    return min_ar <= ar <= max_ar


def find_best_match(
    posts: list[dict[str, Any]],
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    target_count: int,
    *,
    api_source: str,
    skip_similar: bool,
    target_resolution: Optional[tuple[int, int]] = None,
    target_aspect_ratio: Optional[float] = None,
    resolution_std: Optional[tuple[float, float]] = None,
    source_image_ids: Optional[set[str]] = None,
    aspect_ratio_filter_enabled: bool = False,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> tuple[Optional[dict[str, Any]], float]:
    if source_image_ids is None:
        source_image_ids = set()

    candidates = posts[::2] if skip_similar else posts
    best_post = None
    best_score = float("-inf")

    for post in candidates:
        post_id, _, _, _ = booru_api.post_fields(post, api_source)
        if post_id and post_id in source_image_ids:
            continue
        pw, ph = booru_api.post_dimensions(post, api_source)
        if not check_aspect_ratio(
            pw, ph,
            enabled=aspect_ratio_filter_enabled,
            min_ar=min_aspect_ratio,
            max_ar=max_aspect_ratio,
        ):
            continue
        post_tags = booru_api.post_tag_list(post, api_source)
        tag_score = calculate_tag_similarity(
            target_weights, post_tags, current_weights, target_count
        )
        res_score = 0.0
        if target_resolution and target_aspect_ratio and pw and ph:
            res_score = calculate_resolution_similarity(
                pw, ph, target_resolution, target_aspect_ratio, resolution_std
            )
        final_score = tag_score + res_score * 0.1
        if final_score > best_score:
            best_score = final_score
            best_post = post
    return best_post, best_score


# ---------------------------------------------------------------------------
# search wrapper（带本地过滤）
# ---------------------------------------------------------------------------


def _search_with_filters(
    tags: list[str],
    *,
    api_source: str,
    user_id: str,
    api_key: str,
    username: str,
    blacklist_tags: set[str],
    exclude_ids: set[str],
    page: int = 1,
    limit: int = 100,
    client: Optional[booru_pool.BooruClient] = None,
) -> list[dict[str, Any]]:
    """搜索 + 本地过滤（黑名单 / 已排除 ID / 缺 id 或 url）。

    PP9: `client` 走统一池子（API token bucket）；不传则直接调底层（旧测兼容）。
    """
    norm = _normalize_tags(tags)
    query = " ".join(norm)
    try:
        if client is not None:
            posts = client.search_posts(
                api_source,
                query,
                page=page,
                limit=limit,
                user_id=user_id,
                api_key=api_key,
                username=username,
            )
        else:
            posts = booru_api.search_posts(
                api_source,
                query,
                page=page,
                limit=limit,
                user_id=user_id,
                api_key=api_key,
                username=username,
            )
    except requests.RequestException:
        return []

    out: list[dict[str, Any]] = []
    for post in posts:
        pid, file_url, _, _ = booru_api.post_fields(post, api_source)
        if not pid or not file_url:
            continue
        if pid in exclude_ids:
            continue
        # 本地黑名单过滤
        if blacklist_tags:
            ptags = booru_api.post_tag_list(post, api_source)
            if any(t in blacklist_tags for t in ptags):
                continue
        out.append(post)
    return out
