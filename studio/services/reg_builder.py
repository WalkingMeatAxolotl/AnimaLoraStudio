"""正则训练集构建器（PP5）。

由 `C:/Users/Mei/Desktop/SD/danbooru/dev/regex_dataset_builder.py` 库化而来：
去掉 input() / json 配置文件，全部参数走 `RegBuildOptions`；进度通过
`on_progress(line)` 推回调用方（worker 转写到日志 + bus.publish）。

**逻辑必须与源脚本一致** —— 阈值 / 常量 / 判定全照搬：
- 标签数递减序列：10 → 5 → 3 → 2 → 1（capped at max_search_tags）
- 每个标签数最多尝试 3 个不同 offset
- failed_tags：单标签搜索失败后不再尝试
- invalid_tag_combinations：找到结果但本批未下载（源数据集已有 / 不符合）
- max_rounds = 50；max_consecutive_failures = 5
- find_best_match skip_similar 取偶数索引（`posts[::2]`）
- 标签相似度 sigmoid 0.1 系数；分辨率打分 aspect 0.6 + resolution 0.4
- 最终分数 = tag_score + resolution_score * 0.1（resolution 作 tie-breaker）
- 80% 达成率算 success
- 每图后 0.5s，每批后 1s

不在范围（→ PP5.5）：分辨率 K-means 聚类后处理、按聚类裁剪到统一分辨率。
"""
from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import requests
from PIL import Image

from ..datasets import IMAGE_EXTS
from . import booru_api, tagedit


ProgressFn = Callable[[str], None]

VIDEO_EXTS = {
    "mp4", "webm", "avi", "mov", "mkv", "flv", "wmv", "mpg", "mpeg", "m4v",
}
# IMAGE_EXTS 在 datasets.py 用 ".xxx" 形式；这里需要不带点的形式（与 file_ext 比对）
_IMAGE_EXT_NODOT = {e.lstrip(".") for e in IMAGE_EXTS}


# ---------------------------------------------------------------------------
# options & meta
# ---------------------------------------------------------------------------


@dataclass
class RegBuildOptions:
    """构建正则集的所有参数。

    `target_count=None` → 用 train 总图片数（与源脚本默认一致）。
    """

    train_dir: Path
    output_dir: Path

    # API 凭据
    api_source: str = "gelbooru"
    user_id: str = ""
    api_key: str = ""
    username: str = ""

    # 上限
    target_count: Optional[int] = None
    max_search_tags: int = 20  # gelbooru 默认 20，danbooru 免费 2 / gold 6 / platinum 12
    batch_size: int = 5

    # 标签
    excluded_tags: list[str] = field(default_factory=list)  # 项目特定（角色名等）
    blacklist_tags: list[str] = field(default_factory=list)  # 全局黑名单

    # 选图策略
    skip_similar: bool = True
    aspect_ratio_filter_enabled: bool = False
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0

    # 文件落盘
    save_tags: bool = False  # PP5 默认 False（auto_tag 走 WD14）
    convert_to_png: bool = True
    remove_alpha_channel: bool = False

    # 后置
    auto_tag: bool = True  # 拉完 reg 后是否跑 WD14
    based_on_version: str = ""  # 仅用于 meta，不影响逻辑


@dataclass
class RegMeta:
    generated_at: float
    based_on_version: str
    api_source: str
    target_count: int
    actual_count: int
    source_tags: list[str]            # 实际用过的搜索 tag（去重）
    excluded_tags: list[str]
    blacklist_tags: list[str]
    failed_tags: list[str]            # 搜索失败的 tag
    train_tag_distribution: dict[str, int]  # train tag 频率（top 50）
    auto_tagged: bool


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
    session: Optional[requests.Session] = None,
) -> list[dict[str, Any]]:
    """搜索 + 本地过滤（黑名单 / 已排除 ID / 缺 id 或 url）。"""
    norm = _normalize_tags(tags)
    query = " ".join(norm)
    try:
        posts = booru_api.search_posts(
            api_source,
            query,
            page=page,
            limit=limit,
            user_id=user_id,
            api_key=api_key,
            username=username,
            session=session,
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


# ---------------------------------------------------------------------------
# main loops
# ---------------------------------------------------------------------------


def _build_for_subfolder(
    subfolder_name: str,
    subfolder_data: dict[str, Any],
    target_weights: dict[str, float],
    output_dir: Path,
    *,
    opts: RegBuildOptions,
    blacklist_tags: set[str],
    failed_tags: set[str],
    source_tags_used: set[str],
    source_image_ids: set[str],
    target_resolution: Optional[tuple[int, int]],
    target_aspect_ratio: Optional[float],
    resolution_std: Optional[tuple[float, float]],
    total_target_count: int,
    total_downloaded_so_far: int,
    on_progress: ProgressFn,
    cancel_event: Optional[threading.Event],
) -> tuple[bool, int]:
    """单子文件夹批量循环。返回 (success_80%达成, 实际下载数)。"""
    label = subfolder_name or "<root>"
    on_progress(f"\n===== 子文件夹 {label} =====")

    target_count = subfolder_data["image_count"]
    remaining_quota = total_target_count - total_downloaded_so_far
    if remaining_quota <= 0:
        on_progress(f"  ⚠️  已达总数量限制 {total_target_count}，跳过 {label}")
        return False, 0
    target_count = min(target_count, remaining_quota)
    on_progress(f"  目标 {target_count} 张，批次 {opts.batch_size}，最多 {opts.max_search_tags} tag")

    if subfolder_name == "":
        out_sub = output_dir
    else:
        out_sub = output_dir / subfolder_name
    out_sub.mkdir(parents=True, exist_ok=True)

    current_weights: dict[str, float] = defaultdict(float)
    downloaded_count = 0
    downloaded_ids: set[str] = set()
    skipped = 0
    failed = 0

    batch_round = 0
    max_rounds = 50
    consecutive_failures = 0
    max_consecutive_failures = 5
    invalid_tag_combinations: set[tuple[str, ...]] = set()

    while downloaded_count < target_count and batch_round < max_rounds:
        if cancel_event and cancel_event.is_set():
            on_progress("  [cancel] 用户中止")
            return False, downloaded_count

        batch_round += 1
        batch_remaining = min(opts.batch_size, target_count - downloaded_count)
        on_progress(f"\n  ----- batch {batch_round} ({downloaded_count}/{target_count}) -----")

        missing_tags = calculate_missing_tags(
            target_weights, current_weights, blacklist_tags, failed_tags
        )
        if not missing_tags:
            on_progress("  所有标签已达目标权重")
            break

        available_tags = [t for t, _ in missing_tags if t not in failed_tags]
        if not available_tags:
            on_progress(f"  ⚠️  所有缺失标签都搜索失败：{list(failed_tags)}")
            break

        info_preview = ", ".join(
            f"{t}(缺{w:.2f})" for t, w in missing_tags[:5]
        )
        on_progress(f"  最缺失: {info_preview}")

        # 标签数递减：10 → 5 → 3 → 2 → 1
        tag_counts_seq = [10, 5, 3, 2, 1]
        tag_counts_seq = [min(tc, opts.max_search_tags) for tc in tag_counts_seq]
        tag_counts_seq = list(dict.fromkeys(tag_counts_seq))  # 去重保序

        posts: list[dict[str, Any]] = []
        search_tags: list[str] = []
        tried_combinations: set[tuple[str, ...]] = set()
        all_tags_failed = False

        for tag_count in tag_counts_seq:
            if len(available_tags) < tag_count:
                continue
            max_attempts = min(3, len(available_tags) - tag_count + 1)
            for offset in range(max_attempts):
                if offset + tag_count > len(available_tags):
                    break
                cand_tags = available_tags[offset:offset + tag_count]
                comb_key = tuple(sorted(cand_tags))
                if comb_key in invalid_tag_combinations:
                    if offset == 0:
                        on_progress(f"    跳过无效组合: {cand_tags}")
                    continue
                if comb_key in tried_combinations:
                    continue
                tried_combinations.add(comb_key)

                on_progress(f"    用 {tag_count} tag 搜索: {cand_tags}")
                posts = _search_with_filters(
                    cand_tags,
                    api_source=opts.api_source,
                    user_id=opts.user_id,
                    api_key=opts.api_key,
                    username=opts.username,
                    blacklist_tags=blacklist_tags,
                    exclude_ids=downloaded_ids,
                    page=1,
                    limit=100,
                )
                if posts:
                    search_tags = cand_tags
                    for t in cand_tags:
                        source_tags_used.add(t)
                    break
                if tag_count == 1:
                    failed_tags.add(cand_tags[0])
                    on_progress(f"    ✗ tag '{cand_tags[0]}' 搜索失败，加入跳过列表")
                    remaining_avail = [
                        t for t, _ in missing_tags if t not in failed_tags
                    ]
                    if not remaining_avail:
                        on_progress(f"  ⚠️  所有缺失标签都已失败：{list(failed_tags)}")
                        all_tags_failed = True
                        break
            if posts or all_tags_failed:
                break

        if not posts:
            consecutive_failures += 1
            on_progress(
                f"  ⚠️  无匹配（连续失败 {consecutive_failures}/{max_consecutive_failures}）"
            )
            # 检测：所有可能组合都被标记为 invalid → 退出
            all_invalid = True
            for tc in tag_counts_seq:
                if len(available_tags) < tc:
                    continue
                tk = tuple(sorted(available_tags[:tc]))
                if tk not in invalid_tag_combinations:
                    all_invalid = False
                    break
            if all_invalid and invalid_tag_combinations:
                on_progress("  所有组合标记 invalid，停止搜索")
                break
            if consecutive_failures >= max_consecutive_failures:
                on_progress(f"  连续 {consecutive_failures} 次失败，停止")
                break
            continue
        consecutive_failures = 0
        on_progress(f"    候选 {len(posts)} 张")

        # 从候选下载本批次
        batch_downloaded = 0
        attempts = 0
        max_attempts = len(posts)

        while batch_downloaded < batch_remaining and attempts < max_attempts:
            if cancel_event and cancel_event.is_set():
                on_progress("  [cancel] 用户中止")
                return False, downloaded_count

            attempts += 1
            best_post, score = find_best_match(
                posts,
                target_weights,
                current_weights,
                target_count,
                api_source=opts.api_source,
                skip_similar=opts.skip_similar,
                target_resolution=target_resolution,
                target_aspect_ratio=target_aspect_ratio,
                resolution_std=resolution_std,
                source_image_ids=source_image_ids,
                aspect_ratio_filter_enabled=opts.aspect_ratio_filter_enabled,
                min_aspect_ratio=opts.min_aspect_ratio,
                max_aspect_ratio=opts.max_aspect_ratio,
            )
            if not best_post:
                break
            posts.remove(best_post)

            pid, file_url, file_ext, _ = booru_api.post_fields(
                best_post, opts.api_source
            )
            if not pid or not file_url:
                skipped += 1
                continue
            if pid in source_image_ids:
                on_progress(f"    跳过（源已有）: {pid}")
                skipped += 1
                downloaded_ids.add(pid)
                continue
            ext_lower = (file_ext or "").lower()
            if ext_lower in VIDEO_EXTS:
                on_progress(f"    跳过（视频）: {pid} .{file_ext}")
                skipped += 1
                downloaded_ids.add(pid)
                continue
            if ext_lower not in _IMAGE_EXT_NODOT:
                on_progress(f"    跳过（非图片）: {pid} .{file_ext}")
                skipped += 1
                downloaded_ids.add(pid)
                continue
            pw, ph = booru_api.post_dimensions(best_post, opts.api_source)
            if not check_aspect_ratio(
                pw, ph,
                enabled=opts.aspect_ratio_filter_enabled,
                min_ar=opts.min_aspect_ratio,
                max_ar=opts.max_aspect_ratio,
            ):
                ar_v = pw / ph if pw and ph else 0
                on_progress(f"    跳过（长宽比 {ar_v:.2f}）: {pid}")
                skipped += 1
                downloaded_ids.add(pid)
                continue

            ext = "png" if opts.convert_to_png else (file_ext or "jpg")
            image_path = out_sub / f"{pid}.{ext}"
            txt_path = out_sub / f"{pid}.txt"

            if image_path.exists():
                try:
                    image_path.unlink()
                except Exception as exc:
                    on_progress(f"    警告：无法删 {image_path.name}: {exc}")

            try:
                final = booru_api.download_image(
                    file_url,
                    image_path,
                    convert_to_png=opts.convert_to_png,
                    remove_alpha_channel=opts.remove_alpha_channel,
                    referer=booru_api.default_base_url(opts.api_source) + "/",
                )
            except Exception as exc:
                on_progress(f"    ✗ 下载失败: {pid} ({exc})")
                if image_path.exists():
                    try:
                        image_path.unlink()
                    except Exception:
                        pass
                failed += 1
                downloaded_ids.add(pid)
                continue

            post_tags = booru_api.post_tag_list(best_post, opts.api_source)
            if opts.save_tags and post_tags:
                txt_path.write_text(", ".join(post_tags), encoding="utf-8")

            for tag in post_tags:
                current_weights[tag] += 1 / target_count

            downloaded_count += 1
            batch_downloaded += 1
            downloaded_ids.add(pid)
            matched = [t for t in post_tags if t in target_weights][:5]
            on_progress(
                f"    [{downloaded_count}/{target_count}] ✓ {pid} "
                f"score={score:.4f} matched={matched}"
            )

            if (
                total_target_count is not None
                and total_downloaded_so_far + downloaded_count >= total_target_count
            ):
                on_progress(f"  已达总数量限制 {total_target_count}")
                break

            if cancel_event:
                if cancel_event.wait(0.5):
                    on_progress("  [cancel] 用户中止")
                    return False, downloaded_count
            else:
                time.sleep(0.5)

        on_progress(f"  本批次下载: {batch_downloaded}")

        if batch_downloaded == 0 and posts and search_tags:
            invalid_tag_combinations.add(tuple(sorted(search_tags)))
            on_progress(f"  ⚠️  组合 {search_tags} 找到候选但未下载，标 invalid")
            continue
        elif batch_downloaded > 0 and search_tags:
            invalid_tag_combinations.discard(tuple(sorted(search_tags)))

        if downloaded_count < target_count:
            if cancel_event:
                if cancel_event.wait(1.0):
                    on_progress("  [cancel] 用户中止")
                    return False, downloaded_count
            else:
                time.sleep(1.0)

    on_progress(
        f"\n  子文件夹 {label} 完成: {downloaded_count}/{target_count} "
        f"(skipped={skipped} failed={failed})"
    )
    success = downloaded_count >= target_count * 0.8
    return success, downloaded_count


def build(
    opts: RegBuildOptions,
    *,
    on_progress: ProgressFn = print,
    cancel_event: Optional[threading.Event] = None,
) -> RegMeta:
    """构建正则集主流程。返回 RegMeta（即使中途取消也尽量返回部分元数据）。

    源脚本逻辑：
    1. analyze_dataset_structure（train_dir）
    2. collect_source_image_ids（避免与 train 撞图）
    3. 自动黑名单：把 based_on_version 标签化加入临时 blacklist（防同人画师）
    4. 各子文件夹按比例分配目标数量，循环 _build_for_subfolder
    5. 写 meta.json
    """
    on_progress(f"[reg] api={opts.api_source} train={opts.train_dir}")

    if not opts.train_dir.exists():
        raise FileNotFoundError(f"train 目录不存在: {opts.train_dir}")
    if opts.api_source == "gelbooru" and not (opts.user_id and opts.api_key):
        raise ValueError("gelbooru 需要 user_id + api_key（去 Settings 配置 secrets.gelbooru）")

    structure = analyze_dataset_structure(opts.train_dir, on_progress)
    if structure["total_images"] == 0:
        raise ValueError(f"train 目录没有任何带 caption 的图片: {opts.train_dir}")

    source_image_ids = collect_source_image_ids(opts.train_dir)
    on_progress(f"[reg] 源图片 ID 共 {len(source_image_ids)} 个，避免重复")

    # 标签集合
    blacklist_tags = set(_normalize_tags(opts.blacklist_tags))
    excluded = set(_normalize_tags(opts.excluded_tags))
    blacklist_tags |= excluded
    # 自动黑名单：based_on_version
    if opts.based_on_version:
        ver_tag = opts.based_on_version.lower().strip().replace(" ", "_")
        if ver_tag and ver_tag not in blacklist_tags:
            blacklist_tags.add(ver_tag)
            on_progress(f"[reg] 自动加入黑名单: {ver_tag}")

    failed_tags: set[str] = set()
    source_tags_used: set[str] = set()

    # 目标数量
    total_target = (
        opts.target_count
        if opts.target_count and opts.target_count > 0
        else structure["total_images"]
    )
    on_progress(
        f"[reg] 目标 {total_target} 张（train 总 {structure['total_images']}），"
        f"子文件夹 {len(structure['subfolders'])} 个"
    )

    # 输出目录
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    target_resolution = structure.get("median_resolution")
    target_aspect_ratio = structure.get("median_aspect_ratio")
    resolution_std = structure.get("resolution_std")

    total_downloaded = 0
    success_subfolder_count = 0
    for sub_name, sub_data in structure["subfolders"].items():
        try:
            ok, dled = _build_for_subfolder(
                sub_name,
                sub_data,
                structure["global_tag_weights"],
                opts.output_dir,
                opts=opts,
                blacklist_tags=blacklist_tags,
                failed_tags=failed_tags,
                source_tags_used=source_tags_used,
                source_image_ids=source_image_ids,
                target_resolution=target_resolution,
                target_aspect_ratio=target_aspect_ratio,
                resolution_std=resolution_std,
                total_target_count=total_target,
                total_downloaded_so_far=total_downloaded,
                on_progress=on_progress,
                cancel_event=cancel_event,
            )
            if ok:
                success_subfolder_count += 1
            total_downloaded += dled
            if total_downloaded >= total_target:
                on_progress(f"[reg] 已达总目标 {total_target}，停止剩余子文件夹")
                break
        except Exception as exc:
            on_progress(f"[reg] 子文件夹 {sub_name} 出错: {exc}")
            import traceback
            on_progress(traceback.format_exc())

    # 写 meta
    top_dist = dict(structure["global_tag_freq"].most_common(50))
    meta = RegMeta(
        generated_at=time.time(),
        based_on_version=opts.based_on_version,
        api_source=opts.api_source,
        target_count=total_target,
        actual_count=total_downloaded,
        source_tags=sorted(source_tags_used),
        excluded_tags=list(excluded),
        blacklist_tags=sorted(blacklist_tags),
        failed_tags=sorted(failed_tags),
        train_tag_distribution=top_dist,
        auto_tagged=False,  # worker 在 auto_tag 完成后改写
    )
    write_meta(opts.output_dir, meta)

    on_progress(
        f"[reg] 完成: {total_downloaded}/{total_target}"
        f" 张（{success_subfolder_count}/{len(structure['subfolders'])} "
        f"子文件夹达 80%）"
    )
    return meta


# ---------------------------------------------------------------------------
# meta IO
# ---------------------------------------------------------------------------


META_FILENAME = "meta.json"


def meta_path(reg_dir: Path) -> Path:
    return reg_dir / META_FILENAME


def write_meta(reg_dir: Path, meta: RegMeta) -> Path:
    import json
    p = meta_path(reg_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def read_meta(reg_dir: Path) -> Optional[RegMeta]:
    import json
    p = meta_path(reg_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return RegMeta(**data)
    except Exception:
        return None


def update_meta_auto_tagged(reg_dir: Path, auto_tagged: bool) -> None:
    """auto_tag 完成后改写 meta.auto_tagged。"""
    m = read_meta(reg_dir)
    if m is None:
        return
    m.auto_tagged = auto_tagged
    write_meta(reg_dir, m)


# ---------------------------------------------------------------------------
# preview helper（端点 GET /reg/preview-tags 用）
# ---------------------------------------------------------------------------


def preview_train_tag_distribution(
    train_dir: Path, top: int = 20
) -> list[tuple[str, int]]:
    """轻量扫 train 的 tag 频率，返回 top N。不读图片尺寸（快）。"""
    counter: Counter[str] = Counter()
    if not train_dir.exists():
        return []
    for img in train_dir.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        counter.update(analyze_tags_in_file(img))
    return counter.most_common(top)
