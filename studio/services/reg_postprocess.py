"""正则集长宽比聚类后处理（PP5.5；2026-04-28 修正）。

**目的**：把长宽比（ar）接近的图 center-crop 成相同 ar，让 ARB 训练时多张图
落入同一桶 → 减少桶数。**只对齐 ar，不强制对齐分辨率** —— 训练 dataloader
会按桶把每张图 resize 到桶分辨率，所以 reg 集只要 ar 一致即可，分辨率保留
原图（不 upscale 不糊）。

由 `regex_dataset_builder.py` 的 postprocess 块库化而来；2026-04-28 修了
源脚本两个 bug：
1. KMeans 特征里掺了 log(width)，导致同 ar 但分辨率差异大的图被分到不同
   cluster；现仅用 [aspect_ratio]
2. smart 模式 ar 完全相等时走 stretch 公式，把 resize 比例算成 crop_ratio；
   现 smart 永远只 crop ar 不 resize，crop_ratio 纯按 ar 差算

算法：
- min_cluster_size = 2（< 2 → 不聚类，全放 cluster 0）
- 特征：仅 [aspect_ratio]（z-score 标准化）
- KMeans(random_state=42, n_init=10)，从 k=1 递增到 max_k = len(images)，找
  第一个让所有 cluster 中 max_crop ≤ max_crop_ratio 的方案
- 找不到满足限制的 K → 保持原样不修改（返回 None)
- 合并相似聚类：abs aspect 差 < 0.02 OR 相对差 < 5% 且合并后仍满足
  max_crop 限制
- inplace = True 永远（PP5.5 决议：不做备份）

method 语义：
- `smart` (默认)：仅 center crop 到 target_ar，**保留原分辨率**（推荐；
  ar 一致即落同 ARB 桶，不会因 upscale 糊化小图）
- `stretch`：直接拉伸到 target_w × target_h（变形）
- `crop`：先按 target_ar center crop 再 resize 到 target_w × target_h

用户视角入口：`postprocess(reg_dir, *, method='smart', max_crop_ratio=0.1)`，
返回 dict 摘要。失败 / 找不到 K 都不抛异常。
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from ..datasets import IMAGE_EXTS

ProgressFn = Callable[[str], None]

VALID_METHODS = {"smart", "stretch", "crop"}


# ---------------------------------------------------------------------------
# image collection
# ---------------------------------------------------------------------------


@dataclass
class _ImageInfo:
    path: Path
    width: int
    height: int
    aspect_ratio: float


def _collect_images(reg_dir: Path) -> list[_ImageInfo]:
    """递归扫 reg_dir 下所有图片，去重（小写文件名重复时保留第一份）。"""
    out: list[_ImageInfo] = []
    seen_lower: set[str] = set()
    for f in sorted(reg_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        key = f.name.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        try:
            with Image.open(f) as im:
                w, h = im.size
        except Exception:
            continue
        if not w or not h or h <= 0:
            continue
        out.append(_ImageInfo(path=f, width=w, height=h, aspect_ratio=w / h))
    return out


# ---------------------------------------------------------------------------
# crop ratio
# ---------------------------------------------------------------------------


def calculate_crop_ratio(
    img_w: int, img_h: int, target_w: int, target_h: int, method: str = "smart"
) -> float:
    """smart / stretch / crop 三种方法的「成本」估计，用于聚类阶段判 max_crop。

    - smart：纯按 ar 差算，crop_ratio = 1 - min(orig_ar, target_ar) /
      max(orig_ar, target_ar)。同 ar 返回 0；不参考绝对分辨率（修源脚本 bug）。
    - stretch：max(|w_diff|/w, |h_diff|/h)，衡量拉伸幅度。
    - crop：跟 smart 类似的 ar 差，但实际跑 resize_and_crop 时还要 resize 到
      target_w × target_h，所以加上 resize 维度也合理；保持源脚本公式。
    """
    if not img_w or not img_h or not target_w or not target_h:
        return 1.0
    if method == "smart":
        orig_ar = img_w / img_h
        target_ar = target_w / target_h
        big = max(orig_ar, target_ar)
        small = min(orig_ar, target_ar)
        return 1.0 - small / big if big > 0 else 0.0
    if method == "stretch":
        wr = abs(img_w - target_w) / max(img_w, 1)
        hr = abs(img_h - target_h) / max(img_h, 1)
        return max(wr, hr)
    if method == "crop":
        original_ar = img_w / img_h
        target_ar = target_w / target_h
        if original_ar > target_ar:
            crop_w = img_h * target_ar
            return (img_w - crop_w) / img_w if img_w > 0 else 0.0
        crop_h = img_w / target_ar
        return (img_h - crop_h) / img_h if img_h > 0 else 0.0
    # 默认
    wr = abs(img_w - target_w) / max(img_w, 1)
    hr = abs(img_h - target_h) / max(img_h, 1)
    return max(wr, hr)


# ---------------------------------------------------------------------------
# target resolution（中位数 + 调整到目标 aspect）
# ---------------------------------------------------------------------------


def _determine_target_resolution(cluster: list[_ImageInfo]) -> tuple[int, int]:
    widths = [i.width for i in cluster]
    heights = [i.height for i in cluster]
    return int(np.median(widths)), int(np.median(heights))


def _adjusted_target_for_cluster(
    cluster: list[_ImageInfo],
) -> tuple[int, int, float]:
    """返回 (target_w, target_h, target_ar) — 中位数分辨率，但调到中位数 AR。"""
    target_ar = float(np.median([i.aspect_ratio for i in cluster]))
    tw_med, th_med = _determine_target_resolution(cluster)
    ar_med = tw_med / th_med if th_med > 0 else 1.0
    if abs(ar_med - target_ar) > 0.01:
        if ar_med > target_ar:
            tw = int(th_med * target_ar)
            th = th_med
        else:
            tw = tw_med
            th = int(tw_med / target_ar)
    else:
        tw, th = tw_med, th_med
    return tw, th, target_ar


def _max_crop_in_cluster(cluster: list[_ImageInfo], method: str) -> tuple[float, int, int]:
    tw, th, _ = _adjusted_target_for_cluster(cluster)
    return (
        max(
            calculate_crop_ratio(i.width, i.height, tw, th, method)
            for i in cluster
        ),
        tw,
        th,
    )


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


def _merge_same_aspect_ratio_clusters(
    clusters: dict[int, list[_ImageInfo]],
    max_crop_ratio: float,
    method: str,
) -> dict[int, list[_ImageInfo]]:
    """合并相似 aspect ratio 的聚类（abs < 0.02 OR 相对 < 5% 且合并后仍满足）。"""
    if len(clusters) <= 1:
        return clusters
    info: dict[int, dict[str, Any]] = {}
    for cid, imgs in clusters.items():
        info[cid] = {
            "images": imgs,
            "target_ar": float(np.median([i.aspect_ratio for i in imgs])),
        }
    merged: dict[int, list[_ImageInfo]] = {}
    used: set[int] = set()
    new_id = 0
    sorted_ids = sorted(info.items(), key=lambda x: x[1]["target_ar"])

    for cid1, info1 in sorted_ids:
        if cid1 in used:
            continue
        bucket = [info1]
        used.add(cid1)
        target_ar = info1["target_ar"]
        changed = True
        while changed:
            changed = False
            for cid2, info2 in sorted_ids:
                if cid2 in used:
                    continue
                ar2 = info2["target_ar"]
                ar_diff_abs = abs(target_ar - ar2)
                ar_diff_rel = ar_diff_abs / max(target_ar, ar2, 0.001)
                if ar_diff_abs >= 0.02 and ar_diff_rel >= 0.05:
                    continue
                # 试合并：把当前 bucket + info2 一起算 max_crop
                merged_imgs: list[_ImageInfo] = []
                for b in bucket:
                    merged_imgs.extend(b["images"])
                merged_imgs.extend(info2["images"])
                mc, _, _ = _max_crop_in_cluster(merged_imgs, method)
                if mc <= max_crop_ratio:
                    bucket.append(info2)
                    used.add(cid2)
                    target_ar = float(np.median([i.aspect_ratio for i in merged_imgs]))
                    changed = True
                    break
        out_imgs: list[_ImageInfo] = []
        for b in bucket:
            out_imgs.extend(b["images"])
        merged[new_id] = out_imgs
        new_id += 1
    # 漏网（理论上不会有）
    for cid, imgs in clusters.items():
        if cid not in used:
            merged[new_id] = imgs
            new_id += 1
    return merged


def cluster_by_resolution(
    images: list[_ImageInfo], max_crop_ratio: float, method: str = "smart"
) -> Optional[dict[int, list[_ImageInfo]]]:
    """从 k=1 递增找第一个满足 max_crop ≤ limit 的 K，再做合并。

    特征仅 [aspect_ratio]（修源脚本 bug：原来 [ar, log(width)] 会让同 ar 但
    分辨率差异大的图分到不同 cluster，与 ARB 分桶目的不符）。

    返回 None 表示找不到满足限制的方案 — 上层应该保持原样不动文件。
    """
    if len(images) < 2:
        return {0: list(images)} if images else None

    features = np.array([[i.aspect_ratio] for i in images], dtype=float)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1, std)
    normalized = (features - mean) / std

    max_k = len(images)
    for k in range(1, max_k + 1):
        if k >= len(images):
            continue
        try:
            if k == 1:
                test_clusters: dict[int, list[_ImageInfo]] = {0: list(images)}
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(normalized)
                test_clusters = defaultdict(list)
                for idx, img in enumerate(images):
                    test_clusters[int(labels[idx])].append(img)
                test_clusters = dict(test_clusters)
        except Exception:
            continue

        all_valid = True
        for imgs in test_clusters.values():
            mc, _, _ = _max_crop_in_cluster(imgs, method)
            if mc > max_crop_ratio:
                all_valid = False
                break
        if all_valid:
            return _merge_same_aspect_ratio_clusters(
                test_clusters, max_crop_ratio, method
            )
    return None


# ---------------------------------------------------------------------------
# resize / crop
# ---------------------------------------------------------------------------


def resize_and_crop_image(
    image_path: Path, target_w: int, target_h: int, output_path: Path, method: str
) -> bool:
    """smart / stretch / crop 三种实际写盘行为。失败返回 False。

    smart 模式只 center-crop 到 target_ar，**保留原分辨率**（不 resize 不
    upscale），因为 ARB 训练时 dataloader 会按桶 resize。stretch / crop
    保持源脚本行为（强制对齐到 target_w × target_h）。
    """
    try:
        with Image.open(image_path) as img:
            ow, oh = img.size
            original_ar = ow / oh if oh > 0 else 1.0
            target_ar = target_w / target_h if target_h > 0 else 1.0

            if method == "smart":
                # 仅 center-crop 到 target_ar；保留尽可能大的原分辨率。
                if abs(original_ar - target_ar) < 1e-6:
                    img.save(output_path, quality=95)
                    return True
                if original_ar > target_ar:
                    crop_w = max(1, int(round(oh * target_ar)))
                    left = (ow - crop_w) // 2
                    cropped = img.crop((left, 0, left + crop_w, oh))
                else:
                    crop_h = max(1, int(round(ow / target_ar)))
                    top = (oh - crop_h) // 2
                    cropped = img.crop((0, top, ow, top + crop_h))
                cropped.save(output_path, quality=95)
            elif method == "stretch":
                resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                resized.save(output_path, quality=95)
            elif method == "crop":
                if original_ar > target_ar:
                    crop_w = int(oh * target_ar)
                    left = (ow - crop_w) // 2
                    cropped = img.crop((left, 0, left + crop_w, oh))
                else:
                    crop_h = int(ow / target_ar)
                    top = (oh - crop_h) // 2
                    cropped = img.crop((0, top, ow, top + crop_h))
                resized = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
                resized.save(output_path, quality=95)
            else:
                return False
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def postprocess(
    reg_dir: Path,
    *,
    method: str = "smart",
    max_crop_ratio: float = 0.1,
    on_progress: ProgressFn = print,
    cancel_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    """对 reg_dir 下所有图做分辨率聚类后处理（inplace 永远 True）。

    返回：
        {
            "clusters": int | None,        # None = 找不到满足限制的 K
            "processed": int,              # 实际改动的图数（不算 size 已匹配的）
            "skipped": int,                # size 已匹配 / 跳过
            "method": str,
            "max_crop_ratio": float,
            "target_resolutions": [(w, h, count), ...],
        }

    异常都不抛 — 失败时 clusters=None / processed=0。
    """
    if method not in VALID_METHODS:
        on_progress(f"[postprocess] 非法 method: {method}，跳过")
        return {
            "clusters": None, "processed": 0, "skipped": 0,
            "method": method, "max_crop_ratio": max_crop_ratio,
            "target_resolutions": [],
        }

    if not reg_dir.exists():
        on_progress(f"[postprocess] {reg_dir} 不存在，跳过")
        return {
            "clusters": None, "processed": 0, "skipped": 0,
            "method": method, "max_crop_ratio": max_crop_ratio,
            "target_resolutions": [],
        }

    on_progress(f"[postprocess] 收集图片 (method={method}, max_crop={max_crop_ratio})")
    images = _collect_images(reg_dir)
    if not images:
        on_progress("[postprocess] 没有图片，跳过")
        return {
            "clusters": None, "processed": 0, "skipped": 0,
            "method": method, "max_crop_ratio": max_crop_ratio,
            "target_resolutions": [],
        }
    on_progress(f"[postprocess] 共 {len(images)} 张图片")

    clusters = cluster_by_resolution(images, max_crop_ratio, method)
    if clusters is None:
        on_progress(
            f"[postprocess] 无 K 满足 max_crop ≤ {max_crop_ratio}，保持原样不修改"
        )
        return {
            "clusters": None, "processed": 0, "skipped": len(images),
            "method": method, "max_crop_ratio": max_crop_ratio,
            "target_resolutions": [],
        }

    on_progress(f"[postprocess] 聚类 {len(clusters)} 个 — 详情：")
    # 每个 cluster 详细信息（与源脚本日志对齐）
    for cid in sorted(clusters.keys()):
        cluster = clusters[cid]
        tw, th, tar = _adjusted_target_for_cluster(cluster)
        widths = [i.width for i in cluster]
        heights = [i.height for i in cluster]
        ars = [i.aspect_ratio for i in cluster]
        max_crop = max(
            calculate_crop_ratio(i.width, i.height, tw, th, method) for i in cluster
        )
        on_progress(f"  聚类 {cid}: {len(cluster)} 张")
        on_progress(f"    目标分辨率: {tw}x{th} (长宽比: {tar:.3f})")
        on_progress(
            f"    平均分辨率: {int(np.mean(widths))}x{int(np.mean(heights))}"
        )
        on_progress(
            f"    长宽比范围: {min(ars):.3f} - {max(ars):.3f} "
            f"(平均: {np.mean(ars):.3f})"
        )
        on_progress(f"    最大裁剪比例: {max_crop * 100:.1f}%")
        on_progress(
            f"    分辨率范围: {min(widths)}x{min(heights)} 到 "
            f"{max(widths)}x{max(heights)}"
        )

    processed = 0
    skipped = 0
    targets: list[tuple[int, int, int]] = []
    for cid in sorted(clusters.keys()):
        if cancel_event and cancel_event.is_set():
            on_progress("[postprocess] [cancel] 用户中止")
            break
        cluster = clusters[cid]
        tw, th, target_ar = _adjusted_target_for_cluster(cluster)
        on_progress(
            f"[postprocess] 处理聚类 {cid} ({len(cluster)} 张) → "
            f"{'ar=' + format(target_ar, '.3f') if method == 'smart' else f'{tw}x{th}'}"
        )
        targets.append((tw, th, len(cluster)))
        for info in cluster:
            if cancel_event and cancel_event.is_set():
                break
            # smart 只对齐 ar、保留原分辨率，跳过条件按 ar 判；
            # stretch / crop 仍按 target 分辨率判。
            if method == "smart":
                if abs(info.aspect_ratio - target_ar) < 1e-6:
                    skipped += 1
                    continue
            else:
                if info.width == tw and info.height == th:
                    skipped += 1
                    continue
            ok = resize_and_crop_image(info.path, tw, th, info.path, method)
            if ok:
                processed += 1
            else:
                skipped += 1
                on_progress(f"  ✗ resize 失败: {info.path.name}")

    on_progress(
        f"[postprocess] 完成: processed={processed}, skipped={skipped}, "
        f"clusters={len(clusters)}"
    )
    return {
        "clusters": len(clusters),
        "processed": processed,
        "skipped": skipped,
        "method": method,
        "max_crop_ratio": max_crop_ratio,
        "target_resolutions": targets,
    }
