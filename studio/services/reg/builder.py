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

PR-3.9 后：纯分析 / 评分 / 搜索过滤函数搬到 `analysis.py`，本文件留主流程
（_build_for_subfolder / _build_inner / build）+ RegBuildOptions / RegMeta /
meta CRUD。analysis 的 11 个公开 + 半公开名通过下方 `from .analysis import ...`
re-export 进本模块命名空间，保 `from studio.services.reg.builder import X` /
`reg_builder.X` 旧 import 路径兼容（tests 大量用 attribute 访问形式）。
"""
from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ...services.dataset.scan import IMAGE_EXTS
from ..booru import api as booru_api, pool as booru_pool
from .analysis import (
    _IMAGE_EXT_NODOT,
    _normalize_tags,
    _search_with_filters,
    analyze_dataset_structure,
    analyze_tags_in_file,
    calculate_missing_tags,
    calculate_resolution_similarity,
    calculate_tag_similarity,
    check_aspect_ratio,
    collect_existing_reg_per_subfolder,
    collect_source_image_ids,
    find_best_match,
)


ProgressFn = Callable[[str], None]

VIDEO_EXTS = {
    "mp4", "webm", "avi", "mov", "mkv", "flv", "wmv", "mpg", "mpeg", "m4v",
}


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
    # batch_size = 搜索循环内部「每下 N 张重算 missing_weight」的步进；与 train 子文件夹镜像无关
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
    auto_tag: bool = True  # 拉完 reg 后是否跑 tagger
    auto_tag_kind: str = "wd14"  # A3 — tagger 类型；约束在 VALID_TAGGER_NAMES，UI 目前暴露 wd14/cltagger
    auto_dedup: bool = True  # A4 — build 后自动 dedup + 不够补足循环
    # B1（PR-2）：构建模式。
    # - "mirror"：源脚本旧行为，按 train 子文件夹镜像，每个子文件夹独立按其图数拉。
    # - "flat"：所有图进 `1_data/` 单桶；target_count 决定总图数。
    build_mode: str = "flat"
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
    # A3 — 实际跑过 auto_tag 的 tagger 名（"wd14" / "cltagger" / ...）。
    # None = 没跑或跑前 meta；老 meta（无此字段）按 None 解读。auto_tagged=True
    # 但 auto_tag_kind=None 视为「未知 tagger」（旧版本数据）。
    auto_tag_kind: Optional[str] = None
    # B1（PR-2）：该 reg 集是按哪种 mode 生成的（mirror / flat）。
    # 老 meta（无此字段）= mirror（PR-1 之前默认）。前端切 mode 用此推断当前结构，
    # 不一致时拦截 + 提示先清空。
    build_mode: str = "mirror"
    incremental_runs: int = 0         # 补足跑了多少次（PP5.1）
    # PP5.5 — 后处理摘要（postprocessed_at=None 表示没跑或失败）
    postprocessed_at: Optional[float] = None
    postprocess_clusters: Optional[int] = None
    postprocess_method: Optional[str] = None
    postprocess_max_crop_ratio: Optional[float] = None
    # 生成方式："scrape" = booru 拉取（默认，兼容旧 meta），
    # "ai_base" = base 模型对 train tag 反向出对照图作正则集（先验生成）。
    # 引入此字段是为了让 api_source 字段不被 "ai_generated" 这种伪 source 污染：
    # generation_method="scrape" 时 api_source 才是 "gelbooru"|"danbooru"；
    # generation_method="ai_base" 时 api_source 留空（语义上无来源）。
    generation_method: str = "scrape"


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
    pre_existing: Optional[dict[str, Any]] = None,  # PP5.1
    client: Optional[booru_pool.BooruClient] = None,  # PP9
    deleted_ids: Optional[set[str]] = None,  # A2 — 用户从 UI 删过的 booru ID
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

    # A2 — 用户从 UI 删过的 booru ID 并入 downloaded_ids；这样 search 阶段
    # 的 `exclude_ids=downloaded_ids` 自动把它们排除掉，避免增量补足再拉回来。
    # 不计入 downloaded_count，因为它们已经不在盘上了。
    if deleted_ids:
        downloaded_ids.update(deleted_ids)
        on_progress(
            f"  [a2] 排除已删 booru ID {len(deleted_ids)} 个（来自 reg/.deleted_ids.json）"
        )

    # PP5.1 — incremental：把已有图作为「已下载」计入起点 + 累加 current_weights
    if pre_existing and pre_existing.get("count"):
        existing_count = int(pre_existing["count"])
        downloaded_count = min(existing_count, target_count)
        for pid in pre_existing.get("ids") or set():
            downloaded_ids.add(str(pid))
        for tags in pre_existing.get("tags") or []:
            for t in tags:
                current_weights[t] += 1 / target_count
        on_progress(
            f"  [incremental] 沿用已有 {existing_count} 张（计入起点 {downloaded_count}/{target_count}）"
        )
        if downloaded_count >= target_count:
            on_progress("  [incremental] 已有图已达目标，无需补足")
            return True, downloaded_count

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
                    client=client,
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
                if client is not None:
                    final = client.download_image(
                        file_url,
                        image_path,
                        convert_to_png=opts.convert_to_png,
                        remove_alpha_channel=opts.remove_alpha_channel,
                        referer=booru_api.default_base_url(opts.api_source) + "/",
                        username=opts.username,
                    )
                else:
                    final = booru_api.download_image(
                        file_url,
                        image_path,
                        convert_to_png=opts.convert_to_png,
                        remove_alpha_channel=opts.remove_alpha_channel,
                        referer=booru_api.default_base_url(opts.api_source) + "/",
                        username=opts.username,
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
                # caption 一律空格形式（与 WD14/CLTagger 输出、训练集统一）。下划线
                # 只是 booru 的线格式，仅在匹配 / 查询时用 —— post_tags 原值在下面
                # current_weights 仍按下划线累加，不动。tag-form 约定：用户可见 /
                # caption = 空格；underscore 只在 booru 边界。
                txt_path.write_text(
                    ", ".join(t.replace("_", " ") for t in post_tags),
                    encoding="utf-8",
                )

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

            # PP9 — 删每图 0.5s 硬 sleep；速率由 BooruClient 的 token bucket 控
            if cancel_event and cancel_event.is_set():
                on_progress("  [cancel] 用户中止")
                return False, downloaded_count

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
    incremental: bool = False,
    client: Optional[booru_pool.BooruClient] = None,
) -> RegMeta:
    """构建正则集主流程。返回 RegMeta（即使中途取消也尽量返回部分元数据）。

    源脚本逻辑：
    1. analyze_dataset_structure（train_dir）
    2. collect_source_image_ids（避免与 train 撞图）
    3. 自动黑名单：把 based_on_version 标签化加入临时 blacklist（防同人画师）
    4. 各子文件夹按比例分配目标数量，循环 _build_for_subfolder
    5. 写 meta.json

    PP5.1：`incremental=True` 时保留 output_dir 已有图作为「已下载」起点，
    `current_weights` 从已有 caption 累加，仅补足缺口；旧 meta 的
    `incremental_runs + 1` 写回。
    """
    on_progress(f"[reg] api={opts.api_source} train={opts.train_dir}")

    if not opts.train_dir.exists():
        raise FileNotFoundError(f"train 目录不存在: {opts.train_dir}")
    if opts.api_source == "gelbooru" and not (opts.user_id and opts.api_key):
        raise ValueError("gelbooru 需要 user_id + api_key（去 Settings 配置 secrets.gelbooru）")
    if opts.api_source == "danbooru" and not (opts.username and opts.api_key):
        raise ValueError("danbooru 需要 username + api_key（去 Settings 配置 secrets.danbooru）")

    # PP9 — 没传 client 就建一个（按 secrets.download.* 调速），用完关掉
    owns_client = False
    if client is None:
        try:
            from .. import secrets as _secrets
            d = _secrets.load().download
            cfg = booru_pool.BooruPoolConfig(
                parallel_workers=d.parallel_workers,
                api_rate_per_sec=d.api_rate_per_sec,
                cdn_rate_per_sec=d.cdn_rate_per_sec,
            )
        except Exception:  # noqa: BLE001
            cfg = booru_pool.BooruPoolConfig()
        client = booru_pool.BooruClient(cfg)
        owns_client = True

    try:
        return _build_inner(
            opts,
            client=client,
            on_progress=on_progress,
            cancel_event=cancel_event,
            incremental=incremental,
        )
    finally:
        if owns_client:
            client.close()


def _build_inner(
    opts: RegBuildOptions,
    *,
    client: booru_pool.BooruClient,
    on_progress: ProgressFn,
    cancel_event: Optional[threading.Event],
    incremental: bool,
) -> RegMeta:
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
    # B1（PR-2）：build_mode 决定子文件夹分配
    # - mirror：源脚本旧行为，按 train 子文件夹镜像，每个独立按图数拉
    # - flat：所有图进 1_data/ 单桶，total_target 一次满足
    if opts.build_mode == "flat":
        subfolders_plan: dict[str, dict[str, Any]] = {
            "1_data": {"image_count": total_target}
        }
        on_progress(
            f"[reg] flat 模式：目标 {total_target} 张，单桶 1_data/"
        )
    else:
        subfolders_plan = structure["subfolders"]
        on_progress(
            f"[reg] mirror 模式：目标 {total_target} 张（train 总 "
            f"{structure['total_images']}），镜像 {len(subfolders_plan)} 个子文件夹"
        )

    # 输出目录
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    # PP5.1 — incremental 时扫已有图
    pre_existing_per_sub: dict[str, dict[str, Any]] = {}
    prior_meta: Optional[RegMeta] = None
    if incremental:
        pre_existing_per_sub = collect_existing_reg_per_subfolder(opts.output_dir)
        prior_meta = read_meta(opts.output_dir)
        existing_total = sum(b["count"] for b in pre_existing_per_sub.values())
        on_progress(
            f"[reg] incremental 模式：已有 {existing_total} 张图、"
            f"{len(pre_existing_per_sub)} 个子文件夹"
        )

    # A2 — 用户从 UI 删除过的 booru ID（含跨子文件夹），无论 incremental 与否都
    # 应排除：fresh build 时 .deleted_ids.json 已被 DELETE /reg 清掉；
    # incremental 时这个集合才非空，避免补足把删除的图再拉回。
    deleted_ids = read_deleted_ids(opts.output_dir)
    if deleted_ids:
        on_progress(f"[reg] 已删 booru ID 共 {len(deleted_ids)} 个（A2 排除）")

    target_resolution = structure.get("median_resolution")
    target_aspect_ratio = structure.get("median_aspect_ratio")
    resolution_std = structure.get("resolution_std")

    total_downloaded = 0
    success_subfolder_count = 0
    for sub_name, sub_data in subfolders_plan.items():
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
                pre_existing=pre_existing_per_sub.get(sub_name),
                client=client,
                deleted_ids=deleted_ids,
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
    # incremental 时，failed_tags / source_tags / auto_tagged / runs 都基于旧 meta 合并
    if incremental and prior_meta is not None:
        merged_failed = sorted(set(failed_tags) | set(prior_meta.failed_tags))
        merged_source = sorted(set(source_tags_used) | set(prior_meta.source_tags))
        merged_excluded = sorted(set(excluded) | set(prior_meta.excluded_tags))
        runs = prior_meta.incremental_runs + 1
    else:
        merged_failed = sorted(failed_tags)
        merged_source = sorted(source_tags_used)
        merged_excluded = sorted(excluded)
        runs = 0
    meta = RegMeta(
        generated_at=time.time(),
        based_on_version=opts.based_on_version,
        api_source=opts.api_source,
        target_count=total_target,
        actual_count=total_downloaded,
        source_tags=merged_source,
        excluded_tags=list(merged_excluded),
        blacklist_tags=sorted(blacklist_tags),
        failed_tags=merged_failed,
        train_tag_distribution=top_dist,
        auto_tagged=False,  # worker 在 auto_tag 完成后改写
        incremental_runs=runs,
        build_mode=opts.build_mode,
    )
    write_meta(opts.output_dir, meta)

    on_progress(
        f"[reg] 完成: {total_downloaded}/{total_target}"
        f" 张（{success_subfolder_count}/{len(subfolders_plan)} "
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


def update_meta_auto_tagged(
    reg_dir: Path, auto_tagged: bool, kind: Optional[str] = None,
) -> None:
    """auto_tag 完成后改写 meta.auto_tagged，同时记录使用的 tagger 名。

    `kind=None` 时只动 `auto_tagged`，不动 `auto_tag_kind`（兼容旧 caller）。
    新 caller 应传 kind 一起写。
    """
    m = read_meta(reg_dir)
    if m is None:
        return
    m.auto_tagged = auto_tagged
    if kind is not None:
        m.auto_tag_kind = kind if auto_tagged else None
    write_meta(reg_dir, m)


# ---------------------------------------------------------------------------
# A2 — 用户删除黑名单（reg/.deleted_ids.json）
# ---------------------------------------------------------------------------


DELETED_IDS_FILENAME = ".deleted_ids.json"


def deleted_ids_path(reg_dir: Path) -> Path:
    return reg_dir / DELETED_IDS_FILENAME


def read_deleted_ids(reg_dir: Path) -> set[str]:
    """读 reg/.deleted_ids.json；不存在 / 损坏返回空 set。"""
    import json
    p = deleted_ids_path(reg_dir)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data if x}
    except Exception:
        pass
    return set()


def clear_reg_dir(reg_dir: Path) -> None:
    """清空 reg/ 内所有内容（图、子文件夹、meta、`.deleted_ids.json` 等），
    保留空目录本身。

    full-mode build 入口用：用户语义是「从零开始」，所以 `.deleted_ids.json`
    也一起清掉（如果想保留 deleted 偏好，应该选 incremental mode）。

    跟 `DELETE /api/projects/{pid}/versions/{vid}/reg` 端点行为一致 ——
    那个端点也是 iterdir + rmtree/unlink children。
    """
    import shutil
    if not reg_dir.exists():
        return
    for child in reg_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def append_deleted_ids(reg_dir: Path, new_ids: list[str]) -> None:
    """把 new_ids（booru ID = 文件名 stem）追加到 reg/.deleted_ids.json，去重保序。

    DELETE /reg 端点清 reg/ 时会一并清掉这个文件（按 rglob 通配删 children），
    所以 fresh build 不会看到上轮的删除黑名单。
    """
    import json
    if not new_ids:
        return
    p = deleted_ids_path(reg_dir)
    existing = read_deleted_ids(reg_dir)
    merged = sorted(existing | {str(x) for x in new_ids if x})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_meta_postprocess(
    reg_dir: Path,
    *,
    when: Optional[float],
    clusters: Optional[int],
    method: Optional[str],
    max_crop_ratio: Optional[float],
) -> None:
    """PP5.5 — 后处理完成后改写 meta 的后处理字段。"""
    m = read_meta(reg_dir)
    if m is None:
        return
    m.postprocessed_at = when
    m.postprocess_clusters = clusters
    m.postprocess_method = method
    m.postprocess_max_crop_ratio = max_crop_ratio
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
