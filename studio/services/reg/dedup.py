"""A4 — reg 集 dedup helper（PR-1）。

把 `services/preprocess/duplicates.py` 的相似度分组算法套到 reg/ 上，
返回每组里要删的相对路径；删除（含 .txt + .deleted_ids.json + meta 更新）
由 `purge_paths` 完成。

两个调用方：
- API endpoint `POST /reg/dedup-purge` —— 用户在 RegPreview 手动触发
- worker reg_build_worker —— `auto_dedup=True` 时 build 后自动跑、配合
  incremental 补足循环到达目标数

不做路径越界校验：调用方负责保证 `relative_paths` 是合法 rdir 内相对路径
（worker 自己生成的必合法；endpoint 路径走 `_safe_join_or_400`）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..dataset.scan import IMAGE_EXTS
from . import builder as reg_builder


def scan_for_dedup(reg_dir: Path) -> list[str]:
    """对 reg/ 跑相似度分组，返回每组里**非保留项**的相对路径列表。

    每组保留 `group[0]`（duplicate_finder 按文件大小 + 像素数排序，
    第一项视作"推荐保留"），其余视作"推荐删除"。无重复组返回 []。

    用默认 `DuplicateOptions()` —— 用户决策（2026-05-30）：reg 集 quality
    bar 比 train 低，不开放参数调整。
    """
    if not reg_dir.exists():
        return []
    from ..preprocess import duplicates as duplicate_finder

    sources: list[tuple[str, Path]] = []
    for f in reg_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            try:
                rel = f.relative_to(reg_dir).as_posix()
            except ValueError:
                continue
            sources.append((rel, f))
    if not sources:
        return []

    options = duplicate_finder.DuplicateOptions()
    try:
        infos = duplicate_finder.build_all_image_infos(sources, options)
        groups, _pair_metrics, _stats = duplicate_finder.group_similar_images(
            infos, options,
        )
    except duplicate_finder.DuplicateFinderError:
        return []

    to_delete: list[str] = []
    for group in groups:
        if len(group) <= 1:
            continue
        for item in group[1:]:
            to_delete.append(item.name)
    return to_delete


def purge_paths(reg_dir: Path, relative_paths: list[str]) -> dict[str, Any]:
    """按相对路径删 reg/ 下的图 + 同名 .txt caption，更新 meta.actual_count，
    booru ID（文件 stem）追加到 `reg/.deleted_ids.json`。

    路径不存在 / 非图：静默跳过。**不做** traversal 校验 —— 调用方负责。

    返回 `{deleted: [rel...], count: int}`。
    """
    deleted: list[str] = []
    deleted_booru_ids: list[str] = []
    for rel in relative_paths:
        if not rel:
            continue
        parts = [p for p in rel.replace("\\", "/").split("/") if p]
        if not parts:
            continue
        target = reg_dir.joinpath(*parts)
        if not target.exists() or target.suffix.lower() not in IMAGE_EXTS:
            continue
        booru_id = target.stem
        try:
            target.unlink()
            txt = target.with_suffix(".txt")
            if txt.exists():
                txt.unlink()
        except OSError:
            continue
        deleted.append(rel)
        deleted_booru_ids.append(booru_id)

    if deleted_booru_ids:
        reg_builder.append_deleted_ids(reg_dir, deleted_booru_ids)

    meta = reg_builder.read_meta(reg_dir)
    if meta is not None and deleted:
        meta.actual_count = max(0, meta.actual_count - len(deleted))
        reg_builder.write_meta(reg_dir, meta)

    return {"deleted": deleted, "count": len(deleted)}
