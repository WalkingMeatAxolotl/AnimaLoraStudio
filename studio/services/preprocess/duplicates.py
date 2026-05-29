"""Duplicate and same-scene variant review for project curation.

This module adapts the standalone duplicate finder into a preprocess review
tool: scan first, return explicit review groups, and only mark audited images
as skipped in the preprocess manifest after confirmation.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ...services.projects import projects
from ...services.dataset import curation
from ...services.dataset.scan import IMAGE_EXTS
from . import manifest as preprocess_manifest

try:  # pragma: no cover - exercised by dependency checks in integration tests
    import imagehash
except Exception as exc:  # noqa: BLE001
    imagehash = None  # type: ignore[assignment]
    _IMAGEHASH_IMPORT_ERROR = exc
else:
    _IMAGEHASH_IMPORT_ERROR = None


RESAMPLE_FILTER = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

DEFAULT_HASH_SIZE = 768
DEFAULT_HASH_WORKERS = max(2, min(16, (os.cpu_count() or 8) // 2))
DEFAULT_TILE_GRIDS = (4, 6)

DEFAULT_STRUCTURE_THRESHOLD = 6
DEFAULT_VARIANT_SCORE = 72.0
DEFAULT_ASPECT_TOLERANCE = 0.045
DEFAULT_MIN_CLOSE_TILES = 0.48
DEFAULT_TILE_MEDIAN = 14.0
DEFAULT_EDGE_THRESHOLD = 24
DEFAULT_MIN_GRAY_CLOSE = 0.42
DEFAULT_GRAY_CLOSE_THRESHOLD = 22
DEFAULT_GRAYPRINT_SIZE = 96
DEFAULT_PREFILTER_PHASH = 22
DEFAULT_PREFILTER_DHASH = 20
DEFAULT_PREFILTER_AHASH = 22
DEFAULT_COLOR_ALERT = 14


MatchScope = Literal["strict", "both"]


from studio.domain.errors import DomainError


class DuplicateFinderError(DomainError):
    """Duplicate finder business error.

    PR-2 C3 加 DomainError base — handler 自动翻 dual-write envelope。
    """
    default_code = "duplicate.error"


@dataclass(frozen=True)
class DuplicateOptions:
    match_scope: MatchScope = "both"
    hash_size: int = DEFAULT_HASH_SIZE
    hash_workers: int = DEFAULT_HASH_WORKERS
    tile_grids: tuple[int, ...] = DEFAULT_TILE_GRIDS
    structure_threshold: int = DEFAULT_STRUCTURE_THRESHOLD
    variant_score: float = DEFAULT_VARIANT_SCORE
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE
    min_close_tiles: float = DEFAULT_MIN_CLOSE_TILES
    tile_median: float = DEFAULT_TILE_MEDIAN
    min_gray_close: float = DEFAULT_MIN_GRAY_CLOSE


@dataclass
class ImageInfo:
    name: str
    path: Path
    width: int
    height: int
    size: int
    phash: Any
    soft_phash: Any
    dhash: Any
    ahash: Any
    colorhash: Any
    grayprint: np.ndarray
    edgehash: Any | None = None
    tilehashes: list[Any] | None = None

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1, self.height)


@dataclass
class PairMetrics:
    score: float
    match_type: str
    structure_diff: int
    phash_diff: int
    soft_phash_diff: int
    dhash_diff: int
    ahash_diff: int
    edge_diff: int
    color_diff: int
    tile_median: float
    tile_mean: float
    tile_close_ratio: float
    aspect_delta: float
    note: str
    gray_diff: float = -1.0
    gray_close_ratio: float = 0.0

    @property
    def is_match(self) -> bool:
        return self.match_type != "different"


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _require_imagehash() -> None:
    if imagehash is None:
        raise DuplicateFinderError(
            "ImageHash is not installed. Please install imagehash>=4.3.0."
        ) from _IMAGEHASH_IMPORT_ERROR


def parse_tile_grids(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        parts = str(value).replace(";", ",").split(",")
    grids: list[int] = []
    for raw in parts:
        part = str(raw).strip()
        if not part:
            continue
        try:
            grid = int(part)
        except ValueError as exc:
            raise DuplicateFinderError(f"invalid tile grid: {part}") from exc
        if grid < 2 or grid > 12:
            raise DuplicateFinderError("tile grids must be between 2 and 12")
        if grid not in grids:
            grids.append(grid)
    if not grids:
        raise DuplicateFinderError("at least one tile grid is required")
    return tuple(grids)


def options_from_payload(payload: dict[str, Any]) -> DuplicateOptions:
    match_scope = payload.get("match_scope", "both")
    if match_scope not in ("strict", "both"):
        raise DuplicateFinderError(f"invalid match scope: {match_scope!r}")
    return DuplicateOptions(
        match_scope=match_scope,
        hash_size=max(0, int(payload.get("hash_size", DEFAULT_HASH_SIZE))),
        hash_workers=max(1, min(32, int(payload.get("hash_workers", DEFAULT_HASH_WORKERS)))),
        tile_grids=parse_tile_grids(payload.get("tile_grids", DEFAULT_TILE_GRIDS)),
        structure_threshold=max(0, int(payload.get("structure_threshold", DEFAULT_STRUCTURE_THRESHOLD))),
        variant_score=float(payload.get("variant_score", DEFAULT_VARIANT_SCORE)),
        aspect_tolerance=max(0.0001, float(payload.get("aspect_tolerance", DEFAULT_ASPECT_TOLERANCE))),
        min_close_tiles=max(0.0, min(1.0, float(payload.get("min_close_tiles", DEFAULT_MIN_CLOSE_TILES)))),
        tile_median=max(0.0, float(payload.get("tile_median", DEFAULT_TILE_MEDIAN))),
        min_gray_close=max(0.0, min(1.0, float(payload.get("min_gray_close", DEFAULT_MIN_GRAY_CLOSE)))),
    )


def scan_project_duplicates(
    conn,
    project_id: int,
    options: DuplicateOptions,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _require_imagehash()
    project, project_dir = curation._project_dir(conn, project_id)  # noqa: SLF001
    sources = _resolve_download_sources(conn, project_id, project_dir)
    if on_progress:
        on_progress({
            "stage": "hashing",
            "idx": 0,
            "total": len(sources),
            "text": f"Preparing hashes for {len(sources)} images...",
        })
    started = time.monotonic()
    infos = build_all_image_infos(sources, options, on_progress=on_progress)
    if on_progress:
        total_pairs = len(infos) * (len(infos) - 1) // 2
        on_progress({
            "stage": "comparing",
            "idx": 0,
            "total": total_pairs,
            "text": f"Comparing {total_pairs} image pairs...",
        })
    groups, pair_metrics, stats = group_similar_images(
        infos,
        options,
        on_progress=on_progress,
    )
    elapsed = time.monotonic() - started

    return {
        "target": "preprocess",
        "match_scope": options.match_scope,
        "total_images": len(sources),
        "readable_images": len(infos),
        "group_count": len(groups),
        "candidate_count": sum(max(0, len(g) - 1) for g in groups),
        "elapsed_seconds": round(elapsed, 3),
        "options": options_to_json(options),
        "stats": stats,
        "groups": [
            _group_to_json(index, group, pair_metrics)
            for index, group in enumerate(groups, start=1)
        ],
    }


def apply_duplicate_removals(
    conn,
    project_id: int,
    *,
    names: list[str],
) -> dict[str, Any]:
    project = projects.get_project(conn, project_id)
    if not project:
        raise DuplicateFinderError(f"project not found: id={project_id}")

    project_dir = projects.project_dir(project["id"], project["slug"])
    for raw_name in names:
        curation._validate_filename(raw_name)  # noqa: SLF001
    return preprocess_manifest.mark_duplicate_removed(project_dir, names)


def options_to_json(options: DuplicateOptions) -> dict[str, Any]:
    return {
        "match_scope": options.match_scope,
        "hash_size": options.hash_size,
        "hash_workers": options.hash_workers,
        "tile_grids": list(options.tile_grids),
        "structure_threshold": options.structure_threshold,
        "variant_score": options.variant_score,
        "aspect_tolerance": options.aspect_tolerance,
        "min_close_tiles": options.min_close_tiles,
        "tile_median": options.tile_median,
        "min_gray_close": options.min_gray_close,
    }


def _resolve_download_sources(
    conn,
    project_id: int,
    project_dir: Path,
) -> list[tuple[str, Path]]:
    names = [item["name"] for item in curation.list_download(conn, project_id)]

    sources: list[tuple[str, Path]] = []
    for name in names:
        curation._validate_filename(name)  # noqa: SLF001
        if Path(name).suffix.lower() not in IMAGE_EXTS:
            continue
        entry = preprocess_manifest.get_entry(project_dir, name)
        if preprocess_manifest.is_duplicate_removed_entry(entry):
            continue
        if entry is not None:
            path = project_dir / "preprocess" / name
        else:
            path = project_dir / "download" / name
        if path.is_file():
            sources.append((name, path))
    return sorted(sources, key=lambda item: item[0].lower())


def oriented_size(img: Image.Image) -> tuple[int, int]:
    width, height = img.size
    try:
        orientation = img.getexif().get(274)
    except Exception:
        orientation = None
    if orientation in {5, 6, 7, 8}:
        return height, width
    return width, height


def load_hash_image(path: Path, max_side: int) -> tuple[Image.Image, int, int]:
    with Image.open(path) as img:
        width, height = oriented_size(img)
        if max_side > 0:
            try:
                img.draft("RGB", (max_side, max_side))
            except Exception:
                pass
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        if max_side > 0:
            rgb.thumbnail((max_side, max_side), RESAMPLE_FILTER)
        return rgb.copy(), width, height


def build_image_info(source: tuple[str, Path], options: DuplicateOptions) -> ImageInfo | None:
    name, path = source
    try:
        rgb, width, height = load_hash_image(path, options.hash_size)
        gray = rgb.convert("L")
        soft_gray = gray.filter(ImageFilter.GaussianBlur(radius=2))
        return ImageInfo(
            name=name,
            path=path,
            width=width,
            height=height,
            size=path.stat().st_size,
            phash=imagehash.phash(gray),
            soft_phash=imagehash.phash(soft_gray),
            dhash=imagehash.dhash(gray),
            ahash=imagehash.average_hash(soft_gray),
            colorhash=imagehash.colorhash(rgb),
            grayprint=build_grayprint(gray),
            edgehash=build_edge_hash(gray),
            tilehashes=build_tile_hashes(soft_gray, options.tile_grids),
        )
    except Exception:
        return None


def build_all_image_infos(
    sources: list[tuple[str, Path]],
    options: DuplicateOptions,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[ImageInfo]:
    workers = max(1, int(options.hash_workers or 1))
    images: list[ImageInfo] = []
    total = len(sources)
    if workers == 1:
        for idx, source in enumerate(sources, start=1):
            info = build_image_info(source, options)
            if info:
                images.append(info)
            if on_progress:
                on_progress({
                    "stage": "hashing",
                    "idx": idx,
                    "total": total,
                    "name": source[0],
                    "text": f"Hashed {idx}/{total}: {source[0]}",
                })
        return images

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(build_image_info, source, options): source for source in sources}
        for idx, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            info = future.result()
            if info:
                images.append(info)
            if on_progress:
                on_progress({
                    "stage": "hashing",
                    "idx": idx,
                    "total": total,
                    "name": source[0],
                    "text": f"Hashed {idx}/{total}: {source[0]}",
                })
    return sorted(images, key=lambda item: item.name.lower())


def build_edge_hash(gray: Image.Image) -> Any:
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    return imagehash.phash(edges)


def build_grayprint(gray: Image.Image) -> np.ndarray:
    small = gray.resize((DEFAULT_GRAYPRINT_SIZE, DEFAULT_GRAYPRINT_SIZE), RESAMPLE_FILTER)
    return np.asarray(small, dtype=np.int16).reshape(-1)


def build_tile_hashes(gray: Image.Image, grids: tuple[int, ...]) -> list[Any]:
    width, height = gray.size
    hashes: list[Any] = []
    for grid in grids:
        for y in range(grid):
            for x in range(grid):
                left = x * width // grid
                upper = y * height // grid
                right = (x + 1) * width // grid
                lower = (y + 1) * height // grid
                hashes.append(imagehash.phash(gray.crop((left, upper, right, lower))))
    return hashes


def hash_distance(a: Any, b: Any) -> int:
    return int(a - b)


def bounded_similarity(distance: float, max_distance: float) -> float:
    if max_distance <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - distance / max_distance))


def aspect_delta(a: ImageInfo, b: ImageInfo) -> float:
    if a.aspect_ratio <= 0 or b.aspect_ratio <= 0:
        return 1.0
    return abs(math.log(a.aspect_ratio / b.aspect_ratio))


def tile_metrics(a: ImageInfo, b: ImageInfo) -> tuple[float, float, float]:
    distances = [hash_distance(ha, hb) for ha, hb in zip(a.tilehashes or [], b.tilehashes or [])]
    if not distances:
        return 999.0, 999.0, 0.0
    close_count = sum(1 for value in distances if value <= 8)
    return float(median(distances)), float(mean(distances)), close_count / len(distances)


def grayprint_metrics(a: ImageInfo, b: ImageInfo) -> tuple[float, float]:
    if a.grayprint is None or b.grayprint is None:
        return 999.0, 0.0
    arr_a = a.grayprint
    arr_b = b.grayprint
    if arr_a.size != arr_b.size or arr_a.size == 0:
        return 999.0, 0.0
    diffs = np.abs(arr_a - arr_b)
    return float(diffs.mean()), float(np.count_nonzero(diffs <= DEFAULT_GRAY_CLOSE_THRESHOLD) / diffs.size)


def prefilter_metrics(a: ImageInfo, b: ImageInfo) -> tuple[int, int, int, int, int, float]:
    phash_diff = hash_distance(a.phash, b.phash)
    soft_phash_diff = hash_distance(a.soft_phash, b.soft_phash)
    dhash_diff = hash_distance(a.dhash, b.dhash)
    ahash_diff = hash_distance(a.ahash, b.ahash)
    structure_diff = min(phash_diff, soft_phash_diff, dhash_diff, ahash_diff)
    ratio_delta = aspect_delta(a, b)
    return phash_diff, soft_phash_diff, dhash_diff, ahash_diff, structure_diff, ratio_delta


def should_do_expensive_compare(a: ImageInfo, b: ImageInfo, options: DuplicateOptions) -> bool:
    phash_diff, soft_phash_diff, dhash_diff, ahash_diff, structure_diff, ratio_delta = prefilter_metrics(a, b)
    if ratio_delta > options.aspect_tolerance:
        return False
    if structure_diff <= options.structure_threshold:
        return True
    if min(phash_diff, soft_phash_diff) <= DEFAULT_PREFILTER_PHASH:
        return True
    if dhash_diff <= DEFAULT_PREFILTER_DHASH or ahash_diff <= DEFAULT_PREFILTER_AHASH:
        return True
    return False


def compare_images(a: ImageInfo, b: ImageInfo, options: DuplicateOptions) -> PairMetrics:
    phash_diff, soft_phash_diff, dhash_diff, ahash_diff, structure_diff, ratio_delta = prefilter_metrics(a, b)
    edge_diff = hash_distance(a.edgehash, b.edgehash)
    color_diff = hash_distance(a.colorhash, b.colorhash)
    tile_median_value, tile_mean_value, tile_close_ratio = tile_metrics(a, b)
    gray_diff_value, gray_close_ratio = grayprint_metrics(a, b)

    strict_duplicate = structure_diff <= options.structure_threshold and ratio_delta <= options.aspect_tolerance
    structure_score = bounded_similarity(structure_diff, 28)
    edge_score = bounded_similarity(edge_diff, 30)
    tile_median_score = bounded_similarity(tile_median_value, 22)
    gray_score = 0.55 * bounded_similarity(gray_diff_value, 55) + 0.45 * gray_close_ratio
    aspect_score = bounded_similarity(ratio_delta, options.aspect_tolerance * 2)
    score = 100 * (
        0.25 * structure_score
        + 0.18 * edge_score
        + 0.32 * tile_close_ratio
        + 0.10 * tile_median_score
        + 0.10 * gray_score
        + 0.05 * aspect_score
    )

    local_structure_match = (
        tile_close_ratio >= options.min_close_tiles
        or edge_diff <= DEFAULT_EDGE_THRESHOLD
        or gray_close_ratio >= options.min_gray_close
    )
    variant_match = (
        ratio_delta <= options.aspect_tolerance
        and score >= options.variant_score
        and tile_median_value <= options.tile_median
        and local_structure_match
    )

    notes: list[str] = []
    if strict_duplicate:
        match_type = "strict-duplicate"
        notes.append("whole-image structure is very close")
    elif variant_match:
        match_type = "same-scene-variant"
        notes.append("likely same-scene variant")
    else:
        match_type = "different"

    if tile_close_ratio >= 0.75:
        notes.append("most tiles match")
    elif tile_close_ratio >= options.min_close_tiles:
        notes.append("some tiles match")
    if edge_diff <= DEFAULT_EDGE_THRESHOLD:
        notes.append("edge composition is close")
    if gray_close_ratio >= 0.65:
        notes.append("grayprint largely matches")
    elif gray_close_ratio >= options.min_gray_close:
        notes.append("grayprint partially matches")
    if color_diff >= DEFAULT_COLOR_ALERT:
        notes.append("color differs")
    if ratio_delta > options.aspect_tolerance:
        notes.append("aspect ratio differs")

    return PairMetrics(
        score=round(score, 2),
        match_type=match_type,
        structure_diff=structure_diff,
        phash_diff=phash_diff,
        soft_phash_diff=soft_phash_diff,
        dhash_diff=dhash_diff,
        ahash_diff=ahash_diff,
        edge_diff=edge_diff,
        color_diff=color_diff,
        tile_median=round(tile_median_value, 2),
        tile_mean=round(tile_mean_value, 2),
        tile_close_ratio=round(tile_close_ratio, 3),
        aspect_delta=round(ratio_delta, 4),
        note="; ".join(notes) if notes else "not similar enough",
        gray_diff=round(gray_diff_value, 2),
        gray_close_ratio=round(gray_close_ratio, 3),
    )


def match_in_scope(metrics: PairMetrics, match_scope: MatchScope) -> bool:
    if match_scope == "strict":
        return metrics.match_type == "strict-duplicate"
    return metrics.is_match


def group_similar_images(
    images: list[ImageInfo],
    options: DuplicateOptions,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[list[ImageInfo]], dict[tuple[str, str], PairMetrics], dict[str, int]]:
    uf = UnionFind([img.name for img in images])
    pair_metrics: dict[tuple[str, str], PairMetrics] = {}
    total_pairs = len(images) * (len(images) - 1) // 2
    skipped_pairs = 0
    prefiltered_pairs = 0

    compared_so_far = 0
    for i, img_a in enumerate(images):
        for img_b in images[i + 1 :]:
            compared_so_far += 1
            if aspect_delta(img_a, img_b) > max(options.aspect_tolerance * 3.33, 0.15):
                skipped_pairs += 1
            elif not should_do_expensive_compare(img_a, img_b, options):
                prefiltered_pairs += 1
            else:
                metrics = compare_images(img_a, img_b, options)
                if match_in_scope(metrics, options.match_scope):
                    uf.union(img_a.name, img_b.name)
                    pair_metrics[(img_a.name, img_b.name)] = metrics
        if on_progress:
            on_progress({
                "stage": "comparing",
                "idx": compared_so_far,
                "total": total_pairs,
                "name": img_a.name,
                "text": f"Compared {compared_so_far}/{total_pairs} pairs...",
            })

    grouped: dict[str, list[ImageInfo]] = {}
    for img in images:
        grouped.setdefault(uf.find(img.name), []).append(img)
    groups = [sort_group_for_keep(group) for group in grouped.values() if len(group) > 1]
    groups.sort(key=lambda group: (-len(group), group[0].name.lower()))
    return groups, pair_metrics, {
        "total_pairs": total_pairs,
        "aspect_skipped_pairs": skipped_pairs,
        "prefiltered_pairs": prefiltered_pairs,
        "compared_pairs": total_pairs - skipped_pairs - prefiltered_pairs,
    }


def sort_group_for_keep(group: list[ImageInfo]) -> list[ImageInfo]:
    return sorted(group, key=lambda x: (x.pixels, x.size, x.name.lower()), reverse=True)


def _group_to_json(
    group_id: int,
    group: list[ImageInfo],
    pair_metrics: dict[tuple[str, str], PairMetrics],
) -> dict[str, Any]:
    keep = group[0]
    return {
        "group_id": group_id,
        "keep": keep.name,
        "items": [
            _item_to_json(item, keep, pair_metrics)
            for item in group
        ],
        "best": _metrics_to_json(best_pair_metrics(group, pair_metrics)),
    }


def _item_to_json(
    item: ImageInfo,
    keep: ImageInfo,
    pair_metrics: dict[tuple[str, str], PairMetrics],
) -> dict[str, Any]:
    metrics = None if item.name == keep.name else get_pair_metrics(pair_metrics, keep.name, item.name)
    if metrics is None and item.name != keep.name:
        metrics = best_link_to_group(item, pair_metrics)
    return {
        "name": item.name,
        "keep": item.name == keep.name,
        "width": item.width,
        "height": item.height,
        "filesize_kb": item.size // 1024,
        "metrics": _metrics_to_json(metrics),
    }


def _metrics_to_json(metrics: PairMetrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        "score": metrics.score,
        "match_type": metrics.match_type,
        "structure_diff": metrics.structure_diff,
        "phash_diff": metrics.phash_diff,
        "soft_phash_diff": metrics.soft_phash_diff,
        "dhash_diff": metrics.dhash_diff,
        "ahash_diff": metrics.ahash_diff,
        "edge_diff": metrics.edge_diff,
        "color_diff": metrics.color_diff,
        "tile_median": metrics.tile_median,
        "tile_mean": metrics.tile_mean,
        "tile_close_ratio": metrics.tile_close_ratio,
        "gray_diff": metrics.gray_diff,
        "gray_close_ratio": metrics.gray_close_ratio,
        "aspect_delta": metrics.aspect_delta,
        "note": metrics.note,
    }


def best_pair_metrics(
    group: list[ImageInfo],
    pair_metrics: dict[tuple[str, str], PairMetrics],
) -> PairMetrics | None:
    best: PairMetrics | None = None
    for i, img_a in enumerate(group):
        for img_b in group[i + 1 :]:
            metrics = get_pair_metrics(pair_metrics, img_a.name, img_b.name)
            if metrics and (best is None or metrics.score > best.score):
                best = metrics
    return best


def get_pair_metrics(
    pair_metrics: dict[tuple[str, str], PairMetrics],
    a: str,
    b: str,
) -> PairMetrics | None:
    return pair_metrics.get((a, b)) or pair_metrics.get((b, a))


def best_link_to_group(
    item: ImageInfo,
    pair_metrics: dict[tuple[str, str], PairMetrics],
) -> PairMetrics | None:
    best: PairMetrics | None = None
    for (a, b), metrics in pair_metrics.items():
        if item.name in (a, b) and (best is None or metrics.score > best.score):
            best = metrics
    return best
