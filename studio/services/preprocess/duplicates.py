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

DEFAULT_DETECT_BLUR = False
DEFAULT_BLUR_SCORE_THRESHOLD = 80.0
DEFAULT_BLUR_LOCAL_RATIO = 0.06
DEFAULT_BLUR_GRID = 12
DEFAULT_BLUR_TILE_LAPLACIAN = 45.0
DEFAULT_BLUR_TILE_STD = 30.0
DEFAULT_BLUR_BACKGROUND_VALUE = 245.0
DEFAULT_BLUR_BACKGROUND_SATURATION = 18.0

DEFAULT_DETECT_CROPS = False
DEFAULT_CROP_SCORE = 0.74
DEFAULT_CROP_HASH_THRESHOLD = 30
DEFAULT_CROP_MAX_SIDE = 384
DEFAULT_CROP_PREFILTER_SEGMENTS = 1
DEFAULT_CROP_PREFILTER_BIT_ERROR_RATE = 0.25
DEFAULT_CROP_MIN_WINDOW_RATIO = 0.12
DEFAULT_CROP_MAX_WINDOW_RATIO = 0.92
DEFAULT_CROP_SCAN_DIVISIONS = 14
DEFAULT_CROP_SCALES = (0.35, 0.45, 0.55, 0.60, 0.65, 0.75, 0.85, 0.92)


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
    detect_blur: bool = DEFAULT_DETECT_BLUR
    blur_score_threshold: float = DEFAULT_BLUR_SCORE_THRESHOLD
    blur_local_ratio: float = DEFAULT_BLUR_LOCAL_RATIO
    detect_crops: bool = DEFAULT_DETECT_CROPS
    crop_score: float = DEFAULT_CROP_SCORE
    crop_hash_threshold: int = DEFAULT_CROP_HASH_THRESHOLD
    crop_max_side: int = DEFAULT_CROP_MAX_SIDE


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
    blur_score: float = 0.0
    local_blur_ratio: float = 0.0
    largest_blur_region_ratio: float = 0.0
    crop_hash: Any | None = None
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


@dataclass
class BlurCandidate:
    image: ImageInfo
    reason: str


@dataclass
class CropRelation:
    source: ImageInfo
    crop: ImageInfo
    score: float
    source_window: tuple[int, int, int, int]
    window_ratio: float
    segment_matches: int
    segment_coverage: float
    note: str

    @property
    def source_area(self) -> int:
        return self.source.width * self.source.height

    @property
    def crop_area(self) -> int:
        return self.crop.width * self.crop.height

    @property
    def area_ratio(self) -> float:
        smaller = max(1, min(self.source_area, self.crop_area))
        larger = max(self.source_area, self.crop_area)
        return round(larger / smaller, 4)

    @property
    def larger_image(self) -> str:
        if self.source_area > self.crop_area:
            return self.source.name
        if self.crop_area > self.source_area:
            return self.crop.name
        return "same_area"

    @property
    def relation_kind(self) -> str:
        if self.crop_area > self.source_area:
            return "crop_upscaled"
        if self.crop_area < self.source_area:
            return "crop_smaller"
        return "crop_same_area"


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
        detect_blur=bool(payload.get("detect_blur", DEFAULT_DETECT_BLUR)),
        blur_score_threshold=max(0.0, float(payload.get("blur_score_threshold", DEFAULT_BLUR_SCORE_THRESHOLD))),
        blur_local_ratio=max(0.0, min(1.0, float(payload.get("blur_local_ratio", DEFAULT_BLUR_LOCAL_RATIO)))),
        detect_crops=bool(payload.get("detect_crops", DEFAULT_DETECT_CROPS)),
        crop_score=max(0.0, min(1.0, float(payload.get("crop_score", DEFAULT_CROP_SCORE)))),
        crop_hash_threshold=max(0, int(payload.get("crop_hash_threshold", DEFAULT_CROP_HASH_THRESHOLD))),
        crop_max_side=max(64, int(payload.get("crop_max_side", DEFAULT_CROP_MAX_SIDE))),
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
    blur_candidates = find_blur_candidates(infos, options) if options.detect_blur else []
    crop_relations: list[CropRelation] = []
    if options.detect_crops:
        if on_progress:
            total_pairs = len(infos) * (len(infos) - 1) // 2
            on_progress({
                "stage": "crop_relations",
                "idx": 0,
                "total": total_pairs,
                "text": f"Checking {total_pairs} crop/scale relation pairs...",
            })
        crop_relations = find_crop_relations(infos, options, on_progress=on_progress)
    elapsed = time.monotonic() - started

    return {
        "target": "preprocess",
        "match_scope": options.match_scope,
        "total_images": len(sources),
        "readable_images": len(infos),
        "group_count": len(groups),
        "candidate_count": sum(max(0, len(g) - 1) for g in groups),
        "blur_candidate_count": len(blur_candidates),
        "crop_relation_count": len(crop_relations),
        "elapsed_seconds": round(elapsed, 3),
        "options": options_to_json(options),
        "stats": stats,
        "groups": [
            _group_to_json(index, group, pair_metrics)
            for index, group in enumerate(groups, start=1)
        ],
        "blur_candidates": [
            _blur_candidate_to_json(candidate)
            for candidate in blur_candidates
        ],
        "crop_relations": [
            _crop_relation_to_json(relation)
            for relation in crop_relations
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
        "detect_blur": options.detect_blur,
        "blur_score_threshold": options.blur_score_threshold,
        "blur_local_ratio": options.blur_local_ratio,
        "detect_crops": options.detect_crops,
        "crop_score": options.crop_score,
        "crop_hash_threshold": options.crop_hash_threshold,
        "crop_max_side": options.crop_max_side,
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
        if options.detect_blur:
            blur_score_value, local_blur_ratio_value, largest_blur_region_ratio = blur_metrics(rgb, gray)
        else:
            blur_score_value, local_blur_ratio_value, largest_blur_region_ratio = 0.0, 0.0, 0.0
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
            blur_score=blur_score_value,
            local_blur_ratio=local_blur_ratio_value,
            largest_blur_region_ratio=largest_blur_region_ratio,
            crop_hash=build_crop_hash(rgb) if options.detect_crops else None,
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


def build_laplacian(gray: Image.Image) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float32)
    lap = -4 * arr
    lap[:-1, :] += arr[1:, :]
    lap[1:, :] += arr[:-1, :]
    lap[:, :-1] += arr[:, 1:]
    lap[:, 1:] += arr[:, :-1]
    return lap


def largest_connected_region(cells: set[tuple[int, int]]) -> int:
    seen: set[tuple[int, int]] = set()
    largest = 0
    for cell in cells:
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        size = 0
        while stack:
            x, y = stack.pop()
            size += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (x + dx, y + dy)
                if neighbor in cells and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        largest = max(largest, size)
    return largest


def blur_metrics(rgb: Image.Image, gray: Image.Image) -> tuple[float, float, float]:
    laplacian = build_laplacian(gray)
    score = float(laplacian.var())
    gray_arr = np.asarray(gray, dtype=np.float32)
    hsv = rgb.convert("HSV")
    saturation = np.asarray(hsv.split()[1], dtype=np.float32)
    value = np.asarray(hsv.split()[2], dtype=np.float32)

    width, height = gray.size
    flagged: set[tuple[int, int]] = set()
    grid = DEFAULT_BLUR_GRID
    for tile_y in range(grid):
        for tile_x in range(grid):
            left = tile_x * width // grid
            right = (tile_x + 1) * width // grid
            upper = tile_y * height // grid
            lower = (tile_y + 1) * height // grid
            tile_laplacian = laplacian[upper:lower, left:right]
            tile_gray = gray_arr[upper:lower, left:right]
            tile_saturation = saturation[upper:lower, left:right]
            tile_value = value[upper:lower, left:right]
            if tile_laplacian.size == 0:
                continue

            low_detail = (
                float(tile_laplacian.var()) <= DEFAULT_BLUR_TILE_LAPLACIAN
                and float(tile_gray.std()) <= DEFAULT_BLUR_TILE_STD
            )
            not_plain_light_background = (
                float(tile_value.mean()) < DEFAULT_BLUR_BACKGROUND_VALUE
                or float(tile_saturation.mean()) > DEFAULT_BLUR_BACKGROUND_SATURATION
            )
            if low_detail and not_plain_light_background:
                flagged.add((tile_x, tile_y))

    total_tiles = max(1, grid * grid)
    local_ratio = len(flagged) / total_tiles
    largest_ratio = largest_connected_region(flagged) / total_tiles
    return round(score, 3), round(local_ratio, 4), round(largest_ratio, 4)


def build_crop_hash(rgb: Image.Image) -> Any | None:
    if not hasattr(imagehash, "crop_resistant_hash"):
        return None
    try:
        return imagehash.crop_resistant_hash(
            rgb,
            segmentation_image_size=300,
            min_segment_size=500,
        )
    except Exception:
        return None


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


def find_blur_candidates(images: list[ImageInfo], options: DuplicateOptions) -> list[BlurCandidate]:
    candidates: list[BlurCandidate] = []
    for image in images:
        reasons: list[str] = []
        if image.blur_score <= options.blur_score_threshold:
            reasons.append(f"global blur score {image.blur_score:.1f}")
        if image.largest_blur_region_ratio >= options.blur_local_ratio:
            reasons.append(f"connected low-detail region {image.largest_blur_region_ratio:.1%}")
        elif image.local_blur_ratio >= options.blur_local_ratio * 1.8:
            reasons.append(f"total low-detail area {image.local_blur_ratio:.1%}")
        if reasons:
            candidates.append(BlurCandidate(image=image, reason="; ".join(reasons)))
    candidates.sort(
        key=lambda item: (
            -item.image.largest_blur_region_ratio,
            -item.image.local_blur_ratio,
            item.image.blur_score,
            item.image.name.lower(),
        )
    )
    return candidates


def crop_hash_summary(a: ImageInfo, b: ImageInfo) -> tuple[int, float, str]:
    best_matches = 0
    best_coverage = 0.0
    notes: list[str] = []
    for left, right, label in ((a, b, f"{a.name}->segments"), (b, a, f"{b.name}->segments")):
        if left.crop_hash is None or right.crop_hash is None:
            continue
        segment_hashes = getattr(left.crop_hash, "segment_hashes", [])
        if not segment_hashes:
            continue
        matches, total_distance = left.crop_hash.hash_diff(
            right.crop_hash,
            bit_error_rate=DEFAULT_CROP_PREFILTER_BIT_ERROR_RATE,
        )
        coverage = matches / max(1, len(segment_hashes))
        notes.append(f"{label}: matches={matches}, coverage={coverage:.2f}, distance={total_distance}")
        if matches > best_matches or (matches == best_matches and coverage > best_coverage):
            best_matches = matches
            best_coverage = coverage
    return best_matches, best_coverage, "; ".join(notes)


def should_check_crop_relation(a: ImageInfo, b: ImageInfo, options: DuplicateOptions) -> tuple[bool, int, float, str]:
    phash_diff, soft_phash_diff, dhash_diff, ahash_diff, structure_diff, ratio_delta = prefilter_metrics(a, b)
    if structure_diff <= options.structure_threshold and ratio_delta <= options.aspect_tolerance:
        return False, 0, 0.0, "whole-image duplicate handled by review groups"

    segment_matches, segment_coverage, hash_note = crop_hash_summary(a, b)
    hash_close = min(phash_diff, soft_phash_diff, dhash_diff, ahash_diff) <= options.crop_hash_threshold
    if segment_matches >= DEFAULT_CROP_PREFILTER_SEGMENTS or hash_close:
        note_parts = [
            f"hash_diff={structure_diff}",
            f"phash={phash_diff}",
            f"dhash={dhash_diff}",
            f"ahash={ahash_diff}",
        ]
        if hash_note:
            note_parts.append(hash_note)
        return True, segment_matches, segment_coverage, "; ".join(note_parts)
    return False, segment_matches, segment_coverage, hash_note or "crop prefilter not close"


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return 0.0
    arr_a = a - float(a.mean())
    arr_b = b - float(b.mean())
    denom = math.sqrt(float(np.sum(arr_a * arr_a)) * float(np.sum(arr_b * arr_b)))
    if denom <= 1e-6:
        return 0.0
    return float(np.sum(arr_a * arr_b) / denom)


def resize_array(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(img.resize((width, height), RESAMPLE_FILTER), dtype=np.float32)


def crop_match_arrays(
    info: ImageInfo,
    options: DuplicateOptions,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    cached = cache.get(info.name)
    if cached is not None:
        return cached

    rgb, _, _ = load_hash_image(info.path, options.crop_max_side)
    gray = ImageOps.autocontrast(rgb.convert("L"))
    edges = ImageOps.autocontrast(gray.filter(ImageFilter.FIND_EDGES))
    arrays = (
        np.asarray(gray, dtype=np.float32),
        np.asarray(edges, dtype=np.float32),
    )
    cache[info.name] = arrays
    return arrays


def crop_window_score(
    source_gray: np.ndarray,
    source_edges: np.ndarray,
    crop_gray: np.ndarray,
    crop_edges: np.ndarray,
) -> tuple[float, tuple[int, int, int, int], float]:
    source_height, source_width = source_gray.shape
    crop_height, crop_width = crop_gray.shape
    if source_width <= 0 or source_height <= 0 or crop_width <= 0 or crop_height <= 0:
        return 0.0, (0, 0, 0, 0), 0.0

    crop_aspect = crop_width / max(1, crop_height)
    best_score = 0.0
    best_box = (0, 0, 0, 0)
    best_ratio = 0.0

    for scale in DEFAULT_CROP_SCALES:
        if crop_aspect >= source_width / max(1, source_height):
            window_width = int(source_width * scale)
            window_height = int(window_width / crop_aspect)
        else:
            window_height = int(source_height * scale)
            window_width = int(window_height * crop_aspect)

        if window_width <= 16 or window_height <= 16:
            continue
        if window_width > source_width or window_height > source_height:
            continue

        window_ratio = (window_width * window_height) / max(1, source_width * source_height)
        if window_ratio < DEFAULT_CROP_MIN_WINDOW_RATIO or window_ratio > DEFAULT_CROP_MAX_WINDOW_RATIO:
            continue

        target_gray = resize_array(crop_gray, (window_width, window_height))
        target_edges = resize_array(crop_edges, (window_width, window_height))
        step = max(4, min(window_width, window_height) // max(1, DEFAULT_CROP_SCAN_DIVISIONS))
        x_values = list(range(0, max(1, source_width - window_width + 1), step))
        y_values = list(range(0, max(1, source_height - window_height + 1), step))
        if not x_values or x_values[-1] != source_width - window_width:
            x_values.append(source_width - window_width)
        if not y_values or y_values[-1] != source_height - window_height:
            y_values.append(source_height - window_height)

        scale_best_score = 0.0
        scale_best_xy = (0, 0)
        for y in y_values:
            for x in x_values:
                score = crop_similarity_at(
                    source_gray,
                    source_edges,
                    target_gray,
                    target_edges,
                    x,
                    y,
                    window_width,
                    window_height,
                )
                if score > scale_best_score:
                    scale_best_score = score
                    scale_best_xy = (x, y)

        refine_radius = max(1, step)
        refine_step = max(1, step // 4)
        refine_x0 = max(0, scale_best_xy[0] - refine_radius)
        refine_x1 = min(source_width - window_width, scale_best_xy[0] + refine_radius)
        refine_y0 = max(0, scale_best_xy[1] - refine_radius)
        refine_y1 = min(source_height - window_height, scale_best_xy[1] + refine_radius)
        for y in range(refine_y0, refine_y1 + 1, refine_step):
            for x in range(refine_x0, refine_x1 + 1, refine_step):
                score = crop_similarity_at(
                    source_gray,
                    source_edges,
                    target_gray,
                    target_edges,
                    x,
                    y,
                    window_width,
                    window_height,
                )
                if score > best_score:
                    best_score = score
                    best_box = (x, y, window_width, window_height)
                    best_ratio = window_ratio

    return best_score, best_box, best_ratio


def crop_similarity_at(
    source_gray: np.ndarray,
    source_edges: np.ndarray,
    target_gray: np.ndarray,
    target_edges: np.ndarray,
    x: int,
    y: int,
    window_width: int,
    window_height: int,
) -> float:
    source_window_gray = source_gray[y : y + window_height, x : x + window_width]
    source_window_edges = source_edges[y : y + window_height, x : x + window_width]
    gray_score = normalized_correlation(source_window_gray, target_gray)
    edge_score = normalized_correlation(source_window_edges, target_edges)
    return min(1.0, max(0.0, gray_score) + 0.05 * max(0.0, edge_score))


def scale_box_to_original(
    box: tuple[int, int, int, int],
    working_shape: tuple[int, int],
    image: ImageInfo,
) -> tuple[int, int, int, int]:
    working_height, working_width = working_shape
    x, y, width, height = box
    scale_x = image.width / max(1, working_width)
    scale_y = image.height / max(1, working_height)
    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        int(round(width * scale_x)),
        int(round(height * scale_y)),
    )


def build_crop_relation(
    source: ImageInfo,
    crop: ImageInfo,
    options: DuplicateOptions,
    segment_matches: int,
    segment_coverage: float,
    note: str,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> CropRelation:
    source_gray, source_edges = crop_match_arrays(source, options, cache)
    crop_gray, crop_edges = crop_match_arrays(crop, options, cache)
    score, box, window_ratio = crop_window_score(source_gray, source_edges, crop_gray, crop_edges)
    original_box = scale_box_to_original(box, source_gray.shape, source)
    return CropRelation(
        source=source,
        crop=crop,
        score=round(score, 4),
        source_window=original_box,
        window_ratio=round(window_ratio, 4),
        segment_matches=segment_matches,
        segment_coverage=round(segment_coverage, 4),
        note=note,
    )


def find_crop_relations(
    images: list[ImageInfo],
    options: DuplicateOptions,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[CropRelation]:
    relations: list[CropRelation] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    total_pairs = len(images) * (len(images) - 1) // 2
    checked_pairs = 0
    prefiltered_pairs = 0

    for i, img_a in enumerate(images):
        for img_b in images[i + 1 :]:
            checked_pairs += 1
            should_check, segment_matches, segment_coverage, note = should_check_crop_relation(img_a, img_b, options)
            if not should_check:
                prefiltered_pairs += 1
                continue

            relation_ab = build_crop_relation(img_a, img_b, options, segment_matches, segment_coverage, note, cache)
            relation_ba = build_crop_relation(img_b, img_a, options, segment_matches, segment_coverage, note, cache)
            relation = relation_ab if relation_ab.score >= relation_ba.score else relation_ba
            if relation.score >= options.crop_score:
                relations.append(relation)
        if on_progress:
            on_progress({
                "stage": "crop_relations",
                "idx": checked_pairs,
                "total": total_pairs,
                "name": img_a.name,
                "text": f"Checked {checked_pairs}/{total_pairs} crop/scale pairs...",
            })

    relations.sort(key=lambda item: (-item.score, item.source.name.lower(), item.crop.name.lower()))
    if on_progress and prefiltered_pairs:
        on_progress({
            "stage": "crop_relations",
            "idx": total_pairs,
            "total": total_pairs,
            "text": f"Crop prefilter skipped {prefiltered_pairs}/{total_pairs} pairs.",
        })
    return relations


def _blur_candidate_to_json(candidate: BlurCandidate) -> dict[str, Any]:
    image = candidate.image
    return {
        "name": image.name,
        "width": image.width,
        "height": image.height,
        "filesize_kb": image.size // 1024,
        "blur_score": image.blur_score,
        "local_blur_ratio": image.local_blur_ratio,
        "largest_blur_region_ratio": image.largest_blur_region_ratio,
        "reason": candidate.reason,
    }


def _crop_relation_to_json(relation: CropRelation) -> dict[str, Any]:
    x, y, width, height = relation.source_window
    return {
        "source": relation.source.name,
        "crop_candidate": relation.crop.name,
        "score": relation.score,
        "source_width": relation.source.width,
        "source_height": relation.source.height,
        "crop_width": relation.crop.width,
        "crop_height": relation.crop.height,
        "source_area": relation.source_area,
        "crop_area": relation.crop_area,
        "larger_image": relation.larger_image,
        "area_ratio": relation.area_ratio,
        "relation_kind": relation.relation_kind,
        "source_window": {"x": x, "y": y, "width": width, "height": height},
        "window_ratio": relation.window_ratio,
        "segment_matches": relation.segment_matches,
        "segment_coverage": relation.segment_coverage,
        "note": relation.note,
    }
