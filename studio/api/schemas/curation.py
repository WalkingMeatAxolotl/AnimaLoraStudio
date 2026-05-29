"""files/curation/duplicates BaseModel（PR-6.5 commit 4 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ...services.preprocess import duplicates as duplicate_finder


class DeleteFilesRequest(BaseModel):
    names: list[str]


class CopyRequest(BaseModel):
    files: list[str]
    dest_folder: str


class RemoveRequest(BaseModel):
    folder: str
    files: list[str]


class FolderOp(BaseModel):
    op: str  # "create" | "rename" | "delete"
    name: str
    new_name: Optional[str] = None


class DuplicateScanRequest(BaseModel):
    match_scope: str = "both"
    hash_size: int = duplicate_finder.DEFAULT_HASH_SIZE
    hash_workers: int = duplicate_finder.DEFAULT_HASH_WORKERS
    tile_grids: list[int] = list(duplicate_finder.DEFAULT_TILE_GRIDS)
    structure_threshold: int = duplicate_finder.DEFAULT_STRUCTURE_THRESHOLD
    variant_score: float = duplicate_finder.DEFAULT_VARIANT_SCORE
    aspect_tolerance: float = duplicate_finder.DEFAULT_ASPECT_TOLERANCE
    min_close_tiles: float = duplicate_finder.DEFAULT_MIN_CLOSE_TILES
    tile_median: float = duplicate_finder.DEFAULT_TILE_MEDIAN
    min_gray_close: float = duplicate_finder.DEFAULT_MIN_GRAY_CLOSE
    detect_blur: bool = duplicate_finder.DEFAULT_DETECT_BLUR
    blur_score_threshold: float = duplicate_finder.DEFAULT_BLUR_SCORE_THRESHOLD
    blur_local_ratio: float = duplicate_finder.DEFAULT_BLUR_LOCAL_RATIO
    detect_crops: bool = duplicate_finder.DEFAULT_DETECT_CROPS
    crop_score: float = duplicate_finder.DEFAULT_CROP_SCORE
    crop_hash_threshold: int = duplicate_finder.DEFAULT_CROP_HASH_THRESHOLD
    crop_max_side: int = duplicate_finder.DEFAULT_CROP_MAX_SIDE


class DuplicateApplyRequest(BaseModel):
    names: list[str]
