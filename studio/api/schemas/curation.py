"""files/curation/duplicates BaseModel（PR-6.5 commit 4 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ...services import duplicate_finder


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


class DuplicateApplyRequest(BaseModel):
    names: list[str]
