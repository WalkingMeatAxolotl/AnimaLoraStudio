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


class CopyValidationRequest(BaseModel):
    """download → validation 复制（落固定 validation/1_data/，无 dest_folder）。"""
    files: list[str]


class ValidationItem(BaseModel):
    folder: str
    name: str


class RemoveValidationRequest(BaseModel):
    """按 (folder, name) 精确删 —— 多选可能跨 auto-split 的不同 repeat 文件夹。"""
    items: list[ValidationItem]


class FolderOp(BaseModel):
    op: str  # "create" | "rename" | "delete"
    name: str
    new_name: Optional[str] = None


class DuplicateScanRequest(BaseModel):
    # UI 只暴露匹配范围 + 灵敏度；其余阈值/性能参数已固化为 duplicates.DEFAULT_*。
    match_scope: str = "both"
    sensitivity: str = duplicate_finder.DEFAULT_SENSITIVITY


class DuplicateApplyRequest(BaseModel):
    names: list[str]
