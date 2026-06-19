"""文件浏览 / 数据集扫描 / 缩略图（PR-5 从 server.py 抽出）。

3 routes：
    GET /api/datasets             扫描数据集目录（缺省 = repo_root/dataset）
    GET /api/browse               目录浏览（PathPicker 用，allow_outside_repo=True）
    GET /api/datasets/thumbnail   单图缩略图（仅 REPO_ROOT 内）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse

from .. import errors as _errors
from ...domain.errors import InvalidPathError, NotFoundError
from ...services.dataset import browse, scan as datasets
from ...infrastructure.paths import REPO_ROOT

router = APIRouter()


@router.get("/api/datasets")
def get_datasets(path: str = "") -> dict[str, Any]:
    """扫描数据集目录。`?path=` 指定根目录；缺省 = repo_root/dataset。"""
    root = Path(path) if path else REPO_ROOT / "dataset"
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    return datasets.scan_dataset_root(root)


@router.get("/api/browse")
def browse_dir(path: str = "") -> dict[str, Any]:
    """目录浏览（给前端 path picker 用）。缺省 = REPO_ROOT。

    PathPicker 设计本就是给用户选外部模型路径用的（云端机器把模型放数据盘），
    所以这里 allow_outside_repo=True；安全边界在 list_dir 本身（只读 entries
    名字+类型，不返回内容）。
    """
    target = Path(path) if path else REPO_ROOT
    if not target.is_absolute():
        target = (REPO_ROOT / target).resolve()
    return browse.list_dir(target, allow_outside_repo=True)


@router.get("/api/datasets/thumbnail")
def get_dataset_thumbnail(folder: str, name: str) -> FileResponse:
    """返回 dataset 缩略图（实际是原图，前端用 CSS 缩放）。

    `folder` 可以是绝对路径或相对路径（用户在 dataset 浏览器里点出来的），
    `name` 必须是单一文件名（不含分隔符）。最终路径必须在 REPO_ROOT 内。
    """
    _errors._validate_component_or_400(name)
    p = (Path(folder) / name).resolve()
    try:
        p.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise InvalidPathError("Invalid path", http_status=403) from None
    if not p.exists() or p.suffix.lower() not in datasets.IMAGE_EXTS:
        raise NotFoundError(
            "Thumbnail not found", code="dataset.thumbnail_not_found",
        )
    return FileResponse(p)
