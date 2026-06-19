"""HTTPException 包装 helper（PR-5 从 server.py 抽出）。

把 paths 模块的 ValueError / 路径不合法异常统一包成 HTTP 400 / 校验路径
+ data export 文件名解析（用于 /api/preset/export 类下载 endpoint）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..domain.errors import ConflictError, InvalidPathError, ValidationError
from ..paths import DATA_EXPORTS, safe_join, validate_path_component


def _safe_join_or_400(base: Path, *parts: str) -> Path:
    """safe_join 的 HTTPException 版本。把 ValueError 包成 400。"""
    try:
        return safe_join(base, *parts)
    except ValueError as exc:
        raise InvalidPathError("Invalid path", details={"reason": str(exc)}) from exc


def _validate_component_or_400(name: str) -> None:
    """validate_path_component 的 HTTPException 版本（用于不需要 join 的纯名校验）。"""
    try:
        validate_path_component(name)
    except ValueError as exc:
        raise InvalidPathError("Invalid path", details={"reason": str(exc)}) from exc


def _data_export_path(filename: str, suffixes: tuple[str, ...] = (".zip",)) -> Path:
    path = _safe_join_or_400(DATA_EXPORTS, filename)
    allowed = tuple(s.lower() for s in suffixes)
    if allowed and path.suffix.lower() not in allowed:
        label = " / ".join(allowed)
        raise ValidationError(
            f"Select a {label} file",
            code="file.ext_invalid", details={"types": label}, http_status=400,
        )
    return path


def _unique_data_export_path(
    filename: str, suffixes: tuple[str, ...] = (".zip",)
) -> Path:
    base = _data_export_path(filename, suffixes)
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    for i in range(2, 1000):
        candidate = _data_export_path(f"{stem}-{i}{suffix}", suffixes)
        if not candidate.exists():
            return candidate
    raise ConflictError(
        f'Too many files named "{filename}"; rename and try again',
        code="export.name_conflict", details={"name": filename},
    )


def _export_result(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
    }
