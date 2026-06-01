"""HTTPException 包装 helper（PR-5 从 server.py 抽出）。

把 paths 模块的 ValueError / 路径不合法异常统一包成 HTTP 400 / 校验路径
+ data export 文件名解析（用于 /api/preset/export 类下载 endpoint）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from ..services.presets import io as presets_io
from ..paths import DATA_EXPORTS, safe_join, validate_path_component


def _safe_join_or_400(base: Path, *parts: str) -> Path:
    """safe_join 的 HTTPException 版本。把 ValueError 包成 400。"""
    try:
        return safe_join(base, *parts)
    except ValueError as exc:
        raise HTTPException(400, f"invalid path: {exc}") from exc


def _validate_component_or_400(name: str) -> None:
    """validate_path_component 的 HTTPException 版本（用于不需要 join 的纯名校验）。"""
    try:
        validate_path_component(name)
    except ValueError as exc:
        raise HTTPException(400, f"invalid path: {exc}") from exc


def _data_export_path(filename: str, suffixes: tuple[str, ...] = (".zip",)) -> Path:
    path = _safe_join_or_400(DATA_EXPORTS, filename)
    allowed = tuple(s.lower() for s in suffixes)
    if allowed and path.suffix.lower() not in allowed:
        label = " / ".join(allowed)
        raise HTTPException(400, f"请选择 {label} 文件")
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
    raise HTTPException(409, f"导出文件名冲突过多: {filename}")


def _export_result(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
    }


def _preset_err_code(exc: presets_io.PresetError) -> None:
    """PR-2 C4: 把 PresetError 的 message 字符串匹配映射写到 exc.http_status + exc.code，
    让 DomainError handler 翻 dual-write envelope。

    callsite 模式：
        except PresetError as exc:
            _preset_err_code(exc)  # mutate exc.http_status + exc.code
            raise  # handler 翻 envelope（自带 trace_id）

    PR-2 C5 / 0.13.x 的目标：service 内 raise PresetError(msg, http_status=N,
    code="preset.xxx") 直接带这些属性，router 不再需要这个 helper + try/except。
    """
    msg = str(exc)
    if "不存在" in msg:
        exc.http_status = 404
        exc.code = "preset.not_found"
    elif "非法预设名" in msg:
        exc.http_status = 400
        exc.code = "preset.name_invalid"
    elif "已存在" in msg:
        exc.http_status = 400
        exc.code = "preset.exists"
    else:
        exc.http_status = 422
