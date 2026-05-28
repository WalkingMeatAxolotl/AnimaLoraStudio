"""文件系统浏览：给前端的「选路径」控件提供数据。

仅返回目录信息（不读文件内容）。允许的根白名单可由调用方限定，默认仅
`REPO_ROOT` 下；显式传入 path 时校验其在白名单内或为绝对路径下的合法位置。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...paths import REPO_ROOT


class BrowseError(Exception):
    pass


def list_dir(target: Path, *, allow_outside_repo: bool = False) -> dict[str, Any]:
    """列出 target 目录下的子目录和文件名。

    target 若指向一个已存在的文件，回退到其父目录并在返回里通过 `selected`
    字段告诉前端该高亮哪一项（picker 打开到"文件所在目录"是常见用法）。

    所有路径用 POSIX 形式（`/` 分隔符）返回，避免 Windows 反斜杠在前端拼接
    时与 yaml 里存的 forward-slash 风格混用。
    """
    target = target.resolve()
    if not allow_outside_repo:
        try:
            target.relative_to(REPO_ROOT.resolve())
        except ValueError:
            raise BrowseError(f"path outside repo: {target}")

    selected: str | None = None
    if target.exists() and not target.is_dir():
        selected = target.name
        target = target.parent

    if not target.exists():
        raise BrowseError(f"path does not exist: {target}")
    if not target.is_dir():
        raise BrowseError(f"not a directory: {target}")

    entries: list[dict[str, Any]] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            entries.append({
                "name": child.name,
                "type": "dir" if is_dir else "file",
            })
    except PermissionError:
        raise BrowseError(f"permission denied: {target}")

    parent = target.parent.as_posix() if target.parent != target else None
    return {
        "path": target.as_posix(),
        "parent": parent,
        "entries": entries,
        "selected": selected,
    }
