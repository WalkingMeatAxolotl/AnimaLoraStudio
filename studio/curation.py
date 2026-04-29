"""Curation 操作（PP3）：download / train 双面板的后端逻辑。

- `download/` 永远是项目级全量备份，不删
- 复制 / 移除只动 `versions/{label}/train/{folder}/` 的副本
- 文件名做差集：left = download − all-train，right = train 按 folder 分组
- 子文件夹遵 Kohya 风格 N_xxx（PP4 / PP6 训练时仍按 dataset.parse_repeat 解析）
- 每张图返回 `{name, mtime}`（mtime 为 unix 秒），排序由前端按用户偏好决定；
  后端只保证按 name 字典序的稳定输出

约束：
- 不允许 path traversal（folder / 文件名都校验）
- 复制时把同名 .txt / .json metadata 一起带走（如果存在）
- 移除时连带清掉同名 metadata；download/ 一律不动
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from . import projects, versions
from .datasets import IMAGE_EXTS

# Kohya: 可选 `N_` 前缀 + 字母（不允许纯数字 / `5_` 这种空 label）
_FOLDER_PATTERN = re.compile(r"^([0-9]+_)?[A-Za-z][A-Za-z0-9_-]*$")
# 文件名安全：仅允许文件名（含扩展名），不允许任何路径分隔
_FILE_PATTERN = re.compile(r"^[^\\/]+$")


class CurationError(Exception):
    """Curation 业务错误（路径非法 / 不存在 / 冲突）。"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validate_folder(name: str) -> None:
    if not _FOLDER_PATTERN.fullmatch(name):
        raise CurationError(
            f"非法文件夹名: {name!r}（Kohya 风格 N_xxx 或纯字母数字）"
        )


def _validate_filename(name: str) -> None:
    if not _FILE_PATTERN.fullmatch(name) or ".." in name:
        raise CurationError(f"非法文件名: {name!r}")


def _project_dir(conn, project_id: int) -> tuple[dict[str, Any], Path]:
    p = projects.get_project(conn, project_id)
    if not p:
        raise CurationError(f"项目不存在: id={project_id}")
    return p, projects.project_dir(p["id"], p["slug"])


def _version_train_dir(conn, project_id: int, version_id: int) -> tuple[
    dict[str, Any], dict[str, Any], Path
]:
    p = projects.get_project(conn, project_id)
    if not p:
        raise CurationError(f"项目不存在: id={project_id}")
    v = versions.get_version(conn, version_id)
    if not v or v["project_id"] != project_id:
        raise CurationError(f"版本不存在: id={version_id}")
    train_dir = versions.version_dir(p["id"], p["slug"], v["label"]) / "train"
    return p, v, train_dir


def _list_image_entries(d: Path) -> list[dict[str, Any]]:
    """目录下的图像列表 → `[{name, mtime}, ...]`，按 name 字典序稳定输出。

    mtime 取自磁盘 stat，单位为 unix 秒（float）；前端拿到后可按 id / name /
    mtime 自由重排。
    """
    if not d.exists():
        return []
    entries: list[dict[str, Any]] = []
    for f in d.iterdir():
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            mtime = f.stat().st_mtime
        except OSError:
            mtime = 0.0
        entries.append({"name": f.name, "mtime": mtime})
    entries.sort(key=lambda e: e["name"])
    return entries


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


def list_download(conn, project_id: int) -> list[dict[str, Any]]:
    p, pdir = _project_dir(conn, project_id)
    return _list_image_entries(pdir / "download")


def list_train(
    conn, project_id: int, version_id: int
) -> dict[str, list[dict[str, Any]]]:
    """train 子文件夹 → [{name, mtime}, ...]。"""
    _, _, train = _version_train_dir(conn, project_id, version_id)
    if not train.exists():
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for sub in sorted(train.iterdir()):
        if sub.is_dir():
            out[sub.name] = _list_image_entries(sub)
    return out


def curation_view(conn, project_id: int, version_id: int) -> dict[str, Any]:
    """前端用：left = download − train，right = train 按 folder 分组。

    每个文件返回 `{name, mtime}`；前端用 mtime 提供「按下载时间」排序。
    """
    download = list_download(conn, project_id)
    train = list_train(conn, project_id, version_id)
    used: set[str] = set()
    for files in train.values():
        used.update(e["name"] for e in files)
    return {
        "left": [e for e in download if e["name"] not in used],
        "right": train,
        "download_total": len(download),
        "train_total": sum(len(v) for v in train.values()),
        "folders": list(train.keys()),
    }


# ---------------------------------------------------------------------------
# copy / remove
# ---------------------------------------------------------------------------


_META_EXTS = (".txt", ".json")


def copy_to_train(
    conn,
    project_id: int,
    version_id: int,
    files: list[str],
    dest_folder: str,
) -> dict[str, list[str]]:
    """从 download/ 复制选中文件到 train/{dest_folder}/，已存在跳过。

    若目标文件夹不存在自动创建；同名 .txt / .json 一并复制（best-effort，
    metadata 复制失败仅日志，不报错）。
    """
    _validate_folder(dest_folder)
    p, _, train = _version_train_dir(conn, project_id, version_id)
    src_dir = projects.project_dir(p["id"], p["slug"]) / "download"
    dst_dir = train / dest_folder
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    for name in files:
        _validate_filename(name)
        src = src_dir / name
        if not src.exists():
            missing.append(name)
            continue
        dst = dst_dir / name
        if dst.exists():
            skipped.append(name)
            continue
        shutil.copy2(src, dst)
        for ext in _META_EXTS:
            sm = src.with_suffix(ext)
            if sm.exists():
                try:
                    shutil.copy2(sm, dst_dir / sm.name)
                except OSError:
                    pass
        copied.append(name)
    return {"copied": copied, "skipped": skipped, "missing": missing}


def remove_from_train(
    conn,
    project_id: int,
    version_id: int,
    folder: str,
    files: list[str],
) -> dict[str, list[str]]:
    """从 train/{folder}/ 删除文件（含同名 metadata）；download 不动。"""
    _validate_folder(folder)
    _, _, train = _version_train_dir(conn, project_id, version_id)
    fdir = train / folder
    removed: list[str] = []
    missing: list[str] = []
    for name in files:
        _validate_filename(name)
        p = fdir / name
        if not p.exists():
            missing.append(name)
            continue
        p.unlink()
        for ext in _META_EXTS:
            mp = p.with_suffix(ext)
            if mp.exists():
                try:
                    mp.unlink()
                except OSError:
                    pass
        removed.append(name)
    return {"removed": removed, "missing": missing}


# ---------------------------------------------------------------------------
# folder ops
# ---------------------------------------------------------------------------


def create_folder(conn, project_id: int, version_id: int, name: str) -> Path:
    _validate_folder(name)
    _, _, train = _version_train_dir(conn, project_id, version_id)
    target = train / name
    if target.exists():
        raise CurationError(f"文件夹已存在: {name}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def rename_folder(
    conn, project_id: int, version_id: int, name: str, new_name: str
) -> Path:
    _validate_folder(name)
    _validate_folder(new_name)
    if name == new_name:
        return _version_train_dir(conn, project_id, version_id)[2] / name
    _, _, train = _version_train_dir(conn, project_id, version_id)
    src = train / name
    dst = train / new_name
    if not src.exists():
        raise CurationError(f"文件夹不存在: {name}")
    if dst.exists():
        raise CurationError(f"目标已存在: {new_name}")
    src.rename(dst)
    return dst


def delete_folder(conn, project_id: int, version_id: int, name: str) -> None:
    """整个子文件夹连同里面的 train 副本一起删；download 不动。"""
    _validate_folder(name)
    _, _, train = _version_train_dir(conn, project_id, version_id)
    target = train / name
    if not target.exists():
        raise CurationError(f"文件夹不存在: {name}")
    shutil.rmtree(target)


# ---------------------------------------------------------------------------
# stage hint
# ---------------------------------------------------------------------------


def has_train_images(
    conn, project_id: int, version_id: int
) -> bool:
    """该 version 的 train/ 下是否已经有图片（任意子文件夹）。"""
    _, _, train = _version_train_dir(conn, project_id, version_id)
    if not train.exists():
        return False
    for sub in train.iterdir():
        if sub.is_dir() and any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTS
            for f in sub.iterdir()
        ):
            return True
    return False
