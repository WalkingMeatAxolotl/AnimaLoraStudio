"""Curation 操作（PP3）：download / train 双面板的后端逻辑。

- `download/` 永远是项目级全量备份，不删
- 左侧候选 = download/ 列表（**每张图通过 `preprocess_manifest.resolve()` 拿
  实际字节路径**，可能是 download/{name} 也可能是 preprocess/{name}.png——
  对前端透明，见 ADR 0004）
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
from .services import preprocess_manifest

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
    """筛选页左侧候选列表 = 预处理后可独立勾选的所有图。

    历史上这就是 `download/` 的 ls，每张原图一行；ADR 0004 之后引入了"已处理 →
    preprocess/{stem}.png"的隐式映射，但仍维持 1:1 行（前端不感知差异）。

    Multi-crop fan-out（一张原图 → N 张 `X_c0.png` / `X_c1.png` ...）打破了 1:1
    —— 用户期望在筛选里看到 N 行可单独勾选 / 单独丢弃，所以这里**展开派生**：

      - download/X.jpg 在 manifest 里有 origin=X.jpg 的 entries → 列出这些
        preprocess 派生文件名（含 _c0/_c1 等后缀），mtime 取 preprocess/ 副本
      - download/X.jpg 没 entries → 列出 X.jpg 自身（隐式 original）
      - manifest 里有 origin 但 download 原图已被删 → orphan，不列出（curation
        阶段无法重抓，UI 不该展示死链）

    返回的 name 字段对 derived 行是 preprocess 产物名，对 original 行是 download
    文件名。`copy_to_train` 同样接受两种 name；resolve 在那一侧处理。
    """
    p, pdir = _project_dir(conn, project_id)
    download_dir = pdir / "download"
    preprocess_manifest.ensure_manifest(pdir)
    processed = preprocess_manifest.all_processed(pdir)
    removed_origins = preprocess_manifest.duplicate_removed_origins(pdir)

    # origin → [preprocess names...]
    by_origin: dict[str, list[str]] = {}
    for name, entry in processed.items():
        origin = preprocess_manifest.entry_origin(entry, name)
        by_origin.setdefault(origin, []).append(name)

    entries: list[dict[str, Any]] = []
    if download_dir.exists():
        for f in sorted(download_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            derivatives = by_origin.get(f.name)
            if derivatives:
                # Expand each preprocess derivative as its own row
                for pname in sorted(derivatives):
                    pp = pdir / "preprocess" / pname
                    try:
                        mtime = pp.stat().st_mtime
                    except OSError:
                        continue
                    entries.append({"name": pname, "mtime": mtime})
            else:
                if f.name in removed_origins:
                    continue
                try:
                    mtime = f.stat().st_mtime
                except OSError:
                    continue
                entries.append({"name": f.name, "mtime": mtime})
    entries.sort(key=lambda e: e["name"])
    return entries


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

    每个文件返回 `{name, mtime}`；前端用 mtime 提供「按时间」排序。
    左侧的实际字节路径由 resolver 决定（已处理走 preprocess/ 副本，未处理走原图），
    前端通过项目缩略图端点拿，不感知差异。
    """
    left = list_download(conn, project_id)
    train = list_train(conn, project_id, version_id)
    used: set[str] = set()
    for files in train.values():
        used.update(e["name"] for e in files)
    return {
        "left": [e for e in left if e["name"] not in used],
        "right": train,
        # download_total 保留语义：左侧候选总数（与历史 API 兼容）
        "download_total": len(left),
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
    """从工作集复制选中文件到 train/{dest_folder}/，已存在跳过。

    `files` 里每个 name 可能是两种：
      1. **preprocess 派生名**（如 `X.png` 单 crop 或 `X_c0.png` multi-crop）—
         前端 `list_download` 把派生展开过的行选中后传过来。manifest 有 entry
         → 直接复制 `preprocess/{name}` 到 `train/{name}`。
      2. **download 原图名**（如 `Y.jpg`）— 未处理的图。manifest 无 entry →
         复制 `download/{name}` 到 `train/{name}`。

    metadata（`.txt` / `.json`）始终从 download 目录拿（标签不会被预处理改写）：
    多 crop 派生共享同一份原图 caption，复制到 train 下目标文件的 stem 上。
    """
    _validate_folder(dest_folder)
    p, _, train = _version_train_dir(conn, project_id, version_id)
    pdir = projects.project_dir(p["id"], p["slug"])
    download_dir = pdir / "download"
    preprocess_manifest.ensure_manifest(pdir)
    dst_dir = train / dest_folder
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    for name in files:
        _validate_filename(name)
        entry = preprocess_manifest.get_entry(pdir, name)
        if preprocess_manifest.is_duplicate_removed_entry(entry):
            skipped.append(name)
            continue
        if entry is not None:
            # preprocess 派生：bytes 在 preprocess/，metadata 在 download/{origin}.txt
            src = pdir / "preprocess" / name
            meta_stem = Path(
                preprocess_manifest.entry_origin(entry, name)
            ).stem
        else:
            # 未处理：bytes + metadata 都在 download/
            src = download_dir / name
            meta_stem = Path(name).stem

        if not src.exists():
            missing.append(name)
            continue
        dst = dst_dir / name
        if dst.exists():
            skipped.append(name)
            continue
        shutil.copy2(src, dst)
        # metadata 按 download/{meta_stem}.{ext} 找，复制到 train/{dst.stem}.{ext}
        dst_stem = Path(name).stem
        for ext in _META_EXTS:
            sm = download_dir / f"{meta_stem}{ext}"
            if sm.exists():
                try:
                    shutil.copy2(sm, dst_dir / f"{dst_stem}{ext}")
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
