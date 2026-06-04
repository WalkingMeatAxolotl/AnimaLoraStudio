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

from ..projects import projects, versions
from .scan import IMAGE_EXTS
from ..preprocess import manifest as preprocess_manifest

# Kohya: 可选 `N_` 前缀 + 字母（不允许纯数字 / `5_` 这种空 label）
_FOLDER_PATTERN = re.compile(r"^([0-9]+_)?[A-Za-z][A-Za-z0-9_-]*$")
# 文件名安全：仅允许文件名（含扩展名），不允许任何路径分隔
_FILE_PATTERN = re.compile(r"^[^\\/]+$")


from studio.domain.errors import DomainError


class CurationError(DomainError):
    """Curation 业务错误（路径非法 / 不存在 / 冲突）。

    PR-2 C3 加 DomainError base — handler 自动翻 dual-write envelope。
    """
    default_code = "curation.error"


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
    """筛选页左侧候选列表 = `download/` 物理图，每张一行。

    ADR 0010 fixup（2026-06-04）：Curation 跟预处理派生解耦。原 ADR 0004 设计
    会按 manifest 展开 multi-crop 派生（X.jpg → 显示 X_c0.png + X_c1.png），但
    新模型下 list_train 按 origin 去重（fan-out 折叠成一行 X.jpg），left/right
    名字空间不一致会让 `used` 排除失败 → 已加入 train 的图重新出现在 left →
    用户重选 → `copy_to_train` 看到 dst 物理已存在 → skip 报错。

    新行为：list_download 只列 download/ 物理图（不感知 manifest 派生）；
    name 跟 list_train 返回的 origin 在同一命名空间（download 文件名），
    used 排除走得通。预处理派生只在 Preprocess Overview 暴露给用户。

    `duplicate_removed` 也不过滤（PR-4 上一 fixup 决议，去重已下沉 train scope）。
    """
    _, pdir = _project_dir(conn, project_id)
    download_dir = pdir / "download"

    entries: list[dict[str, Any]] = []
    if download_dir.exists():
        for f in sorted(download_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            entries.append({"name": f.name, "mtime": mtime})
    return entries


def list_train(
    conn, project_id: int, version_id: int
) -> dict[str, list[dict[str, Any]]]:
    """train 子文件夹 → `[{name, mtime, origin}, ...]`（按 origin 去重）。

    ADR 0010 fixup（2026-06-04）：Curation 右侧 train 区显示"用户筛选时
    选了哪些 download 原图"，跟预处理后状态解耦：

    - 按 manifest entry.origin **去重**：multi-crop fan-out 派生（X_c0.png +
      X_c1.png 同 origin=X.jpg）只显示一条
    - 不过滤 `duplicate_removed` entry：用户当时筛选选了它，时间上在预处理
      之前，UI 该显示
    - 扫 train manifest 而不是物理文件（fan-out 派生数量跟筛选语义无关）
    - 返回 `name` 用 **origin**（download 文件名），跟 `copy_to_train` /
      `remove_from_train` 的 name 语义对齐到 download scope

    `mtime` 用 manifest entry 的 mtime（物理产物 mtime；多派生取第一条）；
    前端按时间排序仍稳定。老项目 fallback：ensure_train_manifest 重建后
    走同一路径。
    """
    p, v, train = _version_train_dir(conn, project_id, version_id)
    if not train.exists():
        return {}
    pdir = projects.project_dir(p["id"], p["slug"])
    preprocess_manifest.ensure_train_manifest(pdir, v["label"])
    tm = preprocess_manifest.train_load(pdir, v["label"])
    entries = tm.get("images", {})

    out: dict[str, list[dict[str, Any]]] = {}
    for sub in sorted(train.iterdir()):
        if not sub.is_dir():
            continue
        # 物理目录决定显示集合（含 duplicate_removed 软删除的图：物理还在）+
        # 兼容老路径（copy_to_train 不写 manifest，但物理图能扫到）。manifest
        # 仅用于反查 origin → 按 origin 去重（multi-crop fan-out 折叠成一行）。
        items_by_origin: dict[str, dict[str, Any]] = {}
        for raw in _list_image_entries(sub):
            rel = f"{sub.name}/{raw['name']}"
            entry = entries.get(rel, {})
            origin = preprocess_manifest.entry_origin(entry, raw["name"])
            if origin in items_by_origin:
                continue
            items_by_origin[origin] = {
                "name": origin,
                "origin": origin,
                "mtime": raw["mtime"],
            }
        out[sub.name] = sorted(items_by_origin.values(), key=lambda e: e["name"])
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


def copy_download_to_train(
    conn,
    project_id: int,
    version_id: int,
    files: list[str],
    dest_folder: str,
) -> dict[str, list[str]]:
    """ADR 0010 train scope（PR-2 step C）：纯 download → train 复制 + 写
    train manifest entry。简化版替代 `copy_to_train`，PR-3 删老的。

    跟老 `copy_to_train` 的差异：

    - **取消 preprocess 派生分支** — bytes 始终从 `download/{name}` 拿
    - 写 train manifest entry，key = `f"{dest_folder}/{name}"`，
      origin = name（curate 阶段图是原图未处理；后续 preprocess 在 train/
      原地处理时再 update entry）
    - caption (.txt/.json) 仍从 `download/{stem}.{ext}` 复制到
      `train/{dest_folder}/{stem}.{ext}`
    - 不消费 / 不感知 preprocess 派生（multi-crop fan-out 在新模型下发生在
      curate 之后的 preprocess phase）

    `files` 是 download 池里的图名（平铺），不带 folder 前缀。
    """
    _validate_folder(dest_folder)
    p, v, train = _version_train_dir(conn, project_id, version_id)
    pdir = projects.project_dir(p["id"], p["slug"])
    download_dir = pdir / "download"
    dst_dir = train / dest_folder
    dst_dir.mkdir(parents=True, exist_ok=True)
    preprocess_manifest.ensure_train_manifest(pdir, v["label"])

    copied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    for name in files:
        _validate_filename(name)
        src = download_dir / name
        if not src.exists():
            missing.append(name)
            continue
        dst = dst_dir / name
        if dst.exists():
            skipped.append(name)
            continue
        shutil.copy2(src, dst)
        # caption metadata 跟随
        stem = Path(name).stem
        for ext in _META_EXTS:
            sm = download_dir / f"{stem}{ext}"
            if sm.exists():
                try:
                    shutil.copy2(sm, dst_dir / f"{stem}{ext}")
                except OSError:
                    pass
        # 写 train manifest entry，key = "{folder}/{name}"
        rel = f"{dest_folder}/{name}"
        meta: dict[str, Any] = {"origin": name}
        try:
            st = dst.stat()
            meta["mtime"] = st.st_mtime
            meta["size"] = st.st_size
        except OSError:
            pass
        preprocess_manifest.train_add_processed(pdir, v["label"], rel, meta)
        copied.append(name)
    return {"copied": copied, "skipped": skipped, "missing": missing}


def remove_from_train(
    conn,
    project_id: int,
    version_id: int,
    folder: str,
    files: list[str],
) -> dict[str, list[str]]:
    """从 train/{folder}/ 删除 download 原图的所有 train 派生 + 同 stem
    metadata；download 不动。

    ADR 0010 fixup（2026-06-04）：`files` 是 **origin 名**（download 文件
    名），跟 list_train 返回的 `name` 字段一致。本函数查 train manifest
    找所有 origin 匹配的 entry，删它们的 train 物理文件 + manifest entry
    + 同 stem caption (.txt/.json)。这样删一行 = 删该原图在 train 里的
    全部派生（multi-crop fan-out 一并清掉）。
    """
    _validate_folder(folder)
    p, v, train = _version_train_dir(conn, project_id, version_id)
    pdir = projects.project_dir(p["id"], p["slug"])
    fdir = train / folder
    preprocess_manifest.ensure_train_manifest(pdir, v["label"])
    tm = preprocess_manifest.train_load(pdir, v["label"])
    entries = tm.get("images", {})

    # origin → [rel paths in this folder]
    by_origin: dict[str, list[str]] = {}
    for rel, entry in entries.items():
        if "/" not in rel:
            continue
        f, filename = rel.split("/", 1)
        if f != folder:
            continue
        origin = preprocess_manifest.entry_origin(entry, filename)
        by_origin.setdefault(origin, []).append(rel)

    removed: list[str] = []
    missing: list[str] = []
    rels_to_pop: list[str] = []
    for origin_name in files:
        _validate_filename(origin_name)
        rels = by_origin.get(origin_name, [])
        if not rels:
            # manifest 没记 → 兜底直接删 fdir / origin_name（老项目同名场景）
            pp = fdir / origin_name
            if pp.exists():
                pp.unlink()
                for ext in _META_EXTS:
                    mp = pp.with_suffix(ext)
                    if mp.exists():
                        try:
                            mp.unlink()
                        except OSError:
                            pass
                removed.append(origin_name)
            else:
                missing.append(origin_name)
            continue
        # 删所有派生物理文件 + 各派生 stem 的 metadata
        for rel in rels:
            _, filename = rel.split("/", 1)
            pp = fdir / filename
            if pp.exists():
                try:
                    pp.unlink()
                except OSError:
                    pass
            for ext in _META_EXTS:
                mp = pp.with_suffix(ext)
                if mp.exists():
                    try:
                        mp.unlink()
                    except OSError:
                        pass
        rels_to_pop.extend(rels)
        removed.append(origin_name)

    if rels_to_pop:
        preprocess_manifest.train_remove_entries(
            pdir, v["label"], rels_to_pop,
        )
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
