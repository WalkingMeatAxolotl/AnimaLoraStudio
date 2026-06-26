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
            f'Invalid folder name: "{name}"',
            code="curation.folder_name_invalid", details={"name": name},
        )


def _validate_filename(name: str) -> None:
    if not _FILE_PATTERN.fullmatch(name) or ".." in name:
        raise CurationError(
            f'Invalid file name: "{name}"',
            code="curation.file_name_invalid", details={"name": name},
        )


def _project_dir(conn, project_id: int) -> tuple[dict[str, Any], Path]:
    p = projects.get_project(conn, project_id)
    if not p:
        raise CurationError(
            "Project not found", code="project.not_found",
            details={"id": project_id}, http_status=404,
        )
    return p, projects.project_dir(p["id"], p["slug"])


def _resolve_version_dir(conn, project_id: int, version_id: int) -> tuple[
    dict[str, Any], dict[str, Any], Path
]:
    """(project, version, version 根目录)；project/version 不存在抛 404。"""
    p = projects.get_project(conn, project_id)
    if not p:
        raise CurationError(
            "Project not found", code="project.not_found",
            details={"id": project_id}, http_status=404,
        )
    v = versions.get_version(conn, version_id)
    if not v or v["project_id"] != project_id:
        raise CurationError(
            "Version not found", code="version.not_found",
            details={"id": version_id}, http_status=404,
        )
    return p, v, versions.version_dir(p["id"], p["slug"], v["label"])


def _version_train_dir(conn, project_id: int, version_id: int) -> tuple[
    dict[str, Any], dict[str, Any], Path
]:
    p, v, vdir = _resolve_version_dir(conn, project_id, version_id)
    return p, v, vdir / "train"


def _version_validation_dir(conn, project_id: int, version_id: int) -> tuple[
    dict[str, Any], dict[str, Any], Path
]:
    p, v, vdir = _resolve_version_dir(conn, project_id, version_id)
    return p, v, vdir / "validation"


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

    ADR 0010 fixup：Curation 右侧 train 区显示"用户筛选时选了哪些 download 原图"，
    跟预处理后状态解耦：

    - 按 manifest entry.origin **去重**：multi-crop fan-out 派生（X_c0.png +
      X_c1.png 同 origin=X.jpg）只显示一条
    - 物理 iterdir 决定显示集合：duplicate_removed 物理已删 → 自然不出现
      在 Curation；要查看 / 恢复走总览页"已删除"tab
    - 返回 `name` 用 **origin**（download 文件名），跟 `copy_download_to_train` /
      `remove_from_train` 的 name 语义对齐到 download scope

    `mtime` 用物理文件 mtime；前端按时间排序仍稳定。老项目 fallback：
    ensure_train_manifest 重建后走同一路径。
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
        # 物理目录决定显示集合（duplicate_removed 物理已删→不出现）+
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


def list_validation(
    conn, project_id: int, version_id: int
) -> list[dict[str, Any]]:
    """validation/ 下所有子文件夹的图拍平成一条 flat list（手动加的 +
    auto-split 移进来的都列），每条 `{name, mtime, folder}`。

    validation 无 manifest（held-out 集只读，靠目录位置区分身份，见
    eval_validation 模块），所以这里不查 origin / 不去重，`name` 就是物理文件名。
    `folder` 给前端用来定位缩略图（version thumb 的 validation bucket 需要它）
    与精确删除（多选可能跨 auto-split 的不同 repeat 文件夹）。
    """
    _, _, val = _version_validation_dir(conn, project_id, version_id)
    if not val.exists():
        return []
    out: list[dict[str, Any]] = []
    for sub in sorted(p for p in val.iterdir() if p.is_dir()):
        for e in _list_image_entries(sub):
            out.append({"name": e["name"], "mtime": e["mtime"], "folder": sub.name})
    return out


def _used_names(train: dict[str, list[dict[str, Any]]],
                val: list[dict[str, Any]]) -> set[str]:
    """已分配到 train（按 origin）或 validation 的 download 名集合。

    held-out 要求一张图不能同时在 train 和 validation，否则 eval 测的是记忆
    不是泛化。左栏候选从 download 里减掉这个集合，两个 bucket 共用同一池。
    """
    used = {e["name"] for files in train.values() for e in files}
    used |= {e["name"] for e in val}
    return used


def curation_view(conn, project_id: int, version_id: int) -> dict[str, Any]:
    """前端用：left = download − train − validation，right = train 按 folder 分组。

    每个文件返回 `{name, mtime}`；前端用 mtime 提供「按时间」排序。
    左侧的实际字节路径由 resolver 决定（已处理走 preprocess/ 副本，未处理走原图），
    前端通过项目缩略图端点拿，不感知差异。

    left 同时减掉 validation：训练后 auto-split 把图移进 validation/ 后，这些图
    在 download/ 里仍在，旧逻辑会让它们重新冒回 train 左栏候选 → 可被重新加进
    train → 与 validation 重叠泄漏。减掉 validation 既修这个，也让 train /
    validation 两个 curation 视图共用同一候选池。
    """
    left = list_download(conn, project_id)
    train = list_train(conn, project_id, version_id)
    val = list_validation(conn, project_id, version_id)
    used = _used_names(train, val)
    return {
        "left": [e for e in left if e["name"] not in used],
        "right": train,
        # download_total 保留语义：左侧候选总数（与历史 API 兼容）
        "download_total": len(left),
        "train_total": sum(len(v) for v in train.values()),
        "folders": list(train.keys()),
    }


def curation_validation_view(
    conn, project_id: int, version_id: int
) -> dict[str, Any]:
    """验证集筛选视图：left = download − train − validation（与训练集同池），
    right = validation 扁平列表。

    与 `curation_view` 对称，区别只在 right 是 flat（无文件夹概念，见
    `list_validation`）。前端验证集模式渲染右栏用它。
    """
    left = list_download(conn, project_id)
    train = list_train(conn, project_id, version_id)
    val = list_validation(conn, project_id, version_id)
    used = _used_names(train, val)
    return {
        "left": [e for e in left if e["name"] not in used],
        "right": val,
        "download_total": len(left),
        "val_total": len(val),
    }


# ---------------------------------------------------------------------------
# copy / remove
# ---------------------------------------------------------------------------


_META_EXTS = (".txt", ".json")


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
# validation copy / remove（held-out 验证集手动维护）
#
# 与 train 的 copy/remove 对称，但 validation 无 manifest（held-out 集靠目录位置
# 区分身份），所以更简单：纯物理复制 / 删除，不写 manifest、不做 origin 去重。
# 手动加入的图统一落到固定 `validation/1_data/`（复用 DEFAULT_TRAIN_FOLDER，
# 与常见 auto-split 落点一致）；UI 不暴露文件夹概念。
# ---------------------------------------------------------------------------


def copy_download_to_validation(
    conn,
    project_id: int,
    version_id: int,
    files: list[str],
) -> dict[str, list[str]]:
    """download → `validation/1_data/` 复制（连同 caption .txt/.json）。

    `files` 是 download 池里的图名（平铺，不带 folder 前缀）。已在 train（按
    origin）或 validation 里的名字一律 skip —— 防 held-out 泄漏，与
    `curation_view` 左栏候选的排除集一致。验证集的 caption 会被 eval 当生成
    prompt 用（见 eval_samples），所以 sidecar 必须跟着复制。
    """
    p, v, val = _version_validation_dir(conn, project_id, version_id)
    pdir = projects.project_dir(p["id"], p["slug"])
    download_dir = pdir / "download"
    dst_dir = val / versions.DEFAULT_TRAIN_FOLDER
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 已分配集合（train origins ∪ validation 名）——同 curation_view 的排除口径。
    train = list_train(conn, project_id, version_id)
    existing_val = list_validation(conn, project_id, version_id)
    used = _used_names(train, existing_val)

    copied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    for name in files:
        _validate_filename(name)
        if name in used:
            skipped.append(name)
            continue
        src = download_dir / name
        if not src.exists():
            missing.append(name)
            continue
        dst = dst_dir / name
        if dst.exists():
            skipped.append(name)
            continue
        shutil.copy2(src, dst)
        stem = Path(name).stem
        for ext in _META_EXTS:
            sm = download_dir / f"{stem}{ext}"
            if sm.exists():
                try:
                    shutil.copy2(sm, dst_dir / f"{stem}{ext}")
                except OSError:
                    pass
        used.add(name)
        copied.append(name)
    return {"copied": copied, "skipped": skipped, "missing": missing}


def remove_from_validation(
    conn,
    project_id: int,
    version_id: int,
    items: list[dict[str, str]],
) -> dict[str, list[str]]:
    """从 validation/ 删除指定图（连同同 stem caption）；download 不动。

    `items` 是 `[{"folder": ..., "name": ...}]`：多选可能跨 auto-split 的不同
    repeat 文件夹，所以按 (folder, name) 精确定位，而不是按名跨文件夹删。
    """
    _, _, val = _version_validation_dir(conn, project_id, version_id)
    removed: list[str] = []
    missing: list[str] = []
    for item in items:
        folder = item.get("folder", "")
        name = item.get("name", "")
        _validate_folder(folder)
        _validate_filename(name)
        pp = val / folder / name
        if not pp.exists():
            missing.append(name)
            continue
        try:
            pp.unlink()
        except OSError:
            missing.append(name)
            continue
        for ext in _META_EXTS:
            mp = pp.with_suffix(ext)
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
        raise CurationError(
            f'Folder "{name}" already exists',
            code="curation.folder_exists", details={"name": name},
        )
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
        raise CurationError(
            f'Folder "{name}" not found',
            code="curation.folder_not_found", details={"name": name},
            http_status=404,
        )
    if dst.exists():
        raise CurationError(
            f'Folder "{new_name}" already exists',
            code="curation.folder_exists", details={"name": new_name},
        )
    src.rename(dst)
    return dst


def delete_folder(conn, project_id: int, version_id: int, name: str) -> None:
    """整个子文件夹连同里面的 train 副本一起删；download 不动。"""
    _validate_folder(name)
    _, _, train = _version_train_dir(conn, project_id, version_id)
    target = train / name
    if not target.exists():
        raise CurationError(
            f'Folder "{name}" not found',
            code="curation.folder_not_found", details={"name": name},
            http_status=404,
        )
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
