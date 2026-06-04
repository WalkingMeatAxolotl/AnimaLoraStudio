"""预处理状态 manifest（单 JSON 文件，version 级）。

设计见 [ADR 0010](../../docs/adr/0010-preprocess-train-scope.md)
（supersedes ADR 0004 — 老的项目级 preprocess/manifest.json 已只剩只读
fallback 给 ensure_train_manifest 老项目迁移用，不再 mutation）。

简而言之
--------
`projects/{id}-{slug}/versions/{label}/train/manifest.json` 记录该 version
train/ 下每张图的 origin + 状态。

schema（写入用）— 极简：

    {
      "images": {
        "1_data/X.png":    {"origin": "X.png",  "mtime": ..., "size": ..., "processed": true},
        "1_data/Y_c0.png": {"origin": "Y.png",  "mtime": ..., "size": ...},
        "1_data/Y_c1.png": {"origin": "Y.png",  "mtime": ..., "size": ...}
      }
    }

字段：
- entry key = train/ 下的 POSIX 相对路径 `"{folder}/{filename}"`
- `origin` = 该图回溯到 `download/` 里的源文件名（multi-crop 派生共享 origin）
- `processed` = 是否经过 upscale / crop（worker 写 True；curate 复制不写）
- `kind: "duplicate_removed"` 标记人工审核确认跳过；不删 train/ 物理文件

「manifest 没记的图」= 隐式 original（train/ 文件由 curate 阶段刚复制进来）。

并发写
------
服务端单进程，没跨进程写者：`threading.Lock` 串行化进程内所有 mutation。
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

MANIFEST_NAME = "manifest.json"
DUPLICATE_REMOVED_KIND = "duplicate_removed"

# 进程内串行锁。所有 mutation 必须 `with _LOCK:`；read 不需要（json.load 原子）。
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------


def manifest_path(project_dir: Path) -> Path:
    return project_dir / "preprocess" / MANIFEST_NAME


# ---------------------------------------------------------------------------
# 读 / 写
# ---------------------------------------------------------------------------


def _empty_manifest() -> dict[str, Any]:
    return {"images": {}}


def load(project_dir: Path) -> dict[str, Any]:
    """读 manifest；不存在或损坏 → 空 manifest（不抛）。

    单次 read 不上锁——`json.load` 原子，最坏情况是读到旧版本，不会读到半写入。
    `_atomic_write` 用 tmp+rename 保证 rename 是原子的。
    """
    path = manifest_path(project_dir)
    if not path.exists():
        return _empty_manifest()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("images"), dict):
            return _empty_manifest()
        return raw
    except (OSError, json.JSONDecodeError):
        # 损坏不抛——下次写时会覆盖成合法的；上游一致看到空 manifest
        return _empty_manifest()


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """tmp+rename 原子写。同分区写入 + os.replace 保证 reader 永远看到完整 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)  # 跨平台原子 rename


# ---------------------------------------------------------------------------
# Resolver — 下游统一入口
# ---------------------------------------------------------------------------


def entry_origin(entry: dict[str, Any], fallback_name: str) -> str:
    """从一条 entry 提取 origin（指向 download/{...} 的文件名）。

    缺 `origin` 则用 entry 自身的 key（1:1 同名兜底）。
    """
    return entry.get("origin") or fallback_name


def is_duplicate_removed_entry(entry: Optional[dict[str, Any]]) -> bool:
    """是否为人工去重审核确认跳过的 manifest entry。"""
    return bool(entry and entry.get("kind") == DUPLICATE_REMOVED_KIND)


def resolve(project_dir: Path, name: str) -> Optional[Path]:
    """给定产物文件名（如 `foo.png`），返回它实际指向的磁盘路径。

    隐式 original   → `download/{name}`（即使该图不存在；resolver 不做存在性检查）
    manifest 有 entry → `preprocess/{name}`

    存在性由调用方按需 `.exists()` 检查——这样列图时一次 stat 即可，不重复。
    """
    m = load(project_dir)
    entry = m["images"].get(name)
    if entry is None:
        return project_dir / "download" / name
    return project_dir / "preprocess" / name


def resolve_origin(project_dir: Path, download_name: str) -> list[Path]:
    """反向 resolve：给一个 download/{name}，列出 preprocess/ 里所有派生产物。

    - manifest 有 processed entries with `origin == download_name` → 返回它们 [preprocess/X]
    - 只有 duplicate_removed entries 追溯到该 origin → 返回 []（下游跳过）
    - 没有匹配 entry → 回退到 [download/download_name]（隐式 original）
    """
    m = load(project_dir)
    removed = False
    matches: list[Path] = []
    for name, entry in m["images"].items():
        if entry_origin(entry, name) != download_name:
            continue
        if is_duplicate_removed_entry(entry):
            removed = True
            continue
        matches.append(project_dir / "preprocess" / name)
    if matches:
        return matches
    if removed:
        return []
    return [project_dir / "download" / download_name]


def get_entry(project_dir: Path, name: str) -> Optional[dict[str, Any]]:
    """读单条 entry（不存在返 None）。给 thumb endpoint resolve_origin fallback 用。"""
    m = load(project_dir)
    return m["images"].get(name)


# ---------------------------------------------------------------------------
# ADR 0010 — per-version train/ manifest（fallback 重建）
#
# 新模型把 preprocess 产物落到 versions/{label}/train/，状态记到同位
# manifest.json。本节只暴露 fallback 入口：第一次访问某 version 的 train
# manifest 时，按老 project 级 preprocess/manifest.json 隐式重建。
#
# 详 docs/adr/0010-preprocess-train-scope.md + docs/design/preprocess-train-scope-plan.md §3.2。
# 重写逻辑只读老 manifest 元数据，不复制图像 bytes（train/ 已是处理后产物
# 由 curate 阶段复制进去，新模型唯一丢失的是 origin 反查关系）。
# ---------------------------------------------------------------------------

TRAIN_MANIFEST_VERSION = 2


def train_manifest_path(project_dir: Path, version_label: str) -> Path:
    return project_dir / "versions" / version_label / "train" / MANIFEST_NAME


def _scan_train_images(train_dir: Path) -> set[str]:
    """递归 train_dir 一级 sub-folder 收集图片相对路径（POSIX 形式）。

    LoRA 训练用 repeat folder 结构：`train/1_data/X.png` 而不是 `train/X.png`
    （`{N_label}/{image}` 由 dataset_config.toml 解析）。manifest entry key
    用 POSIX 相对路径表达跨 folder 唯一性（同名图可在多个 folder 重复出现）。

    根目录直接放的图忽略（不该有，但防御）；非 image 文件（caption .txt /
    arbitrary）也忽略。
    """
    from ..dataset.scan import IMAGE_EXTS

    if not train_dir.exists():
        return set()
    rel_paths: set[str] = set()
    for sub in train_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                rel_paths.add(f"{sub.name}/{f.name}")
    return rel_paths


def _build_train_manifest_from_legacy(
    legacy: dict[str, Any], train_rel_paths: set[str]
) -> dict[str, Any]:
    """从老 project 级 manifest 抽出 train/ 里实际存在的图的 origin 关系。

    老 manifest entry name 是平铺产物名（如 `X.png`，不含 folder 前缀）。
    新 train/ 实际位置在 sub-folder 里（如 `1_data/X.png`）。匹配规则：

    - 按文件名（rel path 的末段）建索引
    - 老 entry name 找到任意同名 train 文件 → 用该 train rel path 作新 key
    - 跨 sub-folder 同名 → 给每个匹配的 rel path 各加一条 entry（保守，让
      用户在 UI 自决；罕见但合法）

    跳过：(1) train/ 没匹配文件名的 entry、(2) duplicate_removed 老 entry
    （人工去重审核状态不跨模型迁移；新模型走 version 级独立记录）。
    """
    by_filename: dict[str, list[str]] = {}
    for rel in train_rel_paths:
        nm = rel.rsplit("/", 1)[-1]
        by_filename.setdefault(nm, []).append(rel)

    images: dict[str, Any] = {}
    for name, entry in legacy.get("images", {}).items():
        if not isinstance(entry, dict):
            continue
        if is_duplicate_removed_entry(entry):
            continue
        rels = by_filename.get(name)
        if not rels:
            continue
        for rel in rels:
            images[rel] = {
                "origin": entry_origin(entry, name),
                "mtime": entry.get("mtime", 0),
                "size": entry.get("size", 0),
            }
    return {"version": TRAIN_MANIFEST_VERSION, "images": images}


def ensure_train_manifest(project_dir: Path, version_label: str) -> Path:
    """幂等：保证 versions/{label}/train/manifest.json 存在；返回路径。

    Fallback 重建规则（详 ADR 0010 §决策）：

    1. 目标已存在 → 直接返回（O(1) stat，热路径无开销）
    2. 不存在 + 老 `preprocess/manifest.json` 存在 → 按 train/ 实际文件名
       匹配老 entry origin 重建 v2 schema
    3. 老 manifest 也不存在 / 损坏 → 写空 v2 manifest

    train/ 目录不存在时**也会创建**（首次访问该 version 时该目录可能还空）。

    所有 train manifest read 入口都该先过这一道（防御性，幂等代价 = 1 次
    stat）。fork version 时（`versions.py:create_version`）也显式调一次防止
    源 manifest 损坏。

    PR-1 范围：本函数 + 测试。**调用点的接入在 PR-2 范围**（manifest 模块
    瘦身时一并接进所有 read/write 入口）。
    """
    target = train_manifest_path(project_dir, version_label)
    if target.exists():
        return target

    train_dir = target.parent
    legacy_path = manifest_path(project_dir)  # 项目级老 manifest

    with _LOCK:
        # 双检查：拿锁后再看一次（可能别人刚建完）
        if target.exists():
            return target

        train_dir.mkdir(parents=True, exist_ok=True)

        # 收集 train/ 里的图（递归一级 sub-folder，LoRA repeat folder 结构）
        train_rel_paths = _scan_train_images(train_dir)

        # 读老 manifest（不存在 / 损坏 → 空，跟 load() 一致语义）
        legacy: dict[str, Any]
        if legacy_path.exists():
            try:
                raw = json.loads(legacy_path.read_text(encoding="utf-8"))
                legacy = raw if isinstance(raw, dict) else {}
            except (OSError, json.JSONDecodeError):
                legacy = {}
        else:
            legacy = {}

        manifest = _build_train_manifest_from_legacy(legacy, train_rel_paths)
        _atomic_write(target, manifest)
        return target


# ---------------------------------------------------------------------------
# ADR 0010 — train-scope manifest API
#
# 项目 scope 老 API 已删；本节是当前唯一 mutation API。
#
# 关键语义：
# - manifest 落 `versions/{label}/train/manifest.json`
# - entry key 用 **POSIX 相对路径**（如 `"1_data/X.png"`），表达 LoRA repeat
#   folder 结构（`train/{N_label}/{image}`）；跨 folder 同名图各自独立 entry
# - `train_restore(name)` = 从 `download/{entry.origin}` 复制覆盖回 `train/{name}`
#   （不是删 entry；详 ADR 0010 §Restore 语义）；缺 origin 文件时 → no_origin 列表
# - `train_add_processed` size 兜底 stat `train/{name}`
# - 所有 train_xxx 进 mutation 前先调 ensure_train_manifest（防御性，幂等）
#
# 锁仍用模块单 `_LOCK`（version 写不频繁，单锁可接受）。
# ---------------------------------------------------------------------------


def _train_dir(project_dir: Path, version_label: str) -> Path:
    return project_dir / "versions" / version_label / "train"


def _empty_train_manifest() -> dict[str, Any]:
    return {"version": TRAIN_MANIFEST_VERSION, "images": {}}


def _read_train_target(target: Path) -> dict[str, Any]:
    """读 train manifest 文件（target 已知存在）；损坏 → 空 v2 manifest。

    设计跟老 `load()` 一致——损坏不抛，下次写时覆盖。callers 内部用，跟
    `ensure_train_manifest` 配套（callers 已 ensure 过 target 存在）。
    """
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("images"), dict):
            return raw
    except (OSError, json.JSONDecodeError):
        pass
    return _empty_train_manifest()


# ---- read ----------------------------------------------------------------


def train_load(project_dir: Path, version_label: str) -> dict[str, Any]:
    """读 train manifest；不存在则 fallback 重建（详 ADR 0010 §Fallback 重建机制）。

    返回完整 manifest dict `{"version": 2, "images": {...}}`。
    """
    target = ensure_train_manifest(project_dir, version_label)
    return _read_train_target(target)


def train_get_entry(
    project_dir: Path, version_label: str, name: str
) -> Optional[dict[str, Any]]:
    return train_load(project_dir, version_label)["images"].get(name)


def train_all_processed(
    project_dir: Path, version_label: str
) -> dict[str, dict[str, Any]]:
    """非 duplicate_removed 的 entry。"""
    m = train_load(project_dir, version_label)
    return {
        name: entry
        for name, entry in m["images"].items()
        if not is_duplicate_removed_entry(entry)
    }


def train_duplicate_removed(
    project_dir: Path, version_label: str
) -> dict[str, dict[str, Any]]:
    m = train_load(project_dir, version_label)
    return {
        name: entry
        for name, entry in m["images"].items()
        if is_duplicate_removed_entry(entry)
    }


def train_duplicate_removed_origins(
    project_dir: Path, version_label: str
) -> set[str]:
    return {
        entry_origin(entry, name)
        for name, entry in train_duplicate_removed(
            project_dir, version_label
        ).items()
    }


# ---- mutation（必须 with _LOCK）-----------------------------------------


def train_add_processed(
    project_dir: Path,
    version_label: str,
    name: str,
    meta: dict[str, Any],
) -> None:
    """记录一张已处理图（train scope）。

    schema：采纳 `origin / mtime / size / processed`，其他字段（model/scale/
    action/...）丢弃。size 兜底 stat `train/{name}`。

    `processed: bool`（ADR 0010 fixup 2026-06-04）：worker upscale/crop 完成
    后传 `meta["processed"] = True` 标记，curate 时 `copy_download_to_train`
    不传（默认 False 不写字段）。前端用这个字段画"已处理"徽章；详 ADR 0010
    §状态从字段差异隐含推断。
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        origin = meta.get("origin") or name
        entry: dict[str, Any] = {
            "origin": origin,
            "mtime": meta.get("mtime", time.time()),
        }
        if "size" in meta:
            entry["size"] = meta["size"]
        else:
            png = _train_dir(project_dir, version_label) / name
            try:
                entry["size"] = png.stat().st_size
            except OSError:
                entry["size"] = 0
        if meta.get("processed"):
            entry["processed"] = True
        m["images"][name] = entry
        _atomic_write(target, m)


def train_replace_with_crops(
    project_dir: Path,
    version_label: str,
    *,
    source_name: str,
    outputs: list[dict[str, Any]],
) -> None:
    """multi-crop fan-out：把 `source_name` 替换成 N 个 crop 产物 entry。

    操作跟老 `replace_with_crops` 一致——找出所有 origin 与 source_name 匹配
    的旧 entry + source_name 自身全部删除，写入 N 个新 entry（origin 沿用
    旧 entry origin 或回退 source_name）。

    磁盘文件（train/{name}.png 等）由调用方负责，本函数只动 manifest。
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        to_remove = {source_name}
        for nm, entry in m["images"].items():
            if entry_origin(entry, nm) == source_name:
                to_remove.add(nm)
        for nm in to_remove:
            m["images"].pop(nm, None)
        now = time.time()
        for o in outputs:
            entry = {
                "origin": o.get("origin") or source_name,
                "mtime": o.get("mtime", now),
                "size": int(o.get("size", 0)),
            }
            # crop 派生本质是处理操作（ADR 0010 fixup）
            if o.get("processed", True):
                entry["processed"] = True
            m["images"][o["name"]] = entry
        _atomic_write(target, m)


def train_mark_duplicate_removed(
    project_dir: Path,
    version_label: str,
    names: list[str],
) -> dict[str, list[str]]:
    """去重移除（train scope）：物理删除 train/{name} + caption sidecar，manifest
    entry 改为 `kind=duplicate_removed` 作 tombstone（用于总览页"已删除"tab +
    `train_restore_duplicate_removed` 恢复）。

    下游 tagging / training 直接扫 `train/`，物理删除保证图不再出现在 caption
    队列 / dataset_config 列表里。

    每个 version 独立审核（manifest 是 version 级）。fork 时整树复制
    （ADR 0007 `_copytree("train")`）只带物理文件——tombstone 同样会复制因为
    manifest 也在 train/ 下。
    """
    removed: list[str] = []
    missing: list[str] = []
    skipped: list[str] = []
    now = time.time()
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    train_dir = _train_dir(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        for name in names:
            entry = m["images"].get(name)
            if is_duplicate_removed_entry(entry):
                skipped.append(name)
                continue
            src = train_dir / name
            if entry is not None:
                origin = entry_origin(entry, name)
                size = int(entry.get("size", 0) or 0)
            elif src.is_file():
                origin = name
                try:
                    size = src.stat().st_size
                except OSError:
                    size = 0
            else:
                missing.append(name)
                continue
            # 物理删图 + caption sidecar
            if src.is_file():
                try:
                    src.unlink()
                except OSError:
                    pass
            for ext in (".txt", ".json"):
                sidecar = src.with_suffix(ext)
                if sidecar.is_file():
                    try:
                        sidecar.unlink()
                    except OSError:
                        pass
            m["images"][name] = {
                "kind": DUPLICATE_REMOVED_KIND,
                "origin": origin,
                "mtime": now,
                "size": size,
            }
            removed.append(name)
        _atomic_write(target, m)
    return {"removed": removed, "missing": missing, "skipped": skipped}


def train_restore_duplicate_removed(
    project_dir: Path,
    version_label: str,
    names: list[str],
) -> dict[str, list[str]]:
    """撤销去重移除：从 `download/{entry.origin}` 复制图 + caption 覆盖回
    `train/{name}`，并删 manifest entry。

    返回三组：
    - `restored`：成功复原（download 原图存在并已复制覆盖）
    - `missing`：name 没 entry 或 entry 不是 duplicate_removed
    - `no_origin`：entry 是 duplicate_removed 但 `download/{origin}` 物理文件
      缺失——entry 保留，调用方提示用户从外部 import 原图
    """
    import shutil

    restored: list[str] = []
    missing: list[str] = []
    no_origin: list[str] = []
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    download_dir = project_dir / "download"
    train_dir = _train_dir(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        for name in names:
            entry = m["images"].get(name)
            if not is_duplicate_removed_entry(entry):
                missing.append(name)
                continue
            origin = entry_origin(entry, name)
            src = download_dir / origin
            if not src.is_file():
                no_origin.append(name)
                continue
            dst = train_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except OSError:
                no_origin.append(name)
                continue
            # caption 跟随：download/{origin_stem}.{ext} → train/{name_stem}.{ext}
            origin_stem = Path(origin).stem
            for ext in (".txt", ".json"):
                cap_src = download_dir / f"{origin_stem}{ext}"
                if cap_src.is_file():
                    try:
                        shutil.copy2(cap_src, dst.with_suffix(ext))
                    except OSError:
                        pass
            del m["images"][name]
            restored.append(name)
        _atomic_write(target, m)
    return {"restored": restored, "missing": missing, "no_origin": no_origin}


def train_restore(
    project_dir: Path,
    version_label: str,
    names: list[str],
) -> dict[str, list[str]]:
    """复原：从 `download/{entry.origin}` 复制覆盖回 `train/{folder}/{origin}`。

    Multi-crop fan-out 折叠：若 `name` 是某 fan-out 组的成员（同 folder 内多个
    entry 共享同一 origin），整组被复原到 `train/{folder}/{origin}` 一张图，
    sibling 物理文件 + manifest entry + caption sidecar 一并清理。这保证撤销
    a_0/a_1 不会得到"两张同名 A 副本"。

    Caption 跟随：从 `download/{origin_stem}.{ext}` 拷到
    `train/{folder}/{origin_stem}.{ext}`（`.txt` / `.json`）。

    返回三组：
    - `restored`：成功复原的 *输入* name（fan-out 组里其他 sibling 即使被一并
      清理也单独 list 在 restored 里，方便 UI 对账）
    - `missing`：name 在 manifest 没 entry（且不在已被本批次清理过的 sibling 里）
    - `no_origin`：entry 存在但 `download/{origin}` 物理文件缺失——UI 应该
      给用户三选项（拖入替换 / 保留处理后版本 / 从 train 移除）
    """
    import shutil

    restored: list[str] = []
    missing: list[str] = []
    no_origin: list[str] = []
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    download_dir = project_dir / "download"
    train_dir = _train_dir(project_dir, version_label)
    # batch 内 already-handled siblings（避免重复 copy / 误报 missing）
    handled: set[str] = set()
    with _LOCK:
        m = _read_train_target(target)
        for name in names:
            if name in handled:
                restored.append(name)
                continue
            entry = m["images"].get(name)
            if entry is None:
                missing.append(name)
                continue
            origin = entry_origin(entry, name)
            folder = name.rsplit("/", 1)[0] if "/" in name else ""
            src = download_dir / origin
            if not src.is_file():
                no_origin.append(name)
                continue
            # 找 fan-out 组：同 folder 下 origin 一致的全部 entry
            group: list[str] = [
                k for k, e in m["images"].items()
                if (k.rsplit("/", 1)[0] if "/" in k else "") == folder
                and entry_origin(e, k) == origin
            ]
            dst_rel = f"{folder}/{origin}" if folder else origin
            dst = train_dir / dst_rel
            # 删 sibling 物理 + caption（dst 本身先不删——下面 copy 会覆盖）
            for sib in group:
                if sib == dst_rel:
                    continue
                sib_path = train_dir / sib
                if sib_path.is_file():
                    try:
                        sib_path.unlink()
                    except OSError:
                        pass
                for ext in (".txt", ".json"):
                    side = sib_path.with_suffix(ext)
                    if side.is_file():
                        try:
                            side.unlink()
                        except OSError:
                            pass
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except OSError:
                no_origin.append(name)
                continue
            # caption sidecar 跟随
            origin_stem = Path(origin).stem
            for ext in (".txt", ".json"):
                cap_src = download_dir / f"{origin_stem}{ext}"
                if cap_src.is_file():
                    try:
                        shutil.copy2(cap_src, dst.with_suffix(ext))
                    except OSError:
                        pass
            # manifest：删整组 + 写新 entry at dst_rel
            for sib in group:
                m["images"].pop(sib, None)
            try:
                st = src.stat()
                m["images"][dst_rel] = {
                    "origin": origin,
                    "mtime": int(st.st_mtime),
                    "size": st.st_size,
                }
            except OSError:
                m["images"][dst_rel] = {"origin": origin}
            restored.append(name)
            handled.update(group)
        _atomic_write(target, m)
    return {"restored": restored, "missing": missing, "no_origin": no_origin}


def train_swap_entry(
    project_dir: Path,
    version_label: str,
    old_name: str,
    new_name: str,
    meta: dict[str, Any],
) -> None:
    """原子替换 train manifest entry：删 `old_name`，写 `new_name`。

    给 worker 在 upscale 输出扩展名变化时用（如 src=`1_data/X.jpg` →
    dst=`1_data/X.png`），避免 manifest 残留 dangling 老 entry。

    `meta` 跟 `train_add_processed` 一致——只采纳 origin/mtime/size，其他丢弃。
    size 兜底 stat `train/{new_name}`。
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        m["images"].pop(old_name, None)
        origin = meta.get("origin") or new_name
        entry: dict[str, Any] = {
            "origin": origin,
            "mtime": meta.get("mtime", time.time()),
        }
        if "size" in meta:
            entry["size"] = meta["size"]
        else:
            png = _train_dir(project_dir, version_label) / new_name
            try:
                entry["size"] = png.stat().st_size
            except OSError:
                entry["size"] = 0
        if meta.get("processed"):
            entry["processed"] = True
        m["images"][new_name] = entry
        _atomic_write(target, m)


def train_remove_entries(
    project_dir: Path,
    version_label: str,
    names: list[str],
) -> int:
    """批量删 train manifest entries（按 entry key）。返回实际删除数。

    给 `curation.remove_from_train` 用：用户从 train 删一张 download 原图时，
    后者按 origin 反查得到 N 个派生 rel path（multi-crop fan-out），一次性
    pop + 原子写盘比逐条 mutation 高效。
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    removed = 0
    with _LOCK:
        m = _read_train_target(target)
        for name in names:
            if m["images"].pop(name, None) is not None:
                removed += 1
        if removed:
            _atomic_write(target, m)
    return removed


def train_clear_all(project_dir: Path, version_label: str) -> None:
    """清空本 version 的 train manifest 状态——只清 manifest 文件，**不动**
    train/ 物理文件。

    跟老 `clear_all` 语义不同——老的删 preprocess/ PNG 物理产物；新模型下
    train/ 是训练数据本身，删物理文件 = 删训练集，不该由"清空预处理状态"
    引发。调用方如果想完全重做预处理，应该改成"对每张图调 train_restore"
    （复原到 download 原图），不调本函数。

    本函数提供给极端场景（manifest 损坏到不可读）做"清零重建"用。
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    with _LOCK:
        _atomic_write(target, _empty_train_manifest())
