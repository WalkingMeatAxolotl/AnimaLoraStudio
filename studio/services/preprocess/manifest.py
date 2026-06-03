"""预处理状态 manifest（单 JSON 文件，project 级）。

设计见 [ADR 0004](../../docs/adr/0004-preprocess-manifest.md)
和 [crop design](../../docs/design/preprocess-crop-design.md)。

简而言之
--------
`projects/{id}-{slug}/preprocess/manifest.json` 记录**非默认**的预处理决定。

新 schema（写入用）— 极简，只追溯 origin：

    {
      "images": {
        "X.png":    {"origin": "X.png",  "mtime": 1731000000, "size": 1234567},
        "Y_c0.png": {"origin": "Y.png",  "mtime": ...,         "size": ...},
        "Y_c1.png": {"origin": "Y.png",  "mtime": ...,         "size": ...}
      }
    }

老 schema（读时兼容，几个 version 后 deprecate）：

    {"kind": "processed", "source": "bar.jpg", "model": ..., "scale": ...,
     "action": ..., "target_area": ..., "src_size": ..., "dst_size": ..., ...}

读规则：
- `origin` 字段优先；缺失则回退 `source` 字段；都没就用 entry key 自身
- `kind` 字段可有可无；entry 存在且没有显式非 processed kind 时视为"已处理"
- `kind: "duplicate_removed"` 表示人工审核确认跳过该图；不移动 / 删除 download
- 其他字段（model/scale/action/...）读时透传，写时不再产生

「manifest 没记的图」= 用 download/ 原图（隐式 original）。
所有下游（curation 左侧 / thumbnail / copy_to_train）走 `resolve()` 单点拿
实际文件路径。

并发写
------
服务端单进程，没跨进程写者：`threading.Lock` 串行化进程内所有 mutation
（worker 通过 supervisor + 共享内存模型时也走同一把锁）。如果未来出现
跨进程写，升级到 portalocker，函数签名不变。

迁移
----
老项目里有 `*.preprocess.json` per-image sidecar（`studio/preprocess.py:SIDECAR_SUFFIX`）。
`ensure_manifest()` 第一次发现没有 manifest 但有 sidecar → 聚合写一份。
老 sidecar 保留不删，新代码不再读它们。
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

# manifest 文件 + 旧 sidecar 后缀（migration 用）
MANIFEST_NAME = "manifest.json"
LEGACY_SIDECAR_SUFFIX = ".preprocess.json"
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

    优先 `origin`（新 schema），缺失则用 `source`（老 schema），都缺就用 entry
    自身的 key（兼容兜底 — 表示 1:1 同名）。
    """
    return entry.get("origin") or entry.get("source") or fallback_name


def is_duplicate_removed_entry(entry: Optional[dict[str, Any]]) -> bool:
    """是否为人工去重审核确认跳过的 manifest entry。"""
    return bool(entry and entry.get("kind") == DUPLICATE_REMOVED_KIND)


def resolve(project_dir: Path, name: str) -> Optional[Path]:
    """给定产物文件名（如 `foo.png`），返回它实际指向的磁盘路径。

    隐式 original   → `download/{name}`（即使该图不存在；resolver 不做存在性检查）
    manifest 有 entry → `preprocess/{name}`（新 / 老 schema 都按"已处理"算）

    存在性由调用方按需 `.exists()` 检查——这样列图时一次 stat 即可，不重复。

    历史：旧版本用 `kind != "processed"` 区分"不可解析"的未来状态。新
    schema 不再写 `kind`；只要 entry 存在即视为已处理。读老 entry 若 `kind`
    显式不是 "processed" 仍按未来扩展处理（返 None）。
    """
    m = load(project_dir)
    entry = m["images"].get(name)
    if entry is None:
        return project_dir / "download" / name
    kind = entry.get("kind")
    if kind is not None and kind != "processed":
        # 未知 / 未来扩展（老 schema 才出现 kind != processed） → 不可解析
        return None
    return project_dir / "preprocess" / name


def resolve_origin(project_dir: Path, download_name: str) -> list[Path]:
    """反向 resolve：给一个 download/{name}，列出 preprocess/ 里所有派生产物。

    - manifest 有 processed entries with `origin == download_name` → 返回它们 [preprocess/X]
    - 只有 duplicate_removed entries 追溯到该 origin → 返回 []（下游跳过）
    - 没有匹配 entry → 回退到 [download/download_name]（隐式 original）

    给下游 copy_to_train / curation / 缩略图 用：一张原图可能被 multi-crop 切成
    多张，需要全部喂下去。
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
        if entry.get("kind", "processed") == "processed":
            matches.append(project_dir / "preprocess" / name)
    if matches:
        return matches
    if removed:
        return []
    return [project_dir / "download" / download_name]


def is_origin_duplicate_removed(project_dir: Path, download_name: str) -> bool:
    """给 download/{name} 判断是否已被人工去重审核标记跳过。"""
    return download_name in duplicate_removed_origins(project_dir)


def get_entry(project_dir: Path, name: str) -> Optional[dict[str, Any]]:
    """读单条 entry（不存在返 None）。给 list_processed 拼元数据用。"""
    m = load(project_dir)
    return m["images"].get(name)


def all_processed(project_dir: Path) -> dict[str, dict[str, Any]]:
    """返回 `{name: entry}` 所有"已处理" entry。

    新 schema：任何 entry 都算已处理（无 `kind` 字段）。
    老 schema：兼容性按 `kind == "processed"` 过滤；非 processed 视为未来扩展。
    """
    m = load(project_dir)
    return {
        name: entry
        for name, entry in m["images"].items()
        if entry.get("kind", "processed") == "processed"
    }


def duplicate_removed(project_dir: Path) -> dict[str, dict[str, Any]]:
    """返回 `{name: entry}` 所有人工去重审核确认跳过的 entry。"""
    m = load(project_dir)
    return {
        name: entry
        for name, entry in m["images"].items()
        if is_duplicate_removed_entry(entry)
    }


def duplicate_removed_origins(project_dir: Path) -> set[str]:
    """返回所有被去重审核移除的 root origin 名。

    用于阻止 removed preprocess 派生图回退成 download 原图。
    """
    return {
        entry_origin(entry, name)
        for name, entry in duplicate_removed(project_dir).items()
    }


# ---------------------------------------------------------------------------
# Mutation — 必须 with _LOCK
# ---------------------------------------------------------------------------


def add_processed(project_dir: Path, name: str, meta: dict[str, Any]) -> None:
    """记录一张已处理图。

    新 schema 只写 `{origin, mtime, size}`：worker meta 可能带很多过程信息
    （model/scale/action/...），这里**只采纳 origin / mtime / size**，其他
    字段丢弃。这是 schema 简化的一部分（见 module docstring）。

    `origin` 从 meta 的 `origin` 或 `source` 取；都没就用 entry name 自身
    （= 1:1 同名场景）。`mtime` 用 meta 的或 time.time()。`size` 用 meta 的
    或对 preprocess/{name} 一次 stat。
    """
    with _LOCK:
        m = load(project_dir)
        origin = meta.get("origin") or meta.get("source") or name
        entry: dict[str, Any] = {
            "origin": origin,
            "mtime": meta.get("mtime", time.time()),
        }
        if "size" in meta:
            entry["size"] = meta["size"]
        else:
            png = project_dir / "preprocess" / name
            try:
                entry["size"] = png.stat().st_size
            except OSError:
                entry["size"] = 0
        m["images"][name] = entry
        _atomic_write(manifest_path(project_dir), m)


def replace_with_crops(
    project_dir: Path,
    *,
    source_name: str,
    outputs: list[dict[str, Any]],
) -> None:
    """把 `source_name`（preprocess/ 当前文件名）替换为 N 个 crop 产物。

    操作：
      - 找出所有 origin 与 source_name 相同的旧 entry → 全部删除
      - 删除 `source_name` 这条 entry（即使 origin 不匹配；兜底处理 1:1 同名场景）
      - 写入 N 个新 entry，每条形如 {"origin": <root origin>, "mtime", "size"}

    `outputs` 每项：`{"name": "X_c0.png", "origin": "X.png", "mtime": ..., "size": ...}`。
    多裁剪的 `origin` 应**沿用旧 entry 的 origin**（如果有），保证 origin 始终
    指向 download/ 而不是中间产物（即 `X_c0_c0.png` 的 origin 还是 `X.png`）。
    worker 决定 origin 值；本函数不做派生。

    磁盘文件（preprocess/{source_name}.png 等）由调用方负责删除——本函数只动 manifest。
    """
    with _LOCK:
        m = load(project_dir)
        # 删旧：source_name 自身 + 所有 origin 匹配的派生
        to_remove = {source_name}
        for nm, entry in m["images"].items():
            if entry_origin(entry, nm) == source_name:
                to_remove.add(nm)
        for nm in to_remove:
            m["images"].pop(nm, None)
        # 写新
        now = time.time()
        for o in outputs:
            entry: dict[str, Any] = {
                "origin": o.get("origin") or source_name,
                "mtime": o.get("mtime", now),
                "size": int(o.get("size", 0)),
            }
            m["images"][o["name"]] = entry
        _atomic_write(manifest_path(project_dir), m)


def mark_duplicate_removed(project_dir: Path, names: list[str]) -> dict[str, list[str]]:
    """标记人工审核确认去重移除的图，不移动 / 删除任何图片文件。

    `names` 可以是 curation 工作集里的 download 原图名，也可以是 preprocess
    派生产物名。processed entry 会被就地改成 `kind=duplicate_removed`；
    隐式 original 会新增一条同名 entry。
    """
    removed: list[str] = []
    missing: list[str] = []
    skipped: list[str] = []
    now = time.time()
    with _LOCK:
        m = load(project_dir)
        for name in names:
            entry = m["images"].get(name)
            if is_duplicate_removed_entry(entry):
                skipped.append(name)
                continue
            if entry is not None:
                if entry.get("kind", "processed") != "processed":
                    skipped.append(name)
                    continue
                origin = entry_origin(entry, name)
                size = int(entry.get("size", 0) or 0)
            else:
                src = project_dir / "download" / name
                if not src.is_file():
                    missing.append(name)
                    continue
                origin = name
                try:
                    size = src.stat().st_size
                except OSError:
                    size = 0
            m["images"][name] = {
                "kind": DUPLICATE_REMOVED_KIND,
                "origin": origin,
                "mtime": now,
                "size": size,
            }
            removed.append(name)
        _atomic_write(manifest_path(project_dir), m)
    return {"removed": removed, "missing": missing, "skipped": skipped}


def restore(project_dir: Path, names: list[str]) -> dict[str, list[str]]:
    """还原：删 manifest entry + 删 preprocess/{name} PNG。

    回到「隐式 original」——下游 resolve 会重新指向 download/。
    返回 `{restored, missing}`：manifest 里没的记 missing（PNG 没的不算 missing）。
    """
    preprocess_dir = project_dir / "preprocess"
    restored: list[str] = []
    missing: list[str] = []
    with _LOCK:
        m = load(project_dir)
        for name in names:
            if name in m["images"]:
                del m["images"][name]
                restored.append(name)
            else:
                missing.append(name)
            # PNG 不在 manifest 也照删（自愈：orphan PNG 一并清掉）
            png = preprocess_dir / name
            if png.is_file():
                try:
                    png.unlink()
                except OSError:
                    pass
        _atomic_write(manifest_path(project_dir), m)
    return {"restored": restored, "missing": missing}


def clear_all(project_dir: Path) -> None:
    """整项目预处理状态归零：删全部 entry + 删 preprocess/ 下所有 PNG。

    sidecar / manifest.json 本身保留（写空 manifest）。给「重置该项目」操作用。
    """
    preprocess_dir = project_dir / "preprocess"
    with _LOCK:
        if preprocess_dir.exists():
            for f in preprocess_dir.iterdir():
                if f.is_file() and f.suffix.lower() == ".png":
                    try:
                        f.unlink()
                    except OSError:
                        pass
        _atomic_write(manifest_path(project_dir), _empty_manifest())


def restore_duplicate_removed(project_dir: Path, names: list[str]) -> dict[str, list[str]]:
    """撤销去重移除标记：只删 duplicate_removed entry，不碰磁盘文件。"""
    restored: list[str] = []
    missing: list[str] = []
    with _LOCK:
        m = load(project_dir)
        for name in names:
            entry = m["images"].get(name)
            if is_duplicate_removed_entry(entry):
                del m["images"][name]
                restored.append(name)
            else:
                missing.append(name)
        _atomic_write(manifest_path(project_dir), m)
    return {"restored": restored, "missing": missing}


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _scan_legacy_sidecars(preprocess_dir: Path) -> dict[str, dict[str, Any]]:
    """扫 `preprocess/*.preprocess.json` → 聚合成 manifest entries。

    sidecar 文件名约定：`{product_stem}.png.preprocess.json`（见 upscaler 历史
    实现）。entry key 取产物 PNG 名（去掉 `.preprocess.json` 后剩 `.png`）。
    """
    out: dict[str, dict[str, Any]] = {}
    if not preprocess_dir.exists():
        return out
    for sidecar in preprocess_dir.iterdir():
        if not sidecar.is_file() or not sidecar.name.endswith(LEGACY_SIDECAR_SUFFIX):
            continue
        png_name = sidecar.name[: -len(LEGACY_SIDECAR_SUFFIX)]
        # 仅迁移那些产物 PNG 实际存在的（防止 sidecar 残留指向已删图）
        if not (preprocess_dir / png_name).is_file():
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        out[png_name] = {"kind": "processed", **meta}
    return out


def ensure_manifest(project_dir: Path) -> dict[str, Any]:
    """幂等入口：如果 manifest 已存在直接返回；否则从老 sidecar 迁移一次。

    所有列图 / resolve 调用点都该先过这一道，确保老项目第一次访问就 migrate。
    迁移完老 sidecar 保留不删（防御性回滚）。
    """
    path = manifest_path(project_dir)
    if path.exists():
        return load(project_dir)
    preprocess_dir = project_dir / "preprocess"
    with _LOCK:
        # 双检查：拿到锁后再看一次（可能别人刚 migrate 完）
        if path.exists():
            return load(project_dir)
        migrated = _scan_legacy_sidecars(preprocess_dir)
        manifest = {"images": migrated}
        _atomic_write(path, manifest)
        return manifest


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
# ADR 0010 — train-scope manifest API（PR-2 step A）
#
# 老的 project-scope API（load / add_processed / restore / mark_duplicate_removed /
# clear_all / replace_with_crops / get_entry / all_processed / duplicate_removed*）
# 暂留作 backward-compat 给老 endpoint 用，PR-3 删。新代码 / 新调用方请用
# `train_xxx` 系列。
#
# 关键语义差异（vs ADR 0004 老 API）：
# - manifest 落 `versions/{label}/train/manifest.json`（PR-1 已加 ensure 路径）
# - entry key 用 **POSIX 相对路径**（如 `"1_data/X.png"`），表达 LoRA repeat
#   folder 结构（`train/{N_label}/{image}`）；跨 folder 同名图各自独立 entry
# - `train_restore(name)` = 从 `download/{entry.origin}` 复制覆盖回 `train/{name}`
#   （不是删 entry；详 ADR 0010 §Restore 语义）；缺 origin 文件时 → no_origin 列表
# - `train_add_processed` size 兜底 stat `train/{name}`（不是老 `preprocess/{name}`）
# - 所有 train_xxx 进 mutation 前先调 ensure_train_manifest（防御性，幂等）
#
# 锁仍用模块单 `_LOCK`（plan §12.2.1 推荐：version 写不频繁，单锁可接受）。
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
    """非 duplicate_removed 的 entry 视为"已处理"——对新 schema 是 entry 存在
    即视为已处理（隐含状态推断；详 ADR 0010 §Manifest schema v2）。
    """
    m = train_load(project_dir, version_label)
    return {
        name: entry
        for name, entry in m["images"].items()
        if entry.get("kind", "processed") == "processed"
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

    schema 跟老 `add_processed` 一致——只采纳 `origin / mtime / size`，其他
    字段（model/scale/action/...）丢弃。size 兜底 stat `train/{name}`。
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        origin = meta.get("origin") or meta.get("source") or name
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
            m["images"][o["name"]] = entry
        _atomic_write(target, m)


def train_mark_duplicate_removed(
    project_dir: Path,
    version_label: str,
    names: list[str],
) -> dict[str, list[str]]:
    """标记人工审核去重移除（train scope）。

    不动磁盘文件——保留 train/{name} 作"已审核但跳过"标记。下游 curate /
    training 通过查 manifest 判断该图是否参与训练。

    每个 version 独立审核（manifest 是 version 级）。fork 时整树复制
    （ADR 0007 `_copytree("train")`）会把 duplicate_removed 状态带过去。
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
            if entry is not None:
                if entry.get("kind", "processed") != "processed":
                    skipped.append(name)
                    continue
                origin = entry_origin(entry, name)
                size = int(entry.get("size", 0) or 0)
            else:
                src = train_dir / name
                if not src.is_file():
                    missing.append(name)
                    continue
                origin = name
                try:
                    size = src.stat().st_size
                except OSError:
                    size = 0
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
    """撤销去重移除标记——只删 duplicate_removed entry，不动 train/ 物理文件。

    跟老 `restore_duplicate_removed` 同语义，train scope 版。
    """
    restored: list[str] = []
    missing: list[str] = []
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        for name in names:
            entry = m["images"].get(name)
            if is_duplicate_removed_entry(entry):
                del m["images"][name]
                restored.append(name)
            else:
                missing.append(name)
        _atomic_write(target, m)
    return {"restored": restored, "missing": missing}


def train_restore(
    project_dir: Path,
    version_label: str,
    names: list[str],
) -> dict[str, list[str]]:
    """复原：从 `download/{entry.origin}` 复制覆盖回 `train/{name}`。

    跟老 `restore` 语义完全不同——老的是"删 manifest entry + 删 preprocess
    PNG"靠 resolver fallback；新模型下 train/ 是 self-contained 没 fallback，
    必须显式从 download 复制（详 ADR 0010 §Restore 语义）。

    返回三组：
    - `restored`：成功复原（download 原图存在并已复制覆盖）
    - `missing`：name 在 manifest 没 entry（也不复制；调用方决定怎么提示）
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
    with _LOCK:
        m = _read_train_target(target)
        for name in names:
            entry = m["images"].get(name)
            if entry is None:
                missing.append(name)
                continue
            origin = entry_origin(entry, name)
            src = download_dir / origin
            if not src.is_file():
                no_origin.append(name)
                continue
            dst = train_dir / name
            try:
                shutil.copy2(src, dst)
            except OSError:
                no_origin.append(name)
                continue
            try:
                st = src.stat()
                m["images"][name] = {
                    "origin": origin,
                    "mtime": int(st.st_mtime),
                    "size": st.st_size,
                }
            except OSError:
                # 极端：copy2 成功但 stat 失败 — 保守只更 origin
                m["images"][name] = {**entry, "origin": origin}
            restored.append(name)
        _atomic_write(target, m)
    return {"restored": restored, "missing": missing, "no_origin": no_origin}


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
