"""预处理状态 manifest（单 JSON 文件，project 级）。

设计见 [ADR 0004](../../docs/adr/0004-preprocess-manifest.md)。

简而言之
--------
`projects/{id}-{slug}/preprocess/manifest.json` 记录**非默认**的预处理决定：

    {
      "images": {
        "bar.png": {
          "kind": "processed",
          "source": "bar.jpg",
          "model": "RealESRGAN_x4",
          "scale": 4,
          "action": "upscale",
          "target_area": 1048576,
          "src_size": [512, 512],
          "dst_size": [2048, 2048],
          "elapsed_seconds": 12.3,
          "mtime": 1731000000
        }
      }
    }

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


def resolve(project_dir: Path, name: str) -> Optional[Path]:
    """给定产物文件名（如 `foo.png`），返回它实际指向的磁盘路径。

    隐式 original   → `download/{name}`（即使该图不存在；resolver 不做存在性检查）
    kind=processed → `preprocess/{name}`
    其他 kind      → None（兼容未来的 deleted 等状态）

    存在性由调用方按需 `.exists()` 检查——这样列图时一次 stat 即可，不重复。
    """
    m = load(project_dir)
    entry = m["images"].get(name)
    if entry is None:
        return project_dir / "download" / name
    kind = entry.get("kind")
    if kind == "processed":
        return project_dir / "preprocess" / name
    # 未知 / 未来扩展 → 当作"不可解析"，下游 skip
    return None


def get_entry(project_dir: Path, name: str) -> Optional[dict[str, Any]]:
    """读单条 entry（不存在返 None）。给 list_processed 拼元数据用。"""
    m = load(project_dir)
    return m["images"].get(name)


def all_processed(project_dir: Path) -> dict[str, dict[str, Any]]:
    """返回 `{name: entry}` 所有 kind=processed 的 entry。"""
    m = load(project_dir)
    return {
        name: entry
        for name, entry in m["images"].items()
        if entry.get("kind") == "processed"
    }


# ---------------------------------------------------------------------------
# Mutation — 必须 with _LOCK
# ---------------------------------------------------------------------------


def add_processed(project_dir: Path, name: str, meta: dict[str, Any]) -> None:
    """记录一张已处理图。meta 期望包含 source/model/scale/action/.../src_size/dst_size。

    worker 跑完每张图后调一次。`mtime` 字段自动补 time.time() 如果 meta 没带。
    """
    with _LOCK:
        m = load(project_dir)
        entry = {"kind": "processed", **meta}
        entry.setdefault("mtime", time.time())
        m["images"][name] = entry
        _atomic_write(manifest_path(project_dir), m)


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
