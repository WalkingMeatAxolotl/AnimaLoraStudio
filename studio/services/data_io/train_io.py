"""PP7 — 训练集（打标后的 train/）导出 / 导入。

云上 Studio 实例易丢；用户在 ④ 标签编辑那一步往往要花数小时手工调 caption，
真正怕丢的是「打标后的 train/」。本模块把它打成可下载的 zip，并支持导回新建项目。

zip 结构（schema_version 1，旧版 train.zip）：
    {slug}-{label}.train.zip
    ├── manifest.json    # source / stats
    └── train/
        └── {N}_data/    # 原样保留 N
            ├── *.png/...
            └── *.txt    # 可选

zip 结构（schema_version 2，bundle.zip）：
    {slug}-{label}.bundle.zip
    ├── manifest.json    # source / stats / includes
    ├── train/           # 可选：训练集
    │   └── {N}_data/
    │       ├── *.png/...
    │       └── *.txt    # 按 train_captions 决定是否打入
    ├── reg/             # 可选：正则集
    │   ├── meta.json    # 来自 reg/meta.json
    │   └── {folder}/
    │       ├── *.png/...
    │       └── *.txt    # 按 reg_captions 决定是否打入
    └── presets/         # 可选：预设
        └── {name}.yaml

导入语义：永远新建 project + source.version_label（缺失/非法时 v1，不合并到现有），
slug 冲突自动加 -imported-{ts}。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ...services.projects import projects, versions
from ..dataset.scan import IMAGE_EXTS
from ...paths import safe_join

SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 2
MANIFEST_NAME = "manifest.json"
TRAIN_PREFIX = "train/"
REG_PREFIX = "reg/"
PRESETS_PREFIX = "presets/"
CAPTION_EXTS = {".txt"}
# VAE latent 缓存（CachedLatentDataset 的 {名}.npz / {名}.r{reso}.npz，紧挨图片）
LATENT_CACHE_EXT = ".npz"
# Phase 2 文本缓存：随图 sidecar。prompt 聚合缓存归 task 档案
# （tasks/<id>/.text-cache/），不在 version 树里，bundle 不打包。
# 继续复用现有 train_latent_cache / reg_latent_cache bundle 选项：两者表达的
# 是“携带可重建训练缓存”，不改变默认导出体积。
TEXT_CACHE_SIDECAR_SUFFIX = ".text.safetensors"
# 训练 mask sidecar（train/{folder}/{stem}.mask，与图同目录，详
# services/preprocess/masks.py）。arcname 两段与图片同构。
MASK_SUFFIX = ".mask"


VERSION_CONFIG_ARC = "presets/config.yaml"


@dataclass
class BundleOptions:
    train: bool = True
    train_captions: bool = True
    reg: bool = False
    reg_captions: bool = False
    # True = 导出本 version 的私有 config.yaml（去掉路径字段后）作为可移植训练配置
    include_config: bool = False
    # True = 一并打包对应目录里的 VAE latent 缓存（*.npz），导入后免重新 encode
    train_latent_cache: bool = False
    reg_latent_cache: bool = False
    # True = 一并打包训练 mask sidecar（train/{folder}/{stem}.mask，masked loss 数据面）
    train_masks: bool = False


from studio.domain.errors import DomainError, NotFoundError


class TrainIOError(DomainError):
    """导出 / 导入过程的业务错误。

    PR-2 C3 加 DomainError base — handler 自动翻 dual-write envelope。
    """
    default_code = "train_io.error"


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def export_train(
    conn: sqlite3.Connection, version_id: int, dest: Path
) -> dict[str, Any]:
    """打包 version 的 train/ + manifest.json 到 dest（zip 路径）。

    dest 父目录必须已存在；用 ZIP_STORED（PNG/jpg 已是压缩态）。
    返回 {"manifest": {...}, "size_bytes": int}。
    """
    v = versions.get_version(conn, version_id)
    if not v:
        raise NotFoundError(
            "Version not found", code="version.not_found",
            details={"id": version_id},
        )
    p = projects.get_project(conn, v["project_id"])
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found",
            details={"id": v["project_id"]},
        )

    train_dir = versions.version_dir(p["id"], p["slug"], v["label"]) / "train"
    if not train_dir.exists():
        raise TrainIOError(
            "No images to export", code="dataset.export_empty", http_status=400,
        )

    concepts: list[dict[str, Any]] = []
    image_count = 0
    tagged_count = 0
    payload: list[tuple[Path, str]] = []  # (abs_path, arcname)

    for sub in sorted(train_dir.iterdir()):
        if not sub.is_dir():
            continue
        cnt = 0
        for f in sorted(sub.iterdir()):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                cnt += 1
                arc = f"{TRAIN_PREFIX}{sub.name}/{f.name}"
                payload.append((f, arc))
                if f.with_suffix(".txt").exists():
                    tagged_count += 1
            elif ext in CAPTION_EXTS:
                arc = f"{TRAIN_PREFIX}{sub.name}/{f.name}"
                payload.append((f, arc))
        if cnt > 0:
            concepts.append({"folder": sub.name, "image_count": cnt})
            image_count += cnt

    if image_count == 0:
        raise TrainIOError(
            "No images to export", code="dataset.export_empty", http_status=400,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": time.time(),
        "source": {
            "title": p["title"],
            "version_label": v["label"],
            "slug": p["slug"],
        },
        "stats": {
            "image_count": image_count,
            "tagged_count": tagged_count,
            "untagged_count": image_count - tagged_count,
            "concepts": concepts,
        },
    }

    with zipfile.ZipFile(
        dest, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
        for src, arc in payload:
            zf.write(src, arcname=arc)

    return {"manifest": manifest, "size_bytes": dest.stat().st_size}


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


def _safe_arc(name: str) -> Optional[str]:
    """zip slip 防护：拒绝绝对路径 / .. 段 / 非 train/ 前缀。

    返回去掉 train/ 前缀后的 「{folder}/{filename}」相对部分；
    不合法返回 None。空目录条目（以 / 结尾）也返回 None。
    """
    if not name or name.endswith("/"):
        return None
    norm = name.replace("\\", "/")
    if norm.startswith("/") or ".." in norm.split("/"):
        return None
    if not norm.startswith(TRAIN_PREFIX):
        return None
    inner = norm[len(TRAIN_PREFIX) :]
    parts = inner.split("/")
    # 至少 {folder}/{filename}，再深的层级（{folder}/sub/x.png）拒绝
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return inner


def _read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    try:
        with zf.open(MANIFEST_NAME) as fh:
            return json.loads(fh.read().decode("utf-8"))
    except KeyError as exc:
        raise TrainIOError(
            "Import file is invalid: missing manifest",
            code="dataset.import_invalid", http_status=400,
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TrainIOError(
            f"Import file is invalid: {exc}",
            code="dataset.import_invalid",
            details={"reason": str(exc)}, http_status=400,
        ) from exc


def _resolve_slug_conflict(
    conn: sqlite3.Connection, base: str
) -> str:
    """slug 冲突 → 加 -imported-{ts} 后缀；仍冲突再加 -{n}。"""
    if not conn.execute(
        "SELECT 1 FROM projects WHERE slug = ?", (base,)
    ).fetchone():
        return base
    candidate = f"{base}-imported-{int(time.time())}"
    n = 1
    final = candidate
    while conn.execute(
        "SELECT 1 FROM projects WHERE slug = ?", (final,)
    ).fetchone():
        n += 1
        final = f"{candidate}-{n}"
    return final


def import_train(
    conn: sqlite3.Connection, zip_path: Path
) -> dict[str, Any]:
    """从 zip 解出新建 project + v1，stage=tagging。

    返回 {"project": {...}, "version": {...}, "stats": {...}}。
    """
    if not zip_path.exists():
        raise NotFoundError(
            "File not found", code="file.not_found", http_status=404,
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = _read_manifest(zf)
            source = manifest.get("source") or {}
            title = (source.get("title") or "imported").strip() or "imported"
            base_slug = projects.slugify(source.get("slug") or title)

            # 先扫一遍合法 entry；空内容直接报错避免建空项目
            entries: list[tuple[zipfile.ZipInfo, str]] = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if info.filename == MANIFEST_NAME:
                    continue
                inner = _safe_arc(info.filename)
                if inner is None:
                    raise TrainIOError(
                        "Import file contains an invalid path",
                        code="dataset.import_invalid", http_status=400,
                    )
                entries.append((info, inner))

            if not entries:
                raise TrainIOError(
                    "Import file contains nothing to import",
                    code="dataset.import_empty", http_status=400,
                )

            # 建 project + v1（用 DAO，自动建目录树）
            slug = _resolve_slug_conflict(conn, base_slug)
            note = f"imported from {title!r}"
            p = projects.create_project(conn, title=title, slug=slug, note=note)
            v = versions.create_version(
                conn, project_id=p["id"], label="v1"
            )

            vdir = versions.version_dir(p["id"], p["slug"], v["label"])
            train_dir = vdir / "train"
            seen_folders: set[str] = set()

            for info, inner in entries:
                folder, filename = inner.split("/", 1)
                # 文件名再保险一层 + containment check
                if filename.startswith("."):
                    raise TrainIOError(
                        "Import file contains an invalid filename",
                        code="dataset.import_invalid", http_status=400,
                    )
                try:
                    target = safe_join(train_dir, folder, filename)
                except ValueError as exc:
                    raise TrainIOError(
                        "Import file contains an invalid filename",
                        code="dataset.import_invalid",
                        details={"reason": str(exc)}, http_status=400,
                    ) from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                seen_folders.add(folder)
                with zf.open(info) as src, target.open("wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

            # 写完后再统计 tagged_count —— zip 顺序不定，边写边数会漏（先 .png 后 .txt 时拿到 0）
            image_count = 0
            tagged_count = 0
            for folder in seen_folders:
                fdir = train_dir / folder
                if not fdir.exists():
                    continue
                for f in fdir.iterdir():
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        image_count += 1
                        if f.with_suffix(".txt").exists():
                            tagged_count += 1

            # ADR-0007 PR-5: 不再推 stage；phase cursor 由用户 PhaseHeaderNav 显式推
            v = versions.get_version(conn, v["id"])
            p = projects.get_project(conn, p["id"])
            assert v is not None and p is not None

    except zipfile.BadZipFile as exc:
        raise TrainIOError(
            "Import file is corrupt", code="dataset.import_corrupt",
            http_status=400,
        ) from exc

    stats = {
        "image_count": image_count,
        "tagged_count": tagged_count,
        "untagged_count": image_count - tagged_count,
        "concepts": sorted(seen_folders),
    }
    return {"project": p, "version": v, "stats": stats}


# ---------------------------------------------------------------------------
# bundle export / import (schema_version 2)
# ---------------------------------------------------------------------------


def _collect_train(
    train_dir: Path, include_captions: bool, include_latent_cache: bool = False,
    include_masks: bool = False,
) -> tuple[list[tuple[Path, str]], dict[str, Any]]:
    """扫 train/ 目录，返回 (payload, stats_dict)。

    include_masks 时收集与图同目录的 `{stem}.mask` sidecar。
    """
    payload: list[tuple[Path, str]] = []
    concepts: list[dict[str, Any]] = []
    image_count = 0
    tagged_count = 0
    latent_cache_count = 0
    text_cache_count = 0
    mask_count = 0

    if not train_dir.exists():
        return payload, {
            "image_count": 0, "tagged_count": 0, "concepts": [],
            "latent_cache_count": 0, "text_cache_count": 0, "mask_count": 0,
        }

    for sub in sorted(train_dir.iterdir()):
        if not sub.is_dir():
            continue
        cnt = 0
        for f in sorted(sub.iterdir()):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                cnt += 1
                payload.append((f, f"{TRAIN_PREFIX}{sub.name}/{f.name}"))
                if f.with_suffix(".txt").exists():
                    tagged_count += 1
                    if include_captions:
                        txt = f.with_suffix(".txt")
                        payload.append((txt, f"{TRAIN_PREFIX}{sub.name}/{txt.name}"))
            elif ext == LATENT_CACHE_EXT and include_latent_cache:
                latent_cache_count += 1
                payload.append((f, f"{TRAIN_PREFIX}{sub.name}/{f.name}"))
            elif include_latent_cache and f.name.endswith(
                TEXT_CACHE_SIDECAR_SUFFIX
            ):
                text_cache_count += 1
                payload.append((f, f"{TRAIN_PREFIX}{sub.name}/{f.name}"))
            elif ext == MASK_SUFFIX and include_masks:
                mask_count += 1
                payload.append((f, f"{TRAIN_PREFIX}{sub.name}/{f.name}"))
            elif ext in CAPTION_EXTS and include_captions:
                # .txt 先由图片那侧 include，这里跳过避免重复
                pass
        if cnt > 0:
            concepts.append({"folder": sub.name, "image_count": cnt})
            image_count += cnt

    return payload, {
        "image_count": image_count,
        "tagged_count": tagged_count,
        "concepts": concepts,
        "latent_cache_count": latent_cache_count,
        "text_cache_count": text_cache_count,
        "mask_count": mask_count,
    }


def _collect_reg(
    reg_dir: Path, include_captions: bool, include_latent_cache: bool = False
) -> tuple[list[tuple[Path, str]], dict[str, Any]]:
    """扫 reg/ 目录，返回 (payload, stats_dict)。"""
    payload: list[tuple[Path, str]] = []
    image_count = 0
    latent_cache_count = 0
    text_cache_count = 0

    if not reg_dir.exists():
        return payload, {
            "image_count": 0, "latent_cache_count": 0, "text_cache_count": 0,
        }

    # meta.json
    meta = reg_dir / "meta.json"
    if meta.exists():
        payload.append((meta, f"{REG_PREFIX}meta.json"))

    for item in sorted(reg_dir.iterdir()):
        if item.name == "meta.json":
            continue
        if item.is_dir():
            for f in sorted(item.iterdir()):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in IMAGE_EXTS:
                    image_count += 1
                    payload.append((f, f"{REG_PREFIX}{item.name}/{f.name}"))
                    if include_captions and f.with_suffix(".txt").exists():
                        txt = f.with_suffix(".txt")
                        payload.append((txt, f"{REG_PREFIX}{item.name}/{txt.name}"))
                elif ext == LATENT_CACHE_EXT and include_latent_cache:
                    latent_cache_count += 1
                    payload.append((f, f"{REG_PREFIX}{item.name}/{f.name}"))
                elif (
                    include_latent_cache
                    and f.name.endswith(TEXT_CACHE_SIDECAR_SUFFIX)
                ):
                    text_cache_count += 1
                    payload.append((f, f"{REG_PREFIX}{item.name}/{f.name}"))
                elif ext in CAPTION_EXTS and include_captions:
                    pass  # 由图片侧 include
        elif item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            # reg/ 根目录直接放的图（非标准但兼容）。旁边的 npz 不收：
            # 单层 reg/{file} 在 _safe_arc_bundle 里本就只放行 meta.json。
            image_count += 1
            payload.append((item, f"{REG_PREFIX}{item.name}"))

    return payload, {
        "image_count": image_count,
        "latent_cache_count": latent_cache_count,
        "text_cache_count": text_cache_count,
    }


def export_bundle(
    conn: sqlite3.Connection,
    version_id: int,
    dest: Path,
    opts: BundleOptions,
) -> dict[str, Any]:
    """按 BundleOptions 打包 bundle.zip 到 dest。

    至少需选中 train / reg / include_config 中的一项，否则 raise TrainIOError。
    返回 {"manifest": {...}, "size_bytes": int}。
    """
    if not opts.train and not opts.reg and not opts.include_config:
        raise TrainIOError(
            "Select at least one item to export (training set, regularization "
            "set, or training configuration)",
            code="dataset.export_nothing_selected", http_status=400,
        )

    v = versions.get_version(conn, version_id)
    if not v:
        raise NotFoundError(
            "Version not found", code="version.not_found",
            details={"id": version_id},
        )
    p = projects.get_project(conn, v["project_id"])
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found",
            details={"id": v["project_id"]},
        )

    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    payload: list[tuple[Path, str]] = []
    in_memory: list[tuple[str, str]] = []  # (content_str, arcname) for small generated files

    # --- train section ---
    train_stats: dict[str, Any] = {}
    if opts.train:
        tp, train_stats = _collect_train(
            vdir / "train", opts.train_captions, opts.train_latent_cache,
            include_masks=opts.train_masks,
        )
        if not tp:
            raise TrainIOError(
                "No images to export", code="dataset.export_empty",
                http_status=400,
            )
        payload.extend(tp)

    # --- reg section ---
    reg_stats: dict[str, Any] = {}
    if opts.reg:
        rp, reg_stats = _collect_reg(
            vdir / "reg", opts.reg_captions, opts.reg_latent_cache
        )
        payload.extend(rp)

    # --- version 私有训练配置 ---
    # 导出 version 自己的 config.yaml，剔除 PROJECT_SPECIFIC_FIELDS（路径字段），
    # 保留超参数等可移植内容。导入时 write_version_config(force_project_overrides=True)
    # 会自动填好目标环境的正确路径。
    config_included = False
    if opts.include_config:
        from .. import version_config as _vc
        from ...domain.config_prune import prune_inactive_fields
        import yaml as _yaml
        try:
            # read_version_config 经 pydantic 补回了全部默认字段，bundle 里的
            # config 和落盘 yaml 一样只带生效字段，这里再裁一遍。
            cfg = _vc.read_version_config(p, v)
            portable = prune_inactive_fields(
                {k: v_ for k, v_ in cfg.items() if k not in _vc.PROJECT_SPECIFIC_FIELDS}
            )
            in_memory.append((
                _yaml.safe_dump(portable, allow_unicode=True, sort_keys=False, default_flow_style=False),
                VERSION_CONFIG_ARC,
            ))
            config_included = True
        except _vc.VersionConfigError:
            # version 还没配置训练参数，跳过（不报错，只是不打入 bundle）
            pass

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "exported_at": time.time(),
        "source": {
            "title": p["title"],
            "version_label": v["label"],
            "slug": p["slug"],
            "preset_name": v.get("config_name"),
        },
        "includes": {
            "train": opts.train,
            "train_captions": opts.train_captions,
            "reg": opts.reg,
            "reg_captions": opts.reg_captions,
            "config": config_included,
            "train_latent_cache": opts.train_latent_cache,
            "reg_latent_cache": opts.reg_latent_cache,
            "train_masks": opts.train_masks,
        },
        "stats": {
            "train_image_count": train_stats.get("image_count", 0),
            "train_tagged_count": train_stats.get("tagged_count", 0),
            "reg_image_count": reg_stats.get("image_count", 0),
            "config_included": config_included,
            "latent_cache_count": (
                train_stats.get("latent_cache_count", 0)
                + reg_stats.get("latent_cache_count", 0)
            ),
            "text_cache_count": (
                train_stats.get("text_cache_count", 0)
                + reg_stats.get("text_cache_count", 0)
            ),
            "train_mask_count": train_stats.get("mask_count", 0),
        },
    }

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
        for content, arc in in_memory:
            zf.writestr(arc, content)
        for src, arc in payload:
            zf.write(src, arcname=arc)

    return {"manifest": manifest, "size_bytes": dest.stat().st_size}


def _safe_arc_bundle(name: str) -> Optional[tuple[str, str]]:
    """bundle zip slip 防护。

    返回 (section, inner)；section: "train" | "reg" | "preset"。
    不合法返回 None。
    """
    if not name or name.endswith("/"):
        return None
    norm = name.replace("\\", "/")
    if norm.startswith("/") or ".." in norm.split("/"):
        return None

    if norm.startswith(TRAIN_PREFIX):
        inner = norm[len(TRAIN_PREFIX):]
        parts = inner.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        return ("train", inner)

    if norm.startswith(REG_PREFIX):
        inner = norm[len(REG_PREFIX):]
        # reg/meta.json — 单层
        if inner == "meta.json":
            return ("reg", inner)
        parts = inner.split("/")
        # reg/{folder}/{file} — 两层
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        return ("reg", inner)

    if norm.startswith(PRESETS_PREFIX):
        inner = norm[len(PRESETS_PREFIX):]
        parts = inner.split("/")
        if len(parts) != 1 or not inner or not inner.endswith(".yaml"):
            return None
        return ("preset", inner)

    return None


def _bundle_source_version_label(source: dict[str, Any]) -> str:
    """manifest 是不可信输入 — 校验走 versions 同一套规则（含拒绝 "." / ".."），
    不合法回退 v1 而不是报错，保证老 / 手改 bundle 仍可导入。"""
    raw = str(source.get("version_label") or "v1").strip()
    return raw if versions.is_valid_label(raw) else "v1"


def _bundle_source_preset_name(source: dict[str, Any]) -> Optional[str]:
    raw = source.get("preset_name") or source.get("config_name")
    if raw is None:
        return None
    name = str(raw).strip()
    return name or None


def _restore_preset_name(
    conn: sqlite3.Connection,
    version_id: int,
    preset_name: Optional[str],
    presets_base: Path,
) -> Optional[dict[str, Any]]:
    """preset 真实存在（本机原有，或刚从 bundle 解出）才回填 config_name，
    避免 version 指向不存在的预设；preset_path 自带名字校验，非法名
    （路径分隔符等）按不存在处理。返回更新后的 version；没动返回 None。"""
    if not preset_name:
        return None
    from ..presets import io as _presets_io
    try:
        if not _presets_io.preset_path(preset_name, presets_base).exists():
            return None
    except _presets_io.PresetError:
        return None
    return versions.update_version(conn, version_id, config_name=preset_name)


def import_bundle(
    conn: sqlite3.Connection,
    zip_path: Path,
    presets_base: Path,
) -> dict[str, Any]:
    """从 bundle.zip（schema_version 1 或 2）导入，新建 project + source version。

    v1（旧 train.zip）：等同于 import_train。
    v2：按 manifest.includes 分别处理 train / reg / presets。
    返回 {"project": {...}, "version": {...}, "stats": {...}}。
    """
    if not zip_path.exists():
        raise NotFoundError(
            "File not found", code="file.not_found", http_status=404,
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = _read_manifest(zf)
            schema_ver = manifest.get("schema_version", 1)

            if schema_ver == 1:
                # 旧格式：委托给原有逻辑
                pass  # fallthrough to import_train below

            source = manifest.get("source") or {}
            title = (source.get("title") or "imported").strip() or "imported"
            base_slug = projects.slugify(source.get("slug") or title)
            version_label = _bundle_source_version_label(source)
            preset_name = _bundle_source_preset_name(source)

            if schema_ver == 1:
                # v1：仅 train/ entries，用原有安全检查
                entries_v1: list[tuple[zipfile.ZipInfo, str]] = []
                for info in zf.infolist():
                    if info.is_dir() or info.filename == MANIFEST_NAME:
                        continue
                    inner = _safe_arc(info.filename)
                    if inner is None:
                        raise TrainIOError(
                            "Import file contains an invalid path",
                            code="dataset.import_invalid", http_status=400,
                        )
                    entries_v1.append((info, inner))
                if not entries_v1:
                    raise TrainIOError(
                        "Import file contains nothing to import",
                        code="dataset.import_empty", http_status=400,
                    )

                slug = _resolve_slug_conflict(conn, base_slug)
                p = projects.create_project(conn, title=title, slug=slug,
                                             note=f"imported from {title!r}")
                v = versions.create_version(conn, project_id=p["id"], label=version_label)
                v = _restore_preset_name(conn, v["id"], preset_name, presets_base) or v
                vdir = versions.version_dir(p["id"], p["slug"], v["label"])
                train_dir = vdir / "train"
                seen_folders: set[str] = set()
                for info, inner in entries_v1:
                    folder, filename = inner.split("/", 1)
                    if filename.startswith("."):
                        raise TrainIOError(
                            "Import file contains an invalid filename",
                            code="dataset.import_invalid", http_status=400,
                        )
                    target = safe_join(train_dir, folder, filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    seen_folders.add(folder)
                    with zf.open(info) as src, target.open("wb") as dst:
                        _copy_chunks(src, dst)

                img_cnt, tag_cnt = _count_train(train_dir, seen_folders)
                # ADR-0007 PR-5: 不再推 stage
                v = versions.get_version(conn, v["id"])
                p = projects.get_project(conn, p["id"])
                assert v is not None and p is not None
                return {
                    "project": p, "version": v,
                    "stats": {
                        "train_image_count": img_cnt,
                        "train_tagged_count": tag_cnt,
                        "reg_image_count": 0,
                        "config_imported": False,
                        "preset_count": 0,
                    },
                }

            # v2 bundle
            includes = manifest.get("includes") or {}
            train_entries: list[tuple[zipfile.ZipInfo, str]] = []
            reg_entries: list[tuple[zipfile.ZipInfo, str]] = []
            preset_entries: list[tuple[zipfile.ZipInfo, str]] = []

            for info in zf.infolist():
                if info.is_dir() or info.filename == MANIFEST_NAME:
                    continue
                result = _safe_arc_bundle(info.filename)
                if result is None:
                    raise TrainIOError(
                        "Import file contains an invalid path",
                        code="dataset.import_invalid", http_status=400,
                    )
                section, inner = result
                if section == "train":
                    train_entries.append((info, inner))
                elif section == "reg":
                    reg_entries.append((info, inner))
                elif section == "preset":
                    preset_entries.append((info, inner))

            if not train_entries and not reg_entries and not preset_entries:
                raise TrainIOError(
                    "Import file contains nothing to import",
                    code="dataset.import_empty", http_status=400,
                )

            slug = _resolve_slug_conflict(conn, base_slug)
            p = projects.create_project(conn, title=title, slug=slug,
                                         note=f"imported from {title!r}")
            v = versions.create_version(conn, project_id=p["id"], label=version_label)
            vdir = versions.version_dir(p["id"], p["slug"], v["label"])

            # --- 写入 train ---
            seen_train: set[str] = set()
            npz_targets: list[Path] = []
            if train_entries:
                train_dir = vdir / "train"
                for info, inner in train_entries:
                    folder, filename = inner.split("/", 1)
                    if filename.startswith("."):
                        raise TrainIOError(
                            "Import file contains an invalid filename",
                            code="dataset.import_invalid", http_status=400,
                        )
                    target = safe_join(train_dir, folder, filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    seen_train.add(folder)
                    with zf.open(info) as src, target.open("wb") as dst:
                        _copy_chunks(src, dst)
                    if target.suffix.lower() == LATENT_CACHE_EXT:
                        npz_targets.append(target)

            # --- 写入 reg ---
            reg_image_count = 0
            if reg_entries:
                reg_dir = vdir / "reg"
                for info, inner in reg_entries:
                    if inner == "meta.json":
                        target = reg_dir / "meta.json"
                        target.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        parts = inner.split("/", 1)
                        if len(parts) != 2:
                            raise TrainIOError(
                                "Import file contains an invalid path",
                                code="dataset.import_invalid", http_status=400,
                            )
                        folder, filename = parts
                        if filename.startswith("."):
                            raise TrainIOError(
                                "Import file contains an invalid filename",
                                code="dataset.import_invalid", http_status=400,
                            )
                        target = safe_join(reg_dir, folder, filename)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.suffix.lower() in IMAGE_EXTS:
                            reg_image_count += 1
                        elif target.suffix.lower() == LATENT_CACHE_EXT:
                            npz_targets.append(target)
                    with zf.open(info) as src, target.open("wb") as dst:
                        _copy_chunks(src, dst)

            # latent 缓存 mtime 修正：训练侧 _is_cache_valid 要求 npz mtime ≥
            # 图片 mtime，而逐条解包按字典序 npz 常先于同名图片落盘（.npz < .png）
            # → 缓存会被判失效重 encode。解包完统一 touch 到当前时间。
            for t in npz_targets:
                os.utime(t, None)

            # --- 写入 version 训练配置 / 其他预设 ---
            # presets/config.yaml → 应用到新 version（force_project_overrides 自动填路径）
            # 其他 .yaml → 写入全局预设目录（冲突加 _imported_{n}）
            config_imported = False
            preset_count = 0
            if preset_entries:
                from .. import version_config as _vc
                import yaml as _yaml
                for info, inner in preset_entries:
                    if inner.startswith("."):
                        continue
                    if inner == "config.yaml":
                        try:
                            raw = json.loads(zf.read(info))
                        except (json.JSONDecodeError, ValueError):
                            try:
                                raw = _yaml.safe_load(zf.read(info)) or {}
                            except Exception:
                                raw = {}
                        if isinstance(raw, dict):
                            # 4 全局模型路径字段不在 PROJECT_SPECIFIC_FIELDS 里，
                            # bundle 里带的源机器绝对路径（如 `G:/models/...`）会
                            # 原样落盘，跨机器导入时被 `_absolutize_model_paths`
                            # 拼成 `<repo>/G:/...`。对齐 fork_preset_for_version：
                            # auto_sync_paths=ON 时用本机全局值覆盖 4 字段；OFF
                            # 时尊重 bundle 内容（盘符识别已在
                            # _absolutize_model_paths 修过，POSIX 下不再误拼前缀）。
                            from .. import presets as _presets_svc
                            from .. import models as _md
                            if _presets_svc._auto_sync_paths():
                                family = str(raw.get("model_family") or "anima")
                                try:
                                    raw.update(_md.default_paths_for_new_version(
                                        family=family))
                                except ValueError:
                                    # 未知族的 bundle：不覆路径，交给下面
                                    # write_version_config 的 schema 校验拒绝
                                    pass
                            try:
                                _vc.write_version_config(p, v, raw, force_project_overrides=True)
                                config_imported = True
                            except _vc.VersionConfigError:
                                pass  # 无效 config，跳过，不中断导入
                    else:
                        from ..presets import io as _presets_io

                        try:
                            config, _ = _presets_io.parse_preset_bytes(zf.read(info), inner)
                        except _presets_io.PresetError:
                            continue
                        target_name = Path(inner).stem
                        target = _presets_io.preset_path(target_name, presets_base)
                        if target.exists():
                            n = 1
                            while True:
                                candidate_name = f"{target_name}_imported_{n}"
                                candidate = _presets_io.preset_path(candidate_name, presets_base)
                                if not candidate.exists():
                                    target_name = candidate_name
                                    break
                                n += 1
                        _presets_io.write_preset(target_name, config, presets_base)
                        preset_count += 1

            # preset 解包完成后再回填 config_name：本机真的有这个预设
            # （原本就有，或刚从 bundle 解出）才记，避免悬空引用。
            v = _restore_preset_name(conn, v["id"], preset_name, presets_base) or v

            img_cnt, tag_cnt = _count_train(vdir / "train", seen_train) if train_entries else (0, 0)

            # ADR-0007 PR-5: 不再推 stage
            v = versions.get_version(conn, v["id"])
            p = projects.get_project(conn, p["id"])
            assert v is not None and p is not None

    except zipfile.BadZipFile as exc:
        raise TrainIOError(
            "Import file is corrupt", code="dataset.import_corrupt",
            http_status=400,
        ) from exc

    return {
        "project": p,
        "version": v,
        "stats": {
            "train_image_count": img_cnt,
            "train_tagged_count": tag_cnt,
            "reg_image_count": reg_image_count,
            "config_imported": config_imported,
            "preset_count": preset_count,
        },
    }


def _copy_chunks(src: Any, dst: Any) -> None:
    while True:
        chunk = src.read(64 * 1024)
        if not chunk:
            break
        dst.write(chunk)


def _count_train(train_dir: Path, folders: set[str]) -> tuple[int, int]:
    img = 0
    tagged = 0
    for folder in folders:
        fdir = train_dir / folder
        if not fdir.exists():
            continue
        for f in fdir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                img += 1
                if f.with_suffix(".txt").exists():
                    tagged += 1
    return img, tagged
