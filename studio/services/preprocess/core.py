"""预处理业务层：列表 / 状态 / 启动 job / 还原。

第一阶段只做"放大"，但目录契约和接口预留好裁剪 / 涂抹的位置。

数据模型（ADR 0004）
-------------------
`projects/{id}-{slug}/preprocess/manifest.json` 是状态唯一真理：

    {"images": {"bar.png": {"kind": "processed", "model": "...", "scale": 4, ...}}}

- manifest 没记 → 默认 = 用 download/ 原图
- `kind: processed` → preprocess/{name}.png 是改过的副本

下游（curation / thumbnail / copy_to_train）通过
`studio.services.preprocess_manifest.resolve()` 拿实际文件路径，本模块只负责
**列图状态 + 启动 job + 还原**。

产物文件名规则：固定 `{src_stem}.png`。同 stem 但不同扩展名的源图碰撞时
（如 `cat.jpg` 和 `cat.png` 同存）— 后处理的覆盖前者，并在日志里 warn。

Job 调度
--------
preprocess 是 GPU-bound job kind，走 DATA 槽位：
- 训练正在跑 + 未开 `allow_gpu_during_train` → 推迟
- daemon 占着 VRAM → 触发让位（_maybe_yield_daemon），等下次 tick

不复用 download_worker 的并发设计 —— 串行处理就行，模型加载到 GPU 后
单张耗时 1-3s（4x，512px 输入，cuda）。批量并发的收益不抵 VRAM 风险。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from ...services.projects import jobs as project_jobs, projects
from ...services.dataset.scan import IMAGE_EXTS
from . import manifest as preprocess_manifest


PREPROCESS_KIND = "preprocess"
# 同一个 kind 下用 params['stage'] 分发到不同 worker 分支。默认 'upscale' 兼容
# 历史 job（params 缺 stage 时按放大处理）。
STAGE_UPSCALE = "upscale"
STAGE_CROP = "crop"
DEFAULT_MODEL = "4x-AnimeSharp"
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_PAD = 16
DEFAULT_DEVICE = "auto"
# LoRA 训练桶的目标面积。1024² = 1048576 px 是 SDXL/Flux/Anima 常用桶；用户
# 可以在 UI 选 768²/1024²/1536²/2048² 或自定义边长。
DEFAULT_TARGET_AREA = 1024 * 1024

PRODUCT_SUFFIX = ".png"
# 裁剪框最小归一化边长，画布上小于这个的不算有效（避免误触出零像素图）
MIN_CROP_NORM = 0.02


from studio.domain.errors import DomainError


class PreprocessError(DomainError):
    """预处理业务错误（项目不存在 / 参数非法 / 文件名非法）。

    PR-2 C3 加 DomainError base — handler 自动翻 dual-write envelope。
    """
    default_code = "preprocess.error"


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------


def project_paths(p: dict[str, Any]) -> tuple[Path, Path]:
    """返回 `(download_dir, preprocess_dir)`，不保证存在。"""
    pdir = projects.project_dir(p["id"], p["slug"])
    return pdir / "download", pdir / "preprocess"


def project_root(p: dict[str, Any]) -> Path:
    """项目根目录（manifest 路径基于此）。"""
    return projects.project_dir(p["id"], p["slug"])


def product_path_for(preprocess_dir: Path, source_name: str) -> Path:
    """`download/foo.webp` → `preprocess/foo.png`。"""
    stem = Path(source_name).stem
    return preprocess_dir / f"{stem}{PRODUCT_SUFFIX}"


# ---------------------------------------------------------------------------
# 列表 / 状态（基于 manifest）
# ---------------------------------------------------------------------------


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _download_images(download: Path) -> list[Path]:
    if not download.exists():
        return []
    return sorted([f for f in download.iterdir() if _is_image(f)])


# ---------------------------------------------------------------------------
# 目标选择 + 启动
# ---------------------------------------------------------------------------


_SAFE_NAME_FORBIDDEN = ("/", "\\", "..")


def _validate_name(name: str) -> None:
    if not name or any(t in name for t in _SAFE_NAME_FORBIDDEN):
        raise PreprocessError(f"非法文件名: {name!r}")


def _validate_rel_name(name: str) -> None:
    """ADR 0010 train-scope name 校验：必须形如 `"folder/image"`（POSIX 形式）。

    严格 2 段 + 拒 `..` / 反斜杠 / 绝对路径 / 空段，防 path traversal。
    """
    if not name:
        raise PreprocessError(f"非法 rel name: {name!r}")
    if "\\" in name or name.startswith("/"):
        raise PreprocessError(f"非法 rel name: {name!r}")
    parts = name.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1] or ".." in parts:
        raise PreprocessError(f"非法 rel name（要求 folder/image）: {name!r}")


def _validate_rect(rect: dict[str, Any]) -> dict[str, float]:
    """归一化 + clamp 一条裁剪 rect。非法 → 抛 PreprocessError。"""
    try:
        x = float(rect["x"])
        y = float(rect["y"])
        w = float(rect["w"])
        h = float(rect["h"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreprocessError(f"裁剪 rect 缺字段或类型错: {rect!r}") from exc
    # clamp 到 [0,1]，但保留 w/h 下限校验
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w < MIN_CROP_NORM or h < MIN_CROP_NORM:
        raise PreprocessError(
            f"裁剪框过小（最小 {MIN_CROP_NORM}）: {rect!r}"
        )
    return {"x": x, "y": y, "w": w, "h": h}


# ---------------------------------------------------------------------------
# ADR 0010 — train-scope 列表 / 状态 / job
#
# 新代码用 *_train 系列：
#
# - list_train_images:   列 train/ 全部图 + manifest 元数据
# - summary_train:       train scope 简短统计
# - resolve_targets_train / start_job_train / start_crop_job_train: job 创建
# - list_crop_workspace_train / list_duplicate_removed_workspace_train: 子页工作集
# - restore_products_train: 调 manifest.train_restore（语义：copy download → train）
# ---------------------------------------------------------------------------


def version_train_dir(p: dict[str, Any], version_label: str) -> Path:
    return project_root(p) / "versions" / version_label / "train"


def _train_images_listing(train_dir: Path) -> list[tuple[str, Path]]:
    """递归 train_dir 一级 sub-folder（LoRA repeat folder）收集 `(rel_path, full_path)`。

    rel_path = POSIX 形式 `"{folder}/{image}"`，跟 manifest entry key 一致。
    train_dir 根目录直接放的图忽略（LoRA 训练只读 sub-folder 内）。
    按 rel_path 字典序稳定输出。
    """
    if not train_dir.exists():
        return []
    out: list[tuple[str, Path]] = []
    for sub in train_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if not _is_image(f):
                continue
            out.append((f"{sub.name}/{f.name}", f))
    out.sort(key=lambda t: t[0])
    return out


def list_train_images(
    p: dict[str, Any], version_label: str
) -> list[dict[str, Any]]:
    """列 `versions/{vlabel}/train/` 全部图 + manifest entry 元数据。

    替代老 `list_pending + list_processed` 二元概念——新模型下 train/ 即"训练集
    grid"，无 pending/processed 区分（状态从字段差异隐含推断，详 ADR 0010
    §Manifest schema v2）。

    返回 `[{name, mtime, size, w, h, origin, source, orphan, duplicate_removed,
    model, scale, action, target_area, src_size, dst_size, elapsed_seconds}]`：
    - `origin / source`：都填 `entry.origin`（兼容老前端字段名 source）
    - `orphan`：`download/{origin}` 缺失（restore 会落 no_origin）
    - `duplicate_removed`：bool（默认 False；UI 区分"训练参与" vs "审核跳过"）
    - 老 schema 透传字段（model/scale/...）新 entry 一律 None；前端容忍

    极端情况：manifest 标 duplicate_removed 但 train/ 物理已删（用户外部删）→
    仍报告一条 stale 项，UI 容忍 w/h 为 None。
    """
    from PIL import Image

    pdir = project_root(p)
    download_dir = pdir / "download"
    train_dir = version_train_dir(p, version_label)

    m = preprocess_manifest.train_load(pdir, version_label)
    entries = m["images"]

    download_names = (
        {f.name for f in _download_images(download_dir)}
        if download_dir.exists() else set()
    )

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rel, f in _train_images_listing(train_dir):
        seen.add(rel)
        entry = entries.get(rel, {})
        origin = preprocess_manifest.entry_origin(entry, rel)
        st = f.stat()
        w: Optional[int] = None
        h: Optional[int] = None
        try:
            with Image.open(f) as im:
                w, h = im.size
        except (OSError, ValueError):
            pass
        is_dup = preprocess_manifest.is_duplicate_removed_entry(entry)
        items.append({
            "name": rel,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "w": w, "h": h,
            "origin": origin,
            "source": origin,
            "orphan": origin not in download_names,
            "duplicate_removed": is_dup,
            # ADR 0010 fixup（2026-06-04）：直接读 manifest entry.processed 字段
            "processed": not is_dup and _is_processed(entry),
            "model": entry.get("model"),
            "scale": entry.get("scale"),
            "action": entry.get("action"),
            "target_area": entry.get("target_area"),
            "src_size": entry.get("src_size"),
            "dst_size": entry.get("dst_size"),
            "elapsed_seconds": entry.get("elapsed_seconds"),
        })

    # stale duplicate_removed entry（manifest 有 + train/ 物理无）
    for name, entry in sorted(entries.items()):
        if name in seen:
            continue
        if not preprocess_manifest.is_duplicate_removed_entry(entry):
            continue
        origin = preprocess_manifest.entry_origin(entry, name)
        items.append({
            "name": name,
            "mtime": float(entry.get("mtime", 0.0) or 0.0),
            "size": int(entry.get("size", 0) or 0),
            "w": None, "h": None,
            "origin": origin,
            "source": origin,
            "orphan": origin not in download_names,
            "duplicate_removed": True,
            "processed": False,
            "model": None, "scale": None, "action": None,
            "target_area": None, "src_size": None, "dst_size": None,
            "elapsed_seconds": None,
        })

    return items


def summary_train(p: dict[str, Any], version_label: str) -> dict[str, Any]:
    """train scope 简短统计。

    `image_count` = train/ 里物理图像数 + 仅 manifest 标记 duplicate_removed
    且物理已删的数（罕见 stale entry，仍计入展示）。
    """
    pdir = project_root(p)
    train_dir = version_train_dir(p, version_label)
    m = preprocess_manifest.train_load(pdir, version_label)
    physical = {rel for rel, _ in _train_images_listing(train_dir)}
    soft_removed_only = {
        name for name, entry in m["images"].items()
        if preprocess_manifest.is_duplicate_removed_entry(entry)
        and name not in physical
    }
    return {"image_count": len(physical) + len(soft_removed_only)}


def resolve_targets_train(
    p: dict[str, Any], version_label: str, *,
    mode: str, names: Optional[Iterable[str]] = None,
) -> list[str]:
    """根据 mode + names 返回当前 train/ grid 中要处理的图名列表。

    mode='all' / 'all_force' → train/ 全部图
    mode='selected'          → 名单与 train/ 实存交集
    """
    train_dir = version_train_dir(p, version_label)
    if not train_dir.exists() and mode != "selected":
        return []
    existing = {rel for rel, _ in _train_images_listing(train_dir)}

    if mode in ("all", "all_force"):
        return sorted(existing)
    if mode == "selected":
        if not names:
            raise PreprocessError("mode=selected 时 names 不能为空")
        chosen: list[str] = []
        for n in names:
            _validate_rel_name(n)
            if n in existing:
                chosen.append(n)
        return sorted(set(chosen))
    raise PreprocessError(f"未知 mode: {mode!r}")


def start_job_train(
    conn, *,
    project_id: int,
    version_id: int,
    mode: str = "all",
    names: Optional[list[str]] = None,
    model: str = DEFAULT_MODEL,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_pad: int = DEFAULT_TILE_PAD,
    device: str = DEFAULT_DEVICE,
    target_area: Optional[int] = DEFAULT_TARGET_AREA,
) -> dict[str, Any]:
    """train scope preprocess job。worker 通过 job.version_id 拿 version label
    后从 `versions/{label}/train/` 列源 + 写产物（PR-2 step D 改 worker）。
    """
    p = projects.get_project(conn, project_id)
    if not p:
        raise PreprocessError(f"项目不存在: id={project_id}")
    if mode not in ("all", "selected", "all_force"):
        raise PreprocessError(f"未知 mode: {mode!r}")
    if mode == "selected" and not names:
        raise PreprocessError("mode=selected 必须给 names")
    if names:
        for n in names:
            _validate_rel_name(n)

    params: dict[str, Any] = {
        "stage": STAGE_UPSCALE,
        "mode": mode,
        "model": model,
        "tile_size": int(tile_size),
        "tile_pad": int(tile_pad),
        "device": device,
        "target_area": int(target_area) if target_area else None,
    }
    if names:
        params["names"] = list(names)

    return project_jobs.create_job(
        conn,
        project_id=project_id,
        version_id=version_id,
        kind=PREPROCESS_KIND,
        params=params,
    )


def start_crop_job_train(
    conn, *,
    project_id: int,
    version_id: int,
    crops: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """train scope crop job。`crops` 的源文件名为 `train/` 下当前文件名。"""
    p = projects.get_project(conn, project_id)
    if not p:
        raise PreprocessError(f"项目不存在: id={project_id}")
    if not isinstance(crops, dict) or not crops:
        raise PreprocessError("crops 不能为空")
    sanitized: dict[str, list[dict[str, Any]]] = {}
    for name, rects in crops.items():
        _validate_rel_name(name)
        if not isinstance(rects, list) or not rects:
            raise PreprocessError(f"{name!r} 的 rects 为空")
        out_rects: list[dict[str, Any]] = []
        for r in rects:
            if not isinstance(r, dict):
                raise PreprocessError(f"{name!r} 含非法 rect: {r!r}")
            clean = _validate_rect(r)
            label = r.get("label")
            if label is not None:
                clean["label"] = str(label)[:64]
            out_rects.append(clean)
        sanitized[name] = out_rects

    params = {"stage": STAGE_CROP, "crops": sanitized}
    return project_jobs.create_job(
        conn,
        project_id=project_id,
        version_id=version_id,
        kind=PREPROCESS_KIND,
        params=params,
    )


def _is_processed(entry: dict[str, Any]) -> bool:
    """ADR 0010 状态推断（2026-06-04 fixup）：直接读 manifest entry 的
    `processed` 字段。

    worker upscale/crop 完成后写 `processed: True`；curate 复制原图不写
    （默认 False）。老 entry（没 `processed` 字段）一律视为未处理——
    用户重新跑 preprocess 即升级到新字段。
    """
    return bool(entry.get("processed", False))


def list_crop_workspace_train(
    p: dict[str, Any], version_label: str
) -> list[dict[str, Any]]:
    """裁剪页工作集（train scope）：train/ 里所有图，附像素尺寸 + processed 标记。

    详 `_is_processed` 的判定逻辑。duplicate_removed 的图跳过（不让用户对软
    删除图再裁）。
    """
    from PIL import Image

    pdir = project_root(p)
    train_dir = version_train_dir(p, version_label)
    download_dir = pdir / "download"
    m = preprocess_manifest.train_load(pdir, version_label)
    entries = m["images"]
    removed_origins = preprocess_manifest.train_duplicate_removed_origins(
        pdir, version_label
    )

    items: list[dict[str, Any]] = []
    for rel, f in _train_images_listing(train_dir):
        entry = entries.get(rel, {})
        if preprocess_manifest.is_duplicate_removed_entry(entry):
            continue
        origin = preprocess_manifest.entry_origin(entry, rel)
        if origin in removed_origins:
            continue
        try:
            with Image.open(f) as im:
                w, h = im.size
        except (OSError, ValueError):
            continue
        st = f.stat()
        items.append({
            "name": rel,
            "source": origin,
            "w": w, "h": h,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "processed": _is_processed(entry),
        })
    return items


def list_duplicate_removed_workspace_train(
    p: dict[str, Any], version_label: str
) -> list[dict[str, Any]]:
    """train scope 软删除工作集。物理文件仍在 train/{name}（mark_duplicate_removed
    不删文件），缩略图按 train bucket 取（thumb endpoint PR-3 改）。
    """
    from PIL import Image

    pdir = project_root(p)
    train_dir = version_train_dir(p, version_label)
    removed = preprocess_manifest.train_duplicate_removed(pdir, version_label)

    items: list[dict[str, Any]] = []
    for name in sorted(removed.keys()):
        entry = removed[name]
        origin = preprocess_manifest.entry_origin(entry, name)
        src = train_dir / name
        if not src.is_file():
            items.append({
                "name": name,
                "source": origin,
                "w": None, "h": None,
                "mtime": float(entry.get("mtime", 0.0) or 0.0),
                "size": int(entry.get("size", 0) or 0),
            })
            continue
        try:
            with Image.open(src) as im:
                w, h = im.size
        except (OSError, ValueError):
            w, h = None, None
        st = src.stat()
        items.append({
            "name": name,
            "source": origin,
            "w": w, "h": h,
            "mtime": st.st_mtime,
            "size": st.st_size,
        })
    return items


def restore_products_train(
    p: dict[str, Any], version_label: str, names: Iterable[str],
) -> dict[str, list[str]]:
    """train scope 还原：从 `download/{entry.origin}` 复制覆盖 `train/{name}`。

    返回 `{restored, missing, no_origin}` 三组。详 ADR 0010 §Restore 语义。
    `no_origin` = download 物理缺失，UI 应该给用户三选项（拖入替换 / 保留 /
    从 train 移除）而不是隐瞒失败。
    """
    pdir = project_root(p)
    name_list: list[str] = []
    for raw in names:
        _validate_rel_name(raw)
        name_list.append(raw)
    return preprocess_manifest.train_restore(pdir, version_label, name_list)
