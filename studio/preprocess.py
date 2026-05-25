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

from . import project_jobs, projects
from .datasets import IMAGE_EXTS
from .services import preprocess_manifest


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


class PreprocessError(Exception):
    """预处理业务错误（项目不存在 / 参数非法 / 文件名非法）。"""


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


def list_pending(p: dict[str, Any]) -> list[dict[str, Any]]:
    """grid 中走「download 原图」路径的图：download/ 里有 + manifest 未派生到。

    ADR 0004 Addendum 1 §「Stage 不强制时序」：不存在"未处理 vs 已处理"概念，
    只有"当前态从哪儿读"。这里返回的是当前态从 download/ 直接读的那部分，
    跟 `list_processed`（从 preprocess/ 读的派生）合起来覆盖整个 grid。

    一张 download/X.jpg 有任一 manifest entry 的 origin 指向它（含 multi-crop
    派生 X_c0.png / X_c1.png）→ 派生路径已覆盖，不再走 download 原图路径。

    返回 `[{name, mtime, size, w, h}]`，按 name 字典序。w/h 由 PIL 读图头取得
    （未解码 raster，~1ms/图）；前端的像素分布 histogram 需要把这部分一并
    算上（覆盖整个 grid 的分辨率分布）。
    """
    from PIL import Image

    download, _ = project_paths(p)
    pdir = project_root(p)
    preprocess_manifest.ensure_manifest(pdir)  # 老项目首次访问触发迁移
    processed = preprocess_manifest.all_processed(pdir)
    processed_origins = {
        preprocess_manifest.entry_origin(entry, name)
        for name, entry in processed.items()
    }
    removed_origins = preprocess_manifest.duplicate_removed_origins(pdir)

    items: list[dict[str, Any]] = []
    for f in _download_images(download):
        if f.name in processed_origins or f.name in removed_origins:
            continue
        st = f.stat()
        w: Optional[int] = None
        h: Optional[int] = None
        try:
            with Image.open(f) as im:
                w, h = im.size
        except (OSError, ValueError):
            pass
        items.append({
            "name": f.name,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "w": w,
            "h": h,
        })
    return items


def list_processed(p: dict[str, Any]) -> list[dict[str, Any]]:
    """grid 中走「preprocess/ 派生产物」路径的图：manifest 里所有 entry。

    ADR 0004 Addendum 1 §「Stage 不强制时序」：entry 存在不代表"已处理完毕、
    不要再动"，只代表"当前态从 preprocess/{name}.png 读"。下一次放大 / 裁剪
    会按当前态作为输入再生成新产物，覆盖 entry。

    返回 `[{name, mtime, size, w, h, source, model, scale, src_size, dst_size,
             action, target_area, elapsed_seconds, orphan}]`，按 name 字典序。
    `orphan=True`：manifest 有 entry 但源图（download/{source}）已被删。

    `w, h` 从 PIL 读图头（不解码 raster，~1ms / 图）；新 schema 不再 persist
    dst_size，所以靠现读拿到当前实际像素尺寸供前端 histogram 使用。
    """
    from PIL import Image  # local: PIL 在测试 + worker 环境都有

    download, preprocess = project_paths(p)
    pdir = project_root(p)
    preprocess_manifest.ensure_manifest(pdir)
    processed = preprocess_manifest.all_processed(pdir)

    download_stems = {f.stem for f in _download_images(download)}

    items: list[dict[str, Any]] = []
    for name in sorted(processed.keys()):
        entry = processed[name]
        png = preprocess / name
        try:
            st = png.stat()
            mtime, size = st.st_mtime, st.st_size
        except OSError:
            # 产物 PNG 不存在（manifest entry 残留）—— 仍报告，UI 知道异常
            mtime, size = entry.get("mtime", 0.0), 0
        # 现读像素尺寸（PIL lazy load 只解头部）
        w: Optional[int] = None
        h: Optional[int] = None
        try:
            with Image.open(png) as im:
                w, h = im.size
        except (OSError, ValueError):
            # 文件不存在 / 不是图像 / 损坏 — w/h 留 None，前端容忍
            pass
        # 新 schema 用 origin；老 schema 回退到 source；都没就拿 name 自己
        origin_name = preprocess_manifest.entry_origin(entry, name)
        src_stem = Path(origin_name).stem
        items.append({
            "name": name,
            "mtime": mtime,
            "size": size,
            # 实际像素尺寸（前端 pixel histogram 用）
            "w": w,
            "h": h,
            # source 兼容老前端字段名；origin 是新字段名。两个都填 origin_name。
            "source": origin_name,
            "origin": origin_name,
            # 以下字段老 schema 才有，新 entry 一律 None；前端 sidebar 容忍 null。
            "model": entry.get("model"),
            "scale": entry.get("scale"),
            "action": entry.get("action"),
            "target_area": entry.get("target_area"),
            "src_size": entry.get("src_size"),
            "dst_size": entry.get("dst_size"),
            "elapsed_seconds": entry.get("elapsed_seconds"),
            "orphan": src_stem not in download_stems,
        })
    return items

def summary(p: dict[str, Any]) -> dict[str, Any]:
    """给 status 端点用的简短统计。

    `image_count`：grid 当前展示的图总数 = 走 download 原图路径的 +
    走 preprocess 派生路径的（含 multi-crop fan-out 后的多份）。

    ADR 0004 Addendum 1 §「Stage 不强制时序」—— 不返回 "pending / processed"
    分项，那是把 stage 概念硬塞回了 manifest（早被这条约束否决）。前端要分
    grid 来源，直接看 `pending` / `processed` 两个数组就行。
    """
    download, _ = project_paths(p)
    pdir = project_root(p)
    preprocess_manifest.ensure_manifest(pdir)
    processed = preprocess_manifest.all_processed(pdir)
    processed_origins = {
        preprocess_manifest.entry_origin(entry, name)
        for name, entry in processed.items()
    }
    removed_origins = preprocess_manifest.duplicate_removed_origins(pdir)
    n_pending = sum(
        1 for f in _download_images(download)
        if f.name not in processed_origins and f.name not in removed_origins
    )
    return {"image_count": n_pending + len(processed)}


# ---------------------------------------------------------------------------
# 目标选择 + 启动
# ---------------------------------------------------------------------------


_SAFE_NAME_FORBIDDEN = ("/", "\\", "..")


def _validate_name(name: str) -> None:
    if not name or any(t in name for t in _SAFE_NAME_FORBIDDEN):
        raise PreprocessError(f"非法文件名: {name!r}")


def resolve_targets(
    p: dict[str, Any], *, mode: str, names: Optional[Iterable[str]] = None
) -> list[str]:
    """根据 mode + names 返回当前 grid 中要处理的图名列表（已校验、已去重）。

    ADR 0004 Addendum 1 §「Stage 不强制时序」—— 每个 stage 操作的对象是
    "preprocess/ 当前态的全部图"，不是"manifest 里没记的 download 原图"。所以
    mode='all' 含义 = grid 当前所有图（download 原图未派生 ∪ manifest 派生产物），
    worker 端用 resolver 拿实际源，upscaler 内部 `SKIP_MODEL_RATIO` 决定是否跑
    模型 —— 重复跑放大 / 在裁剪产物上再放大都合法。

    mode='all'       → grid 全部当前图
    mode='selected'  → 名单与 (download 实存 ∪ manifest entry 名) 取交集
    mode='all_force' → 同 'all'（保留别名，语义跟 'all' 已对齐；老前端兼容）
    """
    download, _ = project_paths(p)
    if not download.exists() and mode != "selected":
        return []
    existing = {f.name for f in download.iterdir() if _is_image(f)} if download.exists() else set()
    pdir = project_root(p)
    manifest_names = set(preprocess_manifest.all_processed(pdir).keys())

    if mode in ("all", "all_force"):
        # grid 全部当前图 = list_pending names (download 未派生) ∪ list_processed names
        pending_names = {it["name"] for it in list_pending(p)}
        return sorted(pending_names | manifest_names)
    if mode == "selected":
        if not names:
            raise PreprocessError("mode=selected 时 names 不能为空")
        # selected 允许 download/ 原图 + manifest 里已有 entry 的产物名 ——
        # 后者覆盖"重新上调 / 在裁剪产物上再 upscale"的链路（ADR 0004 Addendum 1
        # §「Stage 不强制时序」）。worker 端用 resolve() 找实际源。
        selectable = existing | manifest_names
        chosen = []
        for n in names:
            _validate_name(n)
            if n in selectable:
                chosen.append(n)
        # 保留唯一 + 字典序，便于日志稳定
        return sorted(set(chosen))
    raise PreprocessError(f"未知 mode: {mode!r}")


def start_job(
    conn,
    *,
    project_id: int,
    mode: str = "all",
    names: Optional[list[str]] = None,
    model: str = DEFAULT_MODEL,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_pad: int = DEFAULT_TILE_PAD,
    device: str = DEFAULT_DEVICE,
    target_area: Optional[int] = DEFAULT_TARGET_AREA,
) -> dict[str, Any]:
    """创建预处理 job。worker 自己读 params 决定要做什么。

    不在此处真去 resolve 目标 — worker 启动时再扫一遍盘，避免 webui 请求
    线程因为大目录列举耗时。
    """
    p = projects.get_project(conn, project_id)
    if not p:
        raise PreprocessError(f"项目不存在: id={project_id}")
    if mode not in ("all", "selected", "all_force"):
        raise PreprocessError(f"未知 mode: {mode!r}")
    if mode == "selected" and not names:
        raise PreprocessError("mode=selected 必须给 names")

    # 简单校验 names 不含路径分隔（worker 还会再 validate 一次）
    if names:
        for n in names:
            _validate_name(n)

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
        kind=PREPROCESS_KIND,
        params=params,
    )


def list_crop_workspace(p: dict[str, Any]) -> list[dict[str, Any]]:
    """裁剪页的工作集：所有可被裁剪的图（来自 preprocess/ + 未处理的 download/）。

    每项含像素尺寸，因为前端聚类 / AR 显示需要。读图头不解码（PIL lazy load
    `.size`），单图 < 1ms。返回 [{name, source, w, h, mtime, size, processed}]。

    source 字段：当前 preprocess 项的 origin（指 download/{...}），下游可以反查
    `download/` 拿原图。
    """
    from PIL import Image

    download, preprocess = project_paths(p)
    pdir = project_root(p)
    preprocess_manifest.ensure_manifest(pdir)
    processed = preprocess_manifest.all_processed(pdir)
    removed_origins = preprocess_manifest.duplicate_removed_origins(pdir)

    items: list[dict[str, Any]] = []
    seen_origins: set[str] = set()

    # 已处理（manifest 里登记的）
    for name in sorted(processed.keys()):
        entry = processed[name]
        png = preprocess / name
        if not png.is_file():
            continue
        try:
            with Image.open(png) as im:
                w, h = im.size
        except OSError:
            continue
        st = png.stat()
        origin = preprocess_manifest.entry_origin(entry, name)
        items.append({
            "name": name,
            "source": origin,
            "w": w,
            "h": h,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "processed": True,
        })
        seen_origins.add(origin)

    # 未处理（download/ 里没在 manifest 中追溯到的）
    if download.exists():
        for f in sorted(_download_images(download)):
            if f.name in seen_origins or f.name in removed_origins:
                continue
            try:
                with Image.open(f) as im:
                    w, h = im.size
            except OSError:
                continue
            st = f.stat()
            items.append({
                "name": f.name,
                "source": f.name,
                "w": w,
                "h": h,
                "mtime": st.st_mtime,
                "size": st.st_size,
                "processed": False,
            })
    return items


def list_duplicate_removed_workspace(p: dict[str, Any]) -> list[dict[str, Any]]:
    """已软删除的图（被去重审核标记 kind=duplicate_removed），按 name 字典序。

    展示在总览页「已删除」tab：物理图仍在 `download/{origin}`，缩略图按
    download bucket + origin 名取（thumb 端点已不再对 duplicate_removed
    返 404）。返回 `[{name, source, w, h, mtime, size}]`，source = origin
    = download/ 下原图名。
    """
    from PIL import Image

    download, _ = project_paths(p)
    pdir = project_root(p)
    preprocess_manifest.ensure_manifest(pdir)
    removed = preprocess_manifest.duplicate_removed(pdir)

    items: list[dict[str, Any]] = []
    for name in sorted(removed.keys()):
        entry = removed[name]
        origin = preprocess_manifest.entry_origin(entry, name)
        src = download / origin
        if not src.is_file():
            # origin 文件已被外部删除：仍展示一条 stale entry 方便用户恢复 entry
            # 自身（thumb 会 404，UI 容忍）；w/h 留 None。
            items.append({
                "name": name,
                "source": origin,
                "w": None,
                "h": None,
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
            "w": w,
            "h": h,
            "mtime": st.st_mtime,
            "size": st.st_size,
        })
    return items


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


def start_crop_job(
    conn,
    *,
    project_id: int,
    crops: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """创建裁剪 job。

    `crops`：`{源文件名: [{x, y, w, h, label?}, ...]}`，每条 rect 归一化 [0,1]。
    源文件名为 preprocess/ 下当前文件名；不存在时 worker 兜底到 download/。

    worker 端逻辑：对每个 source，
      - N=1 → 写 `preprocess/{stem}.png`（覆盖源）
      - N>1 → 写 `preprocess/{stem}_c{n}.png` × N，并删除原 `preprocess/{stem}.png`
      - manifest 用 `replace_with_crops()` 原子替换 entry

    本函数只做参数校验 + 入库；磁盘操作走 worker。
    """
    p = projects.get_project(conn, project_id)
    if not p:
        raise PreprocessError(f"项目不存在: id={project_id}")
    if not isinstance(crops, dict) or not crops:
        raise PreprocessError("crops 不能为空")
    # 校验：名字合法 + 至少一个 rect + 每条 rect 在 [0,1]
    sanitized: dict[str, list[dict[str, Any]]] = {}
    for name, rects in crops.items():
        _validate_name(name)
        if not isinstance(rects, list) or not rects:
            raise PreprocessError(f"{name!r} 的 rects 为空")
        out_rects: list[dict[str, Any]] = []
        for r in rects:
            if not isinstance(r, dict):
                raise PreprocessError(f"{name!r} 含非法 rect: {r!r}")
            clean = _validate_rect(r)
            label = r.get("label")
            if label is not None:
                # 限制 label 长度防止 manifest 膨胀；label 当下其实不持久化，仅
                # worker 日志会带；保留字段方便未来扩展。
                clean["label"] = str(label)[:64]
            out_rects.append(clean)
        sanitized[name] = out_rects

    params = {"stage": STAGE_CROP, "crops": sanitized}
    return project_jobs.create_job(
        conn,
        project_id=project_id,
        kind=PREPROCESS_KIND,
        params=params,
    )


# ---------------------------------------------------------------------------
# 还原（删 manifest entry + 删 preprocess/{name} PNG）
# ---------------------------------------------------------------------------


def restore_products(
    p: dict[str, Any], names: Iterable[str]
) -> dict[str, list[str]]:
    """还原指定产物：manifest 删 entry + 删 preprocess/{name} PNG。

    还原后该图回到「隐式 original」状态——下游 resolve 会重新指向 download/。
    返回 `{restored, missing}`：manifest 里没 entry 的记 missing；PNG 不存在不算
    missing（自愈：orphan PNG 一并清理）。

    `names` 为产物文件名（如 `foo.png`），不是源名。
    """
    pdir = project_root(p)
    name_list: list[str] = []
    for raw in names:
        _validate_name(raw)
        name_list.append(raw)
    return preprocess_manifest.restore(pdir, name_list)


# 旧 API 兼容别名（v1 完整下线 sidecar 后可删；前端调用点已替换为 /restore）
delete_products = restore_products
