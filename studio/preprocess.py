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
DEFAULT_MODEL = "4x-AnimeSharp"
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_PAD = 16
DEFAULT_DEVICE = "auto"
# LoRA 训练桶的目标面积。1024² = 1048576 px 是 SDXL/Flux/Anima 常用桶；用户
# 可以在 UI 选 768²/1024²/1536²/2048² 或自定义边长。
DEFAULT_TARGET_AREA = 1024 * 1024

PRODUCT_SUFFIX = ".png"


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
    """download/ 里存在、但 manifest 没记的图（= 隐式 original = 未处理）。

    返回 `[{name, mtime, size}]`，按 name 字典序。"""
    download, _ = project_paths(p)
    pdir = project_root(p)
    preprocess_manifest.ensure_manifest(pdir)  # 老项目首次访问触发迁移
    processed_names = set(preprocess_manifest.all_processed(pdir).keys())

    items: list[dict[str, Any]] = []
    for f in _download_images(download):
        product_name = product_path_for(Path(), f.name).name  # `{stem}.png`
        if product_name in processed_names:
            continue
        st = f.stat()
        items.append({"name": f.name, "mtime": st.st_mtime, "size": st.st_size})
    return items


def list_processed(p: dict[str, Any]) -> list[dict[str, Any]]:
    """manifest 里 kind=processed 的图，按 name 字典序。

    返回 `[{name, mtime, size, source, model, scale, src_size, dst_size,
             action, target_area, elapsed_seconds, orphan}]`。
    `orphan=True`：manifest 有 entry 但源图（download/{source}）已被删。"""
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
        source_name = entry.get("source") or name
        src_stem = Path(source_name).stem
        items.append({
            "name": name,
            "mtime": mtime,
            "size": size,
            "source": entry.get("source"),
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
    """给 status 端点用的简短统计。"""
    download, _ = project_paths(p)
    pdir = project_root(p)
    preprocess_manifest.ensure_manifest(pdir)
    n_download = len(_download_images(download))
    n_processed = len(preprocess_manifest.all_processed(pdir))
    # pending = download 里没在 processed 集合的（按 stem 匹配）
    processed_stems = {Path(n).stem for n in preprocess_manifest.all_processed(pdir)}
    n_pending = sum(
        1 for f in _download_images(download) if f.stem not in processed_stems
    )
    return {
        "download_count": n_download,
        "processed_count": n_processed,
        "pending_count": n_pending,
    }


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
    """根据 mode + names 返回需要处理的源文件名列表（已校验、已去重）。

    mode='all'      → 所有 download/ 里、manifest 还没记 processed 的图（增量）
    mode='selected' → 名单与 download/ 实际存在的图取交集
    mode='all_force' → 所有 download/ 图（manifest 已有 entry 也重跑）
    """
    download, _ = project_paths(p)
    if not download.exists():
        return []
    existing = {f.name for f in download.iterdir() if _is_image(f)}

    if mode == "all":
        return sorted(it["name"] for it in list_pending(p))
    if mode == "all_force":
        return sorted(existing)
    if mode == "selected":
        if not names:
            raise PreprocessError("mode=selected 时 names 不能为空")
        chosen = []
        for n in names:
            _validate_name(n)
            if n in existing:
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
