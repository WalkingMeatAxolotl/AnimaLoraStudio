"""预处理业务层：列表 / 状态 / 启动 job。

第一阶段只做"放大"，但目录契约和接口预留好裁剪 / 涂抹的位置。

目录契约
--------
`projects/{id}-{slug}/preprocess/` 存所有产物，每张图旁可选 sidecar
`{name}.preprocess.json` 记录来源 + 参数：

    {source, model, scale, tile_size, tile_pad, device, src_size, dst_size,
     elapsed_seconds, mtime}

产物文件名规则：固定 `{src_stem}.png`。同 stem 但不同扩展名的源图碰撞时
（如 `cat.jpg` 和 `cat.png` 同存）— 后处理的覆盖前者，并在日志里 warn。

筛选阶段（curate）发现 preprocess/ 非空时，左侧源切到 preprocess/；
否则继续从 download/ 读。

Job 调度
--------
preprocess 是 GPU-bound job kind，走 DATA 槽位：
- 训练正在跑 + 未开 `allow_gpu_during_train` → 推迟
- daemon 占着 VRAM → 触发让位（_maybe_yield_daemon），等下次 tick

不复用 download_worker 的并发设计 —— 串行处理就行，模型加载到 GPU 后
单张耗时 1-3s（4x，512px 输入，cuda）。批量并发的收益不抵 VRAM 风险。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from . import project_jobs, projects
from .datasets import IMAGE_EXTS


PREPROCESS_KIND = "preprocess"
DEFAULT_MODEL = "4x-AnimeSharp"
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_PAD = 16
DEFAULT_DEVICE = "auto"
# LoRA 训练桶的目标面积。1024² = 1048576 px 是 SDXL/Flux/Anima 常用桶；用户
# 可以在 UI 选 768²/1024²/1536²/2048² 或自定义边长。
DEFAULT_TARGET_AREA = 1024 * 1024

PRODUCT_SUFFIX = ".png"
SIDECAR_SUFFIX = ".preprocess.json"


class PreprocessError(Exception):
    """预处理业务错误（项目不存在 / 参数非法 / 文件名非法）。"""


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------


def project_paths(p: dict[str, Any]) -> tuple[Path, Path]:
    """返回 `(download_dir, preprocess_dir)`，不保证存在。"""
    pdir = projects.project_dir(p["id"], p["slug"])
    return pdir / "download", pdir / "preprocess"


def product_path_for(preprocess_dir: Path, source_name: str) -> Path:
    """`download/foo.webp` → `preprocess/foo.png`。"""
    stem = Path(source_name).stem
    return preprocess_dir / f"{stem}{PRODUCT_SUFFIX}"


def sidecar_for(product_path: Path) -> Path:
    """产物路径 → sidecar 路径。"""
    return product_path.with_suffix(product_path.suffix + SIDECAR_SUFFIX)


# ---------------------------------------------------------------------------
# 列表 / 状态
# ---------------------------------------------------------------------------


def _read_sidecar(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def list_pending(p: dict[str, Any]) -> list[dict[str, Any]]:
    """download/ 里存在、但 preprocess/ 还没有对应产物的图。

    返回 `[{name, mtime, size}]`，按 name 字典序。"""
    download, preprocess = project_paths(p)
    if not download.exists():
        return []
    items: list[dict[str, Any]] = []
    for f in sorted(download.iterdir()):
        if not _is_image(f):
            continue
        product = product_path_for(preprocess, f.name)
        if product.exists():
            continue
        st = f.stat()
        items.append({"name": f.name, "mtime": st.st_mtime, "size": st.st_size})
    return items


def list_processed(p: dict[str, Any]) -> list[dict[str, Any]]:
    """preprocess/ 里已存在的产物。

    返回 `[{name, mtime, size, source, model, scale, src_size, dst_size}]`；
    sidecar 缺失时 source/model 等字段为 None（仍然返回，但 UI 知道是孤儿）。
    """
    download, preprocess = project_paths(p)
    if not preprocess.exists():
        return []
    items: list[dict[str, Any]] = []
    for f in sorted(preprocess.iterdir()):
        if not _is_image(f):
            continue
        st = f.stat()
        meta = _read_sidecar(sidecar_for(f)) or {}
        items.append({
            "name": f.name,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "source": meta.get("source"),
            "model": meta.get("model"),
            "scale": meta.get("scale"),
            "action": meta.get("action"),
            "target_area": meta.get("target_area"),
            "src_size": meta.get("src_size"),
            "dst_size": meta.get("dst_size"),
            "elapsed_seconds": meta.get("elapsed_seconds"),
        })
    # 仅做 sanity：源已被删的孤儿产物加 orphan=True 标记，让 UI 提醒可清理
    download_stems = (
        {f.stem for f in download.iterdir() if _is_image(f)}
        if download.exists() else set()
    )
    for it in items:
        if it["source"]:
            src_stem = Path(it["source"]).stem
        else:
            src_stem = Path(it["name"]).stem
        it["orphan"] = src_stem not in download_stems
    return items


def summary(p: dict[str, Any]) -> dict[str, Any]:
    """给 status 端点用的简短统计。"""
    download, preprocess = project_paths(p)
    n_download = sum(1 for f in download.iterdir() if _is_image(f)) if download.exists() else 0
    n_processed = sum(1 for f in preprocess.iterdir() if _is_image(f)) if preprocess.exists() else 0
    return {
        "download_count": n_download,
        "processed_count": n_processed,
        "pending_count": max(0, n_download - n_processed),
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

    mode='all'      → 所有 download/ 里、preprocess/ 还没产物的图（增量）
    mode='selected' → 名单与 download/ 实际存在的图取交集
    mode='all_force' → 所有 download/ 图（已有产物也重跑）
    """
    download, preprocess = project_paths(p)
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
# 删除产物
# ---------------------------------------------------------------------------


def delete_products(
    p: dict[str, Any], names: Iterable[str]
) -> dict[str, list[str]]:
    """删除 preprocess/ 下指定产物 + 同名 sidecar。

    `names` 为产物文件名（如 `foo.png`），不是源名。"""
    _, preprocess = project_paths(p)
    deleted: list[str] = []
    missing: list[str] = []
    for raw in names:
        _validate_name(raw)
        target = preprocess / raw
        if not target.exists():
            missing.append(raw)
            continue
        try:
            target.unlink()
        except OSError as exc:
            raise PreprocessError(f"删除失败 {raw}: {exc}")
        side = sidecar_for(target)
        if side.exists():
            try:
                side.unlink()
            except OSError:
                pass  # sidecar 删失败不阻断
        deleted.append(raw)
    return {"deleted": deleted, "missing": missing}
