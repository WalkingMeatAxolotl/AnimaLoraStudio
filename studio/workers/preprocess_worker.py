"""预处理 worker 子进程入口（放大 + 裁剪）。

由 supervisor 启动：`python -m studio.workers.preprocess_worker --job-id N`。

读 project_jobs 行 → 按 `params['stage']` 分发：
  - stage='upscale' (默认)：串行调 `studio.services.upscaler.upscale_file()`
  - stage='crop'：用 PIL 把 preprocess/ 下的图按归一化 rect 切成 N 张产物

日志规范：只走 stdout（supervisor 重定向到 log 文件），不要再 open 同一个
log 文件，避免 LogTailer 读两次。

取消：worker 主体在每张图前检测 SIGTERM/CTRL_BREAK 信号（Python 解释器
默认对 SIGTERM 抛 KeyboardInterrupt 在 main thread 里）；当前轮的图处理完
后干净退出，已写盘的产物保留（增量）。
"""
from __future__ import annotations

import json
import logging
import math
import signal
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

logger = logging.getLogger(__name__)

from studio import db
from studio.services.preprocess import core as preprocess
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services import models as model_downloader
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.inference import upscaler


_stop_requested = False


def _on_signal(_signum, _frame) -> None:  # pragma: no cover - signal path
    global _stop_requested
    _stop_requested = True


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGBREAK"):  # Windows
        signal.signal(signal.SIGBREAK, _on_signal)  # type: ignore[attr-defined]


def run(job_id: int) -> int:  # noqa: PLR0912, PLR0915 - 主流程线性可读
    _install_signal_handlers()

    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != preprocess.PREPROCESS_KIND:
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params = job.get("params_decoded") or {}
    # 缺 stage 字段视为老 upscale job（向后兼容）
    stage = params.get("stage", preprocess.STAGE_UPSCALE)

    def log(line: str) -> None:
        print(line, flush=True)

    def emit_event(evt_type: str, **payload) -> None:
        """通过 stdout 标记行 → supervisor 解析 → SSE。供前端实时更新用，
        不会进 job 日志。supervisor 端常量见 `studio/supervisor.py:_EVENT_MARKER`。"""
        try:
            print(f"__EVENT__:{evt_type}:{json.dumps(payload, ensure_ascii=False)}", flush=True)
        except Exception:  # noqa: BLE001 — 推事件失败不影响主流程
            pass

    try:
        with db.connection_for() as conn:
            project = projects.get_project(conn, job["project_id"])
        if not project:
            log(f"[error] project {job['project_id']} missing")
            return 1

        # ADR 0010 train-scope 分发：job.version_id 非空 → 新 train scope 路径；
        # None → 老 project scope 路径（backward-compat，PR-3 删）
        version_id = job.get("version_id")
        if version_id is not None:
            with db.connection_for() as conn:
                version = versions.get_version(conn, version_id)
            if not version:
                log(f"[error] version {version_id} missing")
                return 1
            if stage == preprocess.STAGE_CROP:
                return _run_crop_train(project, version, params, log, emit_event)
            if stage == preprocess.STAGE_UPSCALE:
                return _run_upscale_train(
                    project, version, params, log, emit_event,
                )
            log(f"[error] 未知 stage: {stage!r}")
            return 1

        if stage == preprocess.STAGE_CROP:
            return _run_crop(project, params, log, emit_event)
        if stage != preprocess.STAGE_UPSCALE:
            log(f"[error] 未知 stage: {stage!r}")
            return 1

        mode = params.get("mode", "all")
        names = params.get("names") or None
        model_label = params.get("model", preprocess.DEFAULT_MODEL)
        tile_size = int(params.get("tile_size", preprocess.DEFAULT_TILE_SIZE))
        tile_pad = int(params.get("tile_pad", preprocess.DEFAULT_TILE_PAD))
        device = params.get("device", preprocess.DEFAULT_DEVICE)
        # target_area = None 是 "直接 4×" 路径；非 None 走智能流水（够大跳过模型）
        target_area_raw = params.get("target_area", preprocess.DEFAULT_TARGET_AREA)
        target_area = int(target_area_raw) if target_area_raw else None

        download_dir, preprocess_dir = preprocess.project_paths(project)
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        project_dir = projects.project_dir(project["id"], project["slug"])

        # 模型权重必须先下载（UI 在开始按钮前会引导用户下载）
        model_path = model_downloader.upscaler_target(model_label)
        if not model_path.exists():
            log(
                f"[error] 模型权重不存在：{model_path}（请先在设置页下载 {model_label}）"
            )
            return 1

        try:
            sources = preprocess.resolve_targets(project, mode=mode, names=names)
        except preprocess.PreprocessError as exc:
            log(f"[error] 解析目标失败: {exc}")
            return 1

        total = len(sources)
        if total == 0:
            log("[done] 没有需要处理的图（已全部预处理）")
            return 0

        target_desc = (
            f"{int(math.sqrt(target_area))}²={target_area}px"
            if target_area else "off (直接 4×)"
        )
        log(
            f"[start] mode={mode} model={model_label} tile={tile_size}+{tile_pad} "
            f"device={device} target={target_desc} total={total}"
        )

        # 解析一次实际 device + dtype 并 log，让用户能看出真在用 GPU/fp16 还是
        # 悄悄降级到了 CPU。先做一次以打印诊断信息（也顺便预热模型缓存，省第一张
        # cold-start 时间）。
        try:
            import torch
            resolved_dev = upscaler.resolve_device(device)
            resolved_dtype = upscaler.resolve_dtype("auto", resolved_dev)
            gpu_name = (
                torch.cuda.get_device_name(0)
                if resolved_dev.type == "cuda" and torch.cuda.is_available()
                else "—"
            )
            log(
                f"[device] resolved={resolved_dev} dtype={str(resolved_dtype).replace('torch.', '')} "
                f"gpu={gpu_name} cuda_available={torch.cuda.is_available()}"
            )
            upscaler.load_model(model_path, device=resolved_dev, dtype=resolved_dtype)
            log(f"[model] {model_label} loaded → {resolved_dev}")
        except Exception as exc:  # noqa: BLE001
            log(f"[device] diagnostic failed: {exc}（继续，但可能跑在 CPU 上）")

        succeeded = 0
        failed = 0
        skipped = 0

        for idx, src_name in enumerate(sources, start=1):
            if _stop_requested:
                log(f"[cancel] 收到取消信号，已处理 {idx - 1}/{total}")
                break
            # ADR 0004 §149 resolver 单点：manifest 有 entry → preprocess/{name}；
            # 否则 → download/{name}（隐式 original）。worker 不自己决定源在哪。
            src_path = preprocess_manifest.resolve(project_dir, src_name)
            if src_path is None or not src_path.exists():
                log(f"[skip] ({idx}/{total}) {src_name}: 源已不存在")
                skipped += 1
                emit_event(
                    "preprocess_progress",
                    idx=idx, total=total, name=src_name, status="skip",
                    succeeded=succeeded, failed=failed, skipped=skipped,
                )
                continue
            # origin 沿用 manifest 里已有的（含 multi-crop 的根 origin），没就用 name 本身。
            existing_entry = preprocess_manifest.get_entry(project_dir, src_name)
            origin_name = (
                preprocess_manifest.entry_origin(existing_entry, src_name)
                if existing_entry is not None
                else src_name
            )
            dst_path = preprocess.product_path_for(preprocess_dir, src_name)
            # ADR 0004 Addendum 1 §「Stage 不强制时序」—— 不按 manifest 是否有
            # entry 判断该不该跑。是否真的"动盘 + 跑模型"由 upscaler 内部
            # `SKIP_MODEL_RATIO` 决定（src 面积 ≥ 0.95×target 走 LANCZOS，否则
            # 模型 + LANCZOS）。这样"裁剪后→放大"/"放大后→放大"都是合法链路。
            log(f"[upscale] ({idx}/{total}) {src_name} → {dst_path.name}")
            try:
                meta = upscaler.upscale_file(
                    src_path,
                    dst_path,
                    model_path=model_path,
                    label=model_label,
                    tile_size=tile_size,
                    tile_pad=tile_pad,
                    device=device,
                    target_area=target_area,
                    on_log=log,
                    # 256 给 grid，768 给 curate alt-hover 大图。worker 阶段付一次
                    # decode 代价，前端首次浏览就秒开。
                    prewarm_thumb_sizes=[256, 768],
                )
                # 写 manifest：ADR 0004 — 状态唯一真理，downstream resolve 用
                meta["origin"] = origin_name
                preprocess_manifest.add_processed(
                    project_dir,
                    dst_path.name,
                    meta,
                )
                succeeded += 1
                emit_event(
                    "preprocess_progress",
                    idx=idx, total=total, name=src_name, status="done",
                    action=meta.get("action"),
                    succeeded=succeeded, failed=failed, skipped=skipped,
                )
            except Exception as exc:  # noqa: BLE001 — 单张失败不影响其他
                log(f"[fail] {src_name}: {exc}")
                failed += 1
                emit_event(
                    "preprocess_progress",
                    idx=idx, total=total, name=src_name, status="fail",
                    error=str(exc)[:200],
                    succeeded=succeeded, failed=failed, skipped=skipped,
                )

        log(
            f"[done] succeeded={succeeded} failed={failed} skipped={skipped}"
        )
        # 即使部分失败也返 0（成功完成 job 流程）；失败信息在日志里。
        # 失败率高时用户重跑选中即可。
        return 0
    except Exception as exc:  # noqa: BLE001
        # PR-1 C7: 同 tag_worker — logger.exception 带 trace_id 进 stderr，
        # log 给人读短摘要。
        logger.exception("preprocess worker crashed (job_id=%s)", job_id)
        log(f"[error] {exc}")
        return 1


def _resolve_crop_source(
    project_dir: Path,
    download_dir: Path,
    preprocess_dir: Path,
    name: str,
) -> tuple[Path | None, str]:
    """裁剪源文件解析：
       - manifest 有 entry → preprocess/{name}，origin 取自 entry
       - 没 entry，preprocess/{name} 存在 → 直接用，origin = name
       - 没 entry，preprocess/{name} 不存在，download/{name} 存在 → 兜底 download，origin = name
       - 都不存在 → (None, name)

    返回 (磁盘路径 or None, 该图的 origin)。
    """
    entry = preprocess_manifest.get_entry(project_dir, name)
    if entry is not None:
        origin = preprocess_manifest.entry_origin(entry, name)
        return preprocess_dir / name, origin
    pp = preprocess_dir / name
    if pp.is_file():
        return pp, name
    dl = download_dir / name
    if dl.is_file():
        return dl, name
    return None, name


def _run_crop(
    project: dict[str, Any],
    params: dict[str, Any],
    log: Callable[[str], None],
    emit_event: Callable[..., None],
) -> int:
    """裁剪 stage：每张源图按 N 个归一化 rect 切成 N 个 PNG 产物。

    SSE 节流：crop 速度比 upscale 快（单图 0.3–0.7s），264 张数据集会刷 ~500
    个事件给前端。节流到 ≥1s 间隔 + 始终发首末事件（idx=1 / idx=total / 失败 /
    跳过），保留进度可见性同时不淹没事件流。前端 useEventStream 见到节流后的
    速率即可（不再依赖前端 throttle）。
    """
    download_dir, preprocess_dir = preprocess.project_paths(project)
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    project_dir = projects.project_dir(project["id"], project["slug"])

    crops_param = params.get("crops") or {}
    if not crops_param:
        log("[done] crops 为空，无事可做")
        return 0
    # 排序保证日志稳定
    sources = sorted(crops_param.keys())

    # SSE throttle state — closure over emit_event so call sites stay uniform
    _last_emit_at = [0.0]

    def emit_throttled(*, force: bool, **payload) -> None:
        """Throttle progress emits to ≥1Hz. `force=True` always emits (first /
        last / non-success status — those carry information you don't want
        coalesced)."""
        now = time.monotonic()
        if not force and (now - _last_emit_at[0]) < 1.0:
            return
        _last_emit_at[0] = now
        emit_event("crop_progress", **payload)
    total = len(sources)
    log(f"[start] stage=crop total={total}")

    succeeded = 0
    failed = 0
    skipped = 0

    for idx, src_name in enumerate(sources, start=1):
        if _stop_requested:
            log(f"[cancel] 收到取消信号，已处理 {idx - 1}/{total}")
            break
        is_last = idx == total
        try:
            preprocess._validate_name(src_name)
        except preprocess.PreprocessError as exc:
            log(f"[skip] {src_name}: {exc}")
            skipped += 1
            # skip/fail are info-bearing → always force emit, don't coalesce
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_name, status="skip",
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
            continue

        rects = crops_param[src_name]
        src_path, origin = _resolve_crop_source(
            project_dir, download_dir, preprocess_dir, src_name
        )
        if src_path is None or not src_path.is_file():
            log(f"[skip] ({idx}/{total}) {src_name}: 源不存在")
            skipped += 1
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_name, status="skip",
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
            continue

        n = len(rects)
        src_stem = Path(src_name).stem
        out_names = (
            [f"{src_stem}.png"] if n == 1
            else [f"{src_stem}_c{i}.png" for i in range(n)]
        )

        log(f"[crop] ({idx}/{total}) {src_name} → {n} 个产物")
        try:
            t0 = time.monotonic()
            with Image.open(src_path) as raw:
                raw.load()
                src_img = raw.convert("RGB") if raw.mode != "RGB" else raw.copy()
            sw, sh = src_img.size
            outputs: list[dict[str, Any]] = []
            for r, out_name in zip(rects, out_names):
                left = int(round(r["x"] * sw))
                top = int(round(r["y"] * sh))
                right = int(round((r["x"] + r["w"]) * sw))
                bottom = int(round((r["y"] + r["h"]) * sh))
                # 至少 1 px
                right = max(left + 1, right)
                bottom = max(top + 1, bottom)
                piece = src_img.crop((left, top, right, bottom))
                out_path = preprocess_dir / out_name
                # 写盘前先写到临时文件再 rename，防止半写覆盖原文件被读
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                piece.save(tmp_path, format="PNG", optimize=False)
                # tmp 与目标同目录 → os.replace 跨平台原子
                import os as _os
                _os.replace(tmp_path, out_path)
                try:
                    st = out_path.stat()
                    sz, mt = st.st_size, st.st_mtime
                except OSError:
                    sz, mt = 0, time.time()
                outputs.append({
                    "name": out_name,
                    "origin": origin,
                    "size": sz,
                    "mtime": mt,
                })

            # N>1 的多裁剪：原同名 stem.png 应当删（如果它不在 outputs 列表里）
            if n > 1:
                stale = preprocess_dir / f"{src_stem}.png"
                if stale.is_file() and stale.name not in out_names:
                    try:
                        stale.unlink()
                    except OSError as exc:
                        log(f"   ⚠ 删旧 {stale.name} 失败: {exc}")

            preprocess_manifest.replace_with_crops(
                project_dir,
                source_name=src_name,
                outputs=outputs,
            )
            # 给前端 grid 预热缩略图
            try:
                from studio.services.dataset import thumb_cache
                for out_name in out_names:
                    out_path = preprocess_dir / out_name
                    with Image.open(out_path) as piece:
                        piece.load()
                        thumb_cache.prewarm_from_image(out_path, piece, [256, 768])
            except Exception as exc:  # noqa: BLE001 — 缩略图失败不影响主流程
                log(f"   ⚠ thumb prewarm failed: {exc}")

            elapsed = time.monotonic() - t0
            succeeded += 1
            log(
                f"   ✓ {src_name} → {', '.join(out_names)}  "
                f"({sw}×{sh} → {n} 块, {elapsed:.2f}s)"
            )
            # done events get throttled (high volume); force first/last for UI
            emit_throttled(
                force=(idx == 1 or is_last),
                idx=idx, total=total, name=src_name, status="done",
                n_out=n, outputs=out_names,
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
        except Exception as exc:  # noqa: BLE001 — 单张失败不影响其他
            log(f"[fail] {src_name}: {exc}")
            failed += 1
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_name, status="fail",
                error=str(exc)[:200],
                succeeded=succeeded, failed=failed, skipped=skipped,
            )

    log(f"[done] succeeded={succeeded} failed={failed} skipped={skipped}")
    return 0


def _run_upscale_train(
    project: dict[str, Any],
    version: dict[str, Any],
    params: dict[str, Any],
    log: Callable[[str], None],
    emit_event: Callable[..., None],
) -> int:
    """ADR 0010 train-scope upscale。

    源 + 产物都在 `versions/{label}/train/{folder}/`，manifest 写到
    `versions/{label}/train/manifest.json`。PNG 输出 stem 不变但扩展名可能
    改（X.jpg → X.png）—— `train_swap_entry` 原子替换 manifest，物理 src
    单独 unlink 避免训练集 stem 冲突；caption (.txt) 同 stem 保留不动。
    """
    mode = params.get("mode", "all")
    names = params.get("names") or None
    model_label = params.get("model", preprocess.DEFAULT_MODEL)
    tile_size = int(params.get("tile_size", preprocess.DEFAULT_TILE_SIZE))
    tile_pad = int(params.get("tile_pad", preprocess.DEFAULT_TILE_PAD))
    device = params.get("device", preprocess.DEFAULT_DEVICE)
    target_area_raw = params.get("target_area", preprocess.DEFAULT_TARGET_AREA)
    target_area = int(target_area_raw) if target_area_raw else None

    project_dir = projects.project_dir(project["id"], project["slug"])
    train_dir = preprocess.version_train_dir(project, version["label"])
    train_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_downloader.upscaler_target(model_label)
    if not model_path.exists():
        log(
            f"[error] 模型权重不存在：{model_path}（请先在设置页下载 {model_label}）"
        )
        return 1

    try:
        sources = preprocess.resolve_targets_train(
            project, version["label"], mode=mode, names=names
        )
    except preprocess.PreprocessError as exc:
        log(f"[error] 解析目标失败: {exc}")
        return 1

    total = len(sources)
    if total == 0:
        log("[done] 没有需要处理的图")
        return 0

    target_desc = (
        f"{int(math.sqrt(target_area))}²={target_area}px"
        if target_area else "off (直接 4×)"
    )
    log(
        f"[start] mode={mode} model={model_label} tile={tile_size}+{tile_pad} "
        f"device={device} target={target_desc} total={total} scope=train"
    )

    try:
        import torch
        resolved_dev = upscaler.resolve_device(device)
        resolved_dtype = upscaler.resolve_dtype("auto", resolved_dev)
        gpu_name = (
            torch.cuda.get_device_name(0)
            if resolved_dev.type == "cuda" and torch.cuda.is_available()
            else "—"
        )
        log(
            f"[device] resolved={resolved_dev} dtype={str(resolved_dtype).replace('torch.', '')} "
            f"gpu={gpu_name} cuda_available={torch.cuda.is_available()}"
        )
        upscaler.load_model(model_path, device=resolved_dev, dtype=resolved_dtype)
        log(f"[model] {model_label} loaded → {resolved_dev}")
    except Exception as exc:  # noqa: BLE001
        log(f"[device] diagnostic failed: {exc}（继续，但可能跑在 CPU 上）")

    succeeded = 0
    failed = 0
    skipped = 0

    for idx, src_rel in enumerate(sources, start=1):
        if _stop_requested:
            log(f"[cancel] 收到取消信号，已处理 {idx - 1}/{total}")
            break
        src_path = train_dir / src_rel
        if not src_path.exists():
            log(f"[skip] ({idx}/{total}) {src_rel}: 源已不存在")
            skipped += 1
            emit_event(
                "preprocess_progress",
                idx=idx, total=total, name=src_rel, status="skip",
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
            continue

        # origin 沿用 manifest 已有 entry（multi-crop 派生 root），否则用 rel
        # path 末段（curate 复制图时写的就是 file name == origin）
        existing = preprocess_manifest.train_get_entry(
            project_dir, version["label"], src_rel
        )
        src_filename = src_rel.rsplit("/", 1)[-1]
        if existing is not None:
            origin_name = preprocess_manifest.entry_origin(existing, src_filename)
        else:
            origin_name = src_filename

        # dst rel：同 folder 同 stem，扩展名固定 .png（upscaler 输出 PNG）
        folder, _ = src_rel.split("/", 1)
        src_stem = Path(src_filename).stem
        dst_rel = f"{folder}/{src_stem}.png"
        dst_path = train_dir / dst_rel

        log(f"[upscale] ({idx}/{total}) {src_rel} → {dst_rel}")
        try:
            meta = upscaler.upscale_file(
                src_path,
                dst_path,
                model_path=model_path,
                label=model_label,
                tile_size=tile_size,
                tile_pad=tile_pad,
                device=device,
                target_area=target_area,
                on_log=log,
                prewarm_thumb_sizes=[256, 768],
            )
            meta["origin"] = origin_name
            if dst_rel != src_rel:
                # 扩展名变 → 原子 swap entry + 删 src 物理文件
                preprocess_manifest.train_swap_entry(
                    project_dir, version["label"],
                    old_name=src_rel, new_name=dst_rel, meta=meta,
                )
                try:
                    src_path.unlink()
                except OSError as exc:
                    log(f"   ⚠ 删旧 {src_rel} 失败: {exc}")
            else:
                # 覆盖原文件（src == dst rel path）→ 普通 add_processed
                preprocess_manifest.train_add_processed(
                    project_dir, version["label"], dst_rel, meta,
                )
            succeeded += 1
            emit_event(
                "preprocess_progress",
                idx=idx, total=total, name=src_rel, status="done",
                action=meta.get("action"),
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[fail] {src_rel}: {exc}")
            failed += 1
            emit_event(
                "preprocess_progress",
                idx=idx, total=total, name=src_rel, status="fail",
                error=str(exc)[:200],
                succeeded=succeeded, failed=failed, skipped=skipped,
            )

    log(f"[done] succeeded={succeeded} failed={failed} skipped={skipped}")
    return 0


def _run_crop_train(
    project: dict[str, Any],
    version: dict[str, Any],
    params: dict[str, Any],
    log: Callable[[str], None],
    emit_event: Callable[..., None],
) -> int:
    """ADR 0010 train-scope crop。

    `params['crops']` = `{rel_path: [rects]}`，rel_path 形如 `1_data/X.png`。
    crop 产物输出到同 folder 内：N=1 覆盖源文件名（`folder/stem.png`），
    N>1 fan-out 成 `folder/stem_c0.png` / `folder/stem_c1.png` / ...；
    train_replace_with_crops 原子替换 manifest。
    """
    project_dir = projects.project_dir(project["id"], project["slug"])
    train_dir = preprocess.version_train_dir(project, version["label"])
    train_dir.mkdir(parents=True, exist_ok=True)

    crops_param = params.get("crops") or {}
    if not crops_param:
        log("[done] crops 为空，无事可做")
        return 0
    sources = sorted(crops_param.keys())

    _last_emit_at = [0.0]

    def emit_throttled(*, force: bool, **payload) -> None:
        now = time.monotonic()
        if not force and (now - _last_emit_at[0]) < 1.0:
            return
        _last_emit_at[0] = now
        emit_event("crop_progress", **payload)

    total = len(sources)
    log(f"[start] stage=crop total={total} scope=train")

    succeeded = 0
    failed = 0
    skipped = 0

    for idx, src_rel in enumerate(sources, start=1):
        if _stop_requested:
            log(f"[cancel] 收到取消信号，已处理 {idx - 1}/{total}")
            break
        is_last = idx == total
        try:
            preprocess._validate_rel_name(src_rel)
        except preprocess.PreprocessError as exc:
            log(f"[skip] {src_rel}: {exc}")
            skipped += 1
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_rel, status="skip",
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
            continue

        src_path = train_dir / src_rel
        if not src_path.is_file():
            log(f"[skip] ({idx}/{total}) {src_rel}: 源不存在")
            skipped += 1
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_rel, status="skip",
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
            continue

        # origin 沿用 manifest 已有 entry root，否则用 src filename
        existing = preprocess_manifest.train_get_entry(
            project_dir, version["label"], src_rel
        )
        src_filename = src_rel.rsplit("/", 1)[-1]
        if existing is not None:
            origin = preprocess_manifest.entry_origin(existing, src_filename)
        else:
            origin = src_filename

        rects = crops_param[src_rel]
        n = len(rects)
        folder, _ = src_rel.split("/", 1)
        src_stem = Path(src_filename).stem
        out_rels = (
            [f"{folder}/{src_stem}.png"] if n == 1
            else [f"{folder}/{src_stem}_c{i}.png" for i in range(n)]
        )

        log(f"[crop] ({idx}/{total}) {src_rel} → {n} 个产物")
        try:
            t0 = time.monotonic()
            with Image.open(src_path) as raw:
                raw.load()
                src_img = raw.convert("RGB") if raw.mode != "RGB" else raw.copy()
            sw, sh = src_img.size
            outputs: list[dict[str, Any]] = []
            for r, out_rel in zip(rects, out_rels):
                left = int(round(r["x"] * sw))
                top = int(round(r["y"] * sh))
                right = int(round((r["x"] + r["w"]) * sw))
                bottom = int(round((r["y"] + r["h"]) * sh))
                right = max(left + 1, right)
                bottom = max(top + 1, bottom)
                piece = src_img.crop((left, top, right, bottom))
                out_path = train_dir / out_rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                piece.save(tmp_path, format="PNG", optimize=False)
                import os as _os
                _os.replace(tmp_path, out_path)
                try:
                    st = out_path.stat()
                    sz, mt = st.st_size, st.st_mtime
                except OSError:
                    sz, mt = 0, time.time()
                outputs.append({
                    "name": out_rel,
                    "origin": origin,
                    "size": sz,
                    "mtime": mt,
                })

            # N>1：原 src 物理文件（如果不在 outputs 列表里）应当删
            if n > 1:
                stale_rel = f"{folder}/{src_stem}.png"
                stale_path = train_dir / stale_rel
                if stale_path.is_file() and stale_rel not in out_rels:
                    try:
                        stale_path.unlink()
                    except OSError as exc:
                        log(f"   ⚠ 删旧 {stale_rel} 失败: {exc}")

            preprocess_manifest.train_replace_with_crops(
                project_dir, version["label"],
                source_name=src_rel,
                outputs=outputs,
            )
            # thumb prewarm
            try:
                from studio.services.dataset import thumb_cache
                for out_rel in out_rels:
                    out_path = train_dir / out_rel
                    with Image.open(out_path) as piece:
                        piece.load()
                        thumb_cache.prewarm_from_image(out_path, piece, [256, 768])
            except Exception as exc:  # noqa: BLE001
                log(f"   ⚠ thumb prewarm failed: {exc}")

            elapsed = time.monotonic() - t0
            succeeded += 1
            log(
                f"   ✓ {src_rel} → {', '.join(out_rels)}  "
                f"({sw}×{sh} → {n} 块, {elapsed:.2f}s)"
            )
            emit_throttled(
                force=(idx == 1 or is_last),
                idx=idx, total=total, name=src_rel, status="done",
                n_out=n, outputs=out_rels,
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[fail] {src_rel}: {exc}")
            failed += 1
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_rel, status="fail",
                error=str(exc)[:200],
                succeeded=succeeded, failed=failed, skipped=skipped,
            )

    log(f"[done] succeeded={succeeded} failed={failed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
