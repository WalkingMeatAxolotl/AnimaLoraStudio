"""预处理 worker 子进程入口（放大 + 裁剪）。

由 supervisor 启动：`python -m studio.workers.preprocess_worker --job-id N`。

读 project_jobs 行 → 按 `params['stage']` 分发：
  - stage='upscale' (默认)：串行调 `studio.services.upscaler.upscale_file()`
  - stage='crop'：用 PIL 把 preprocess/ 下的图按归一化 rect 切成 N 张产物

日志规范：只走 logger（supervisor 把 stdout/stderr 合流重定向到 log 文件），
不要再 open 同一个 log 文件，避免 LogTailer 读两次。唯一裸 print 是
`emit_event` 的 `__EVENT__:` stdout 协议行。

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

# 固定名：worker 经 `python -m studio.workers.preprocess_worker` 拉起时 __name__ 是 __main__，
# 行契约里的来源列会失真、也不在 OWN_LOGGER_NAMESPACES 里。
logger = logging.getLogger("studio.workers.preprocess_worker")

from studio.infrastructure.log_messages import msg
from studio.infrastructure.task_log import TaskLog, TaskLogLike
from studio import db
from studio.domain.errors import DomainError
from studio.services.preprocess import core as preprocess
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services import models as model_downloader
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.preprocess import masks as train_masks
from studio.services.preprocess import head_mask
from studio.services.inference import upscaler

from utils.log_throttle import ProgressThrottle, RepeatThrottle


_stop_requested = False


def _on_signal(_signum, _frame) -> None:  # pragma: no cover - signal path
    global _stop_requested
    _stop_requested = True


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGBREAK"):  # Windows
        signal.signal(signal.SIGBREAK, _on_signal)  # type: ignore[attr-defined]


def _unlink_image_and_sidecars(path: Path, *, keep_sidecars: bool = False) -> None:
    """Remove an image and caption sidecars that are no longer part of train/."""
    if path.is_file():
        path.unlink(missing_ok=True)
    if keep_sidecars:
        return
    for ext in (".txt", ".json"):
        path.with_suffix(ext).unlink(missing_ok=True)


def run(job_id: int) -> int:  # noqa: PLR0912, PLR0915 - 主流程线性可读
    _install_signal_handlers()

    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        logger.error("Preprocess job %s not found in the database; nothing to run", job_id)
        return 1
    if job["kind"] != preprocess.PREPROCESS_KIND:
        logger.error(
            "Internal error: job %s has kind=%s, not a preprocess job; aborting",
            job_id, job["kind"],
        )
        return 1

    params = job.get("params_decoded") or {}
    # 缺 stage 字段视为老 upscale job（向后兼容）
    stage = params.get("stage", preprocess.STAGE_UPSCALE)

    log = TaskLog(logger)

    def emit_event(evt_type: str, **payload) -> None:
        """通过 stdout 标记行 → supervisor 解析 → SSE。供前端实时更新用，
        不会进 job 日志。supervisor 端常量见 `studio/supervisor.py:_EVENT_MARKER`。

        这里**必须**是裸 print（stdout 协议行白名单）：走 logger 会被 Human
        formatter 加前缀，supervisor 就认不出 `__EVENT__:` 行头了。"""
        try:
            print(f"__EVENT__:{evt_type}:{json.dumps(payload, ensure_ascii=False)}", flush=True)
        except Exception:  # noqa: BLE001 — 推事件失败不影响主流程
            pass

    try:
        with db.connection_for() as conn:
            project = projects.get_project(conn, job["project_id"])
        if not project:
            log.error(
                "Project %s no longer exists; preprocessing aborted",
                job["project_id"],
            )
            return 1

        version_id = job.get("version_id")
        if version_id is None:
            log.error(
                "Internal error: the preprocess job does not say which version "
                "to work on; aborting"
            )
            return 1
        with db.connection_for() as conn:
            version = versions.get_version(conn, version_id)
        if not version:
            log.error("Version %s no longer exists; preprocessing aborted", version_id)
            return 1
        if stage == preprocess.STAGE_CROP:
            return _run_crop_train(project, version, params, log, emit_event)
        if stage == preprocess.STAGE_HEAD_MASK:
            return _run_head_mask_train(
                job_id, project, version, params, log, emit_event,
            )
        if stage == preprocess.STAGE_UPSCALE:
            return _run_upscale_train(
                project, version, params, log, emit_event,
            )
        log.error("Internal error: unknown preprocess stage %r; aborting", stage)
        return 1
    except Exception:  # noqa: BLE001
        # PR-1 C7: logger.exception 带 trace_id 进 stderr；异常摘要由 traceback
        # 提供，不再另发一条 log 行（C6）。
        logger.exception("Preprocess worker crashed: job=%s", job_id)
        return 1


def _run_head_mask_train(
    job_id: int,
    project: dict[str, Any],
    version: dict[str, Any],
    params: dict[str, Any],
    log: TaskLogLike,
    emit_event: Callable[..., None],
) -> int:
    """Detect every cartoon head and persist reviewable proposals only."""
    scope = str(params.get("scope") or "all")
    names = params.get("names") or None
    confidence = float(params.get("confidence", head_mask.DEFAULT_CONFIDENCE))
    iou_threshold = float(
        params.get("iou_threshold", head_mask.DEFAULT_IOU_THRESHOLD)
    )
    padding_ratio = float(
        params.get("padding_ratio", head_mask.DEFAULT_PADDING_RATIO)
    )
    feather_ratio = float(
        params.get("feather_ratio", head_mask.DEFAULT_FEATHER_RATIO)
    )
    try:
        sources = preprocess.resolve_targets_train(
            project, version["label"], mode=scope, names=names,
        )
    except DomainError as exc:
        log.error("Resolving head-mask images failed: %s", exc)
        return 1
    try:
        status = model_downloader.head_detector_status()
        if not status.get("valid"):
            log.error(
                "Anime head detector is missing or damaged; download it under "
                "Settings -> Preprocess before retrying"
            )
            return 1
        detector = head_mask.HeadDetector(model_downloader.head_detector_target())
    except Exception as exc:  # noqa: BLE001
        log.error("Loading the anime head detector failed: %s", exc)
        return 1

    train_dir = preprocess.version_train_dir(project, version["label"])
    proposals: list[dict[str, Any]] = []
    succeeded = failed = skipped = 0
    total = len(sources)
    log.info(
        "Head detection started: images=%d confidence=%.3f iou=%.3f "
        "padding=%.3f feather=%.3f provider=%s",
        total, confidence, iou_threshold, padding_ratio, feather_ratio,
        detector.provider,
    )
    for idx, name in enumerate(sources, start=1):
        if _stop_requested:
            log.warning(
                "Head detection canceled after %d/%d images; no proposal was saved",
                idx - 1, total,
            )
            return 130
        path = train_dir / name
        if not path.is_file():
            skipped += 1
            emit_event(
                "head_mask_progress", idx=idx, total=total, name=name,
                status="skip", succeeded=succeeded, failed=failed, skipped=skipped,
            )
            continue
        try:
            size, detections = detector.detect(
                path, confidence=confidence, iou_threshold=iou_threshold,
            )
            proposal = head_mask.make_image_proposal(
                name, path, size, detections,
                padding_ratio=padding_ratio,
                feather_ratio=feather_ratio,
            )
            proposals.append(proposal)
            succeeded += 1
            emit_event(
                "head_mask_progress", idx=idx, total=total, name=name,
                status="done", detections=len(detections),
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log.warning("Head detection failed for %s: %s", name, exc)
            emit_event(
                "head_mask_progress", idx=idx, total=total, name=name,
                status="fail", error=str(exc)[:200],
                succeeded=succeeded, failed=failed, skipped=skipped,
            )

    result = head_mask.new_result(
        job_id,
        confidence=confidence,
        iou_threshold=iou_threshold,
        padding_ratio=padding_ratio,
        feather_ratio=feather_ratio,
        provider=detector.provider,
        images=proposals,
    )
    head_mask.write_result(job_id, result)
    heads = sum(len(image["regions"]) for image in proposals)
    log.info(
        "Head detection proposal ready: images=%d heads=%d failed=%d skipped=%d",
        succeeded, heads, failed, skipped,
    )
    return 0


def _run_upscale_train(
    project: dict[str, Any],
    version: dict[str, Any],
    params: dict[str, Any],
    log: TaskLogLike,
    emit_event: Callable[..., None],
) -> int:
    """ADR 0010 train-scope upscale。

    源 + 产物都在 `versions/{label}/train/{folder}/`，manifest 写到
    `versions/{label}/train/manifest.json`。ADR 0010 fixup（2026-06-04）：
    **不改扩展名** —— 同名 in-place 覆盖（X.jpg → X.jpg / X.png → X.png），
    避免 caption 对应关系断裂 / dataset_config 扩展名 glob 失效。upscaler
    按 src 扩展名 save（JPEG quality=95 / PNG 无压缩 / WebP quality=95）；
    manifest entry 加 `processed=True` 标记给 UI 推断徽章用。
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
        log.error(
            "Upscaler weights not found: %s; download %s on the settings page first",
            model_path, model_label,
        )
        return 1

    try:
        sources = preprocess.resolve_targets_train(
            project, version["label"], mode=mode, names=names
        )
    except DomainError as exc:
        log.error(
            "Resolving which images to process failed: %s; preprocessing aborted",
            exc,
        )
        return 1

    total = len(sources)
    if total == 0:
        log.info(msg("worker.preprocess.no_images"))
        return 0

    target_desc = (
        f"{int(math.sqrt(target_area))}²={target_area}px"
        if target_area else "off (直接 4×)"
    )
    log.info(msg(
        "worker.preprocess.upscale_start",
        mode=mode, model=model_label, tile=tile_size, pad=tile_pad,
        device=device, target=target_desc, total=total,
    ))

    try:
        import torch
        resolved_dev = upscaler.resolve_device(device)
        resolved_dtype = upscaler.resolve_dtype("auto", resolved_dev)
        cuda_available = torch.cuda.is_available()
        gpu_name = (
            torch.cuda.get_device_name(0)
            if resolved_dev.type == "cuda" and cuda_available
            else "—"
        )
        log.debug(
            "device: resolved=%s dtype=%s gpu=%s cuda_available=%s",
            resolved_dev, str(resolved_dtype).replace("torch.", ""),
            gpu_name, cuda_available,
        )
        if str(device).startswith("cuda") and not cuda_available:
            log.warning(
                "CUDA was requested but is not available; upscaling runs on the "
                "CPU and will be roughly 10× slower"
            )
        upscaler.load_model(model_path, device=resolved_dev, dtype=resolved_dtype)
        log.info(msg(
            "worker.preprocess.model_ready", model=model_label, device=resolved_dev,
        ))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Device diagnostics failed: %s; upscaling continues but may run on "
            "the CPU (roughly 10× slower)", exc,
        )

    succeeded = 0
    failed = 0
    skipped = 0
    repeat = RepeatThrottle(log)
    # 逐图行降 DEBUG，可见进度由节流后的计数 INFO 承担（Q3 三件套）。
    throttle = ProgressThrottle(total)

    for idx, src_rel in enumerate(sources, start=1):
        if _stop_requested:
            log.warning(
                "Canceled by the user after %d/%d images; the rest were left "
                "unprocessed", idx - 1, total,
            )
            break
        src_path = train_dir / src_rel
        if not src_path.exists():
            repeat.hit(
                "source_gone",
                "%d images skipped: the source file was gone (first: %s)",
                "Image %d/%d skipped: %s no longer exists",
                idx, total, src_rel,
                first=src_rel,
            )
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

        # ADR 0010 fixup：dst == src，in-place 覆盖。upscaler 按 src 扩展名
        # save（JPEG 95 / WebP 95 / PNG 无压缩），保 caption + dataset_config 对
        # 扩展名的依赖；manifest entry 加 processed=True 标记。
        dst_path = src_path
        src_ext = Path(src_filename).suffix.lower()
        if src_ext in (".jpg", ".jpeg"):
            save_kwargs: dict[str, Any] = {"format": "JPEG", "quality": 95}
        elif src_ext == ".webp":
            save_kwargs = {"format": "WEBP", "quality": 95, "method": 6}
        else:
            save_kwargs = {"format": "PNG", "optimize": False}

        log.debug("upscale: %d/%d %s", idx, total, src_rel)
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
                save_kwargs=save_kwargs,
            )
            meta["origin"] = origin_name
            meta["processed"] = True
            preprocess_manifest.train_add_processed(
                project_dir, version["label"], src_rel, meta,
            )
            # mask sidecar 跟随：NEAREST resize 到放大后尺寸（无 mask 时 no-op）
            try:
                with Image.open(dst_path) as up_img:
                    train_masks.resize_mask_like(train_dir, src_rel, up_img.size)
            except Exception as exc:  # noqa: BLE001
                repeat.hit(
                    "mask_resize_failed",
                    "%d masks could not be resized (first: %s); those images are "
                    "upscaled but their masks are stale",
                    "Resizing the mask for %s failed: %s; the image was upscaled "
                    "but its mask still has the old size",
                    src_rel, exc,
                    first=src_rel,
                )
            succeeded += 1
            if throttle.should_emit(idx):
                log.info(msg("worker.preprocess.progress", done=idx, total=total))
            emit_event(
                "preprocess_progress",
                idx=idx, total=total, name=src_rel, status="done",
                action=meta.get("action"),
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
        except Exception as exc:  # noqa: BLE001
            repeat.hit(
                "upscale_failed",
                "%d images failed to upscale (first: %s)",
                "Upscaling %s failed: %s; image left unchanged",
                src_rel, exc,
                first=src_rel,
            )
            failed += 1
            emit_event(
                "preprocess_progress",
                idx=idx, total=total, name=src_rel, status="fail",
                error=str(exc)[:200],
                succeeded=succeeded, failed=failed, skipped=skipped,
            )

    repeat.drain()
    if failed:
        log.warning(
            "Upscaling finished with failures: succeeded=%d failed=%d skipped=%d",
            succeeded, failed, skipped,
        )
    else:
        log.info(msg(
            "worker.preprocess.upscale_done",
            succeeded=succeeded, failed=failed, skipped=skipped,
        ))
    return 0


def _run_crop_train(
    project: dict[str, Any],
    version: dict[str, Any],
    params: dict[str, Any],
    log: TaskLogLike,
    emit_event: Callable[..., None],
) -> int:
    """ADR 0010 train-scope crop。

    `params['crops']` = `{rel_path: [rects]}`，rel_path 形如 `1_data/X.png`。
    crop 产物输出到同 folder 内：N=1 产出 `folder/stem.png`，N>1 fan-out
    成 `folder/stem_c0.png` / `folder/stem_c1.png` / ...；成功后清理不再
    属于 outputs 的旧源图，再用 train_replace_with_crops 原子替换 manifest。
    """
    project_dir = projects.project_dir(project["id"], project["slug"])
    train_dir = preprocess.version_train_dir(project, version["label"])
    train_dir.mkdir(parents=True, exist_ok=True)

    crops_param = params.get("crops") or {}
    if not crops_param:
        log.info(msg("worker.preprocess.no_crops"))
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
    log.info(msg("worker.preprocess.crop_start", total=total))

    succeeded = 0
    failed = 0
    skipped = 0
    repeat = RepeatThrottle(log)
    # 逐图行降 DEBUG，可见进度由节流后的计数 INFO 承担（Q3 三件套）。
    throttle = ProgressThrottle(total)

    for idx, src_rel in enumerate(sources, start=1):
        if _stop_requested:
            log.warning(
                "Canceled by the user after %d/%d images; the rest were left "
                "unprocessed", idx - 1, total,
            )
            break
        is_last = idx == total
        try:
            preprocess._validate_rel_name(src_rel)
        except DomainError as exc:
            repeat.hit(
                "invalid_name",
                "%d images skipped: invalid file name (first: %s)",
                "Image %s skipped: invalid file name (%s)",
                src_rel, exc,
                first=src_rel,
            )
            skipped += 1
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_rel, status="skip",
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
            continue

        src_path = train_dir / src_rel
        if not src_path.is_file():
            repeat.hit(
                "source_gone",
                "%d images skipped: the source file was gone (first: %s)",
                "Image %d/%d skipped: %s no longer exists",
                idx, total, src_rel,
                first=src_rel,
            )
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

        try:
            t0 = time.monotonic()
            with Image.open(src_path) as raw:
                raw.load()
                src_img = raw.convert("RGB") if raw.mode != "RGB" else raw.copy()
            sw, sh = src_img.size
            outputs: list[dict[str, Any]] = []
            crop_boxes: list[tuple[int, int, int, int]] = []
            for r, out_rel in zip(rects, out_rels):
                left = int(round(r["x"] * sw))
                top = int(round(r["y"] * sh))
                right = int(round((r["x"] + r["w"]) * sw))
                bottom = int(round((r["y"] + r["h"]) * sh))
                right = max(left + 1, right)
                bottom = max(top + 1, bottom)
                crop_boxes.append((left, top, right, bottom))
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

            # 输出可能换成 .png 或 fan-out 成多张；源不再属于 outputs 时必须删，
            # 否则从 bundle/版本复制来的 train-only 数据会同时保留原图 + 裁剪图。
            # 已知边界（沿袭旧行为）：`{stem}.png` 进 stale 集是为了清掉历史
            # N=1 crop 的产物；若 train 里恰好有同 stem 的两张独立图
            # （X.jpg + X.png），对 X.jpg fan-out 会把无关的 X.png 一并删掉。
            stale_rels = {src_rel, f"{folder}/{src_stem}.png"} - set(out_rels)
            output_stems = {Path(rel).stem for rel in out_rels}
            for stale_rel in sorted(stale_rels):
                stale_path = train_dir / stale_rel
                has_sidecar = (
                    stale_path.with_suffix(".txt").exists()
                    or stale_path.with_suffix(".json").exists()
                )
                if stale_path.exists() or has_sidecar:
                    try:
                        _unlink_image_and_sidecars(
                            stale_path,
                            keep_sidecars=Path(stale_rel).stem in output_stems,
                        )
                    except OSError as exc:
                        repeat.hit(
                            "stale_unlink_failed",
                            "%d superseded files could not be removed (first: %s); "
                            "they stay in the train folder",
                            "Removing the superseded file %s failed: %s; it stays "
                            "in the train folder",
                            stale_rel, exc,
                            first=stale_rel,
                        )

            # mask sidecar 跟随：同 box 裁剪 + fan-out（源无 mask 时 no-op）
            try:
                train_masks.crop_mask_like(
                    train_dir, src_rel, crop_boxes, out_rels,
                )
            except Exception as exc:  # noqa: BLE001
                repeat.hit(
                    "mask_crop_failed",
                    "%d masks could not be cropped (first: %s); those crops have "
                    "no mask",
                    "Cropping the mask for %s failed: %s; the crops have no mask",
                    src_rel, exc,
                    first=src_rel,
                )

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
                log.debug("thumb prewarm failed for %s: %s", src_rel, exc)

            elapsed = time.monotonic() - t0
            succeeded += 1
            log.debug(
                "crop: %d/%d %s → %s (%dx%d, %d pieces, %.1fs)",
                idx, total, src_rel, ", ".join(out_rels), sw, sh, n, elapsed,
            )
            if throttle.should_emit(idx):
                log.info(msg("worker.preprocess.progress", done=idx, total=total))
            emit_throttled(
                force=(idx == 1 or is_last),
                idx=idx, total=total, name=src_rel, status="done",
                n_out=n, outputs=out_rels,
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
        except Exception as exc:  # noqa: BLE001
            repeat.hit(
                "crop_failed",
                "%d images failed to crop (first: %s)",
                "Cropping %s failed: %s; image left unchanged",
                src_rel, exc,
                first=src_rel,
            )
            failed += 1
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_rel, status="fail",
                error=str(exc)[:200],
                succeeded=succeeded, failed=failed, skipped=skipped,
            )

    repeat.drain()
    if failed:
        log.warning(
            "Cropping finished with failures: succeeded=%d failed=%d skipped=%d",
            succeeded, failed, skipped,
        )
    else:
        log.info(msg(
            "worker.preprocess.crop_done",
            succeeded=succeeded, failed=failed, skipped=skipped,
        ))
    return 0


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
