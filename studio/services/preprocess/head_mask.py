"""Automatic anime-head detection proposals and transactional mask application.

Detection is deliberately separated from application: the worker writes a proposal
snapshot under ``studio_data/tasks/<job>/head-mask/result.json`` and never touches
the source images or ``.mask`` sidecars.  Applying a reviewed proposal merges it
with existing masks using ``pixelwise min`` (255=learn, 0=ignore).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from studio.domain.errors import ConflictError, NotFoundError, ValidationError
from studio.infrastructure.paths import task_dir
from studio.services.models.paths import (
    HEAD_DETECTOR_REVISION,
    head_detector_target,
)
from studio.services.tagging.onnx_base import silenced_fd_stderr

from . import masks as train_masks

logger = logging.getLogger(__name__)

INPUT_SIZE = 640
DEFAULT_CONFIDENCE = 0.413
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_PADDING_RATIO = 0.10
DEFAULT_FEATHER_RATIO = 0.03
RESULT_SCHEMA_VERSION = 1


def result_path(job_id: int) -> Path:
    return task_dir(job_id) / "head-mask" / "result.json"


def apply_state_path(job_id: int) -> Path:
    return task_dir(job_id) / "head-mask" / "apply.json"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_result(job_id: int, value: dict[str, Any]) -> Path:
    path = result_path(job_id)
    _write_json_atomic(path, value)
    return path


def load_result(job_id: int) -> dict[str, Any]:
    path = result_path(job_id)
    if not path.is_file():
        raise NotFoundError(
            "Head-mask proposals are not ready",
            code="preprocess.head_mask_proposals_missing",
            details={"job_id": job_id},
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConflictError(
            "Head-mask proposal file is unreadable",
            code="preprocess.head_mask_proposals_invalid",
            details={"job_id": job_id},
        ) from exc
    if not isinstance(value, dict) or not isinstance(value.get("images"), list):
        raise ConflictError(
            "Head-mask proposal file has an invalid format",
            code="preprocess.head_mask_proposals_invalid",
            details={"job_id": job_id},
        )
    return value


def undo_available(job_id: int) -> bool:
    path = apply_state_path(job_id)
    if not path.is_file():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(state.get("records")) and not bool(state.get("undone"))


def source_snapshot(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"mtime_ns": int(st.st_mtime_ns), "file_size": int(st.st_size)}


def proposal_stale_reason(image: dict[str, Any], train_dir: Path) -> str | None:
    path = train_dir / str(image.get("name") or "")
    try:
        snap = source_snapshot(path)
    except OSError:
        return "missing"
    if int(image.get("source_mtime_ns") or -1) != snap["mtime_ns"]:
        return "mtime_changed"
    if int(image.get("source_file_size") or -1) != snap["file_size"]:
        return "content_size_changed"
    try:
        from PIL import Image

        with Image.open(path) as raw:
            size = raw.size
    except (OSError, ValueError):
        return "unreadable"
    expected = image.get("size") or []
    if list(size) != list(expected):
        return "dimensions_changed"
    return None


def result_with_staleness(result: dict[str, Any], train_dir: Path) -> dict[str, Any]:
    images = []
    stale_count = 0
    for item in result["images"]:
        reason = proposal_stale_reason(item, train_dir)
        stale = reason is not None
        stale_count += int(stale)
        images.append({**item, "stale": stale, "stale_reason": reason})
    return {**result, "images": images, "stale_count": stale_count}


def _letterbox(image: "Any") -> tuple[np.ndarray, float, int, int]:
    from PIL import Image

    width, height = image.size
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    left = (INPUT_SIZE - new_w) // 2
    top = (INPUT_SIZE - new_h) // 2
    canvas.paste(resized, (left, top))
    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(arr[None]), scale, left, top


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[_iou(boxes[current], boxes[rest]) <= threshold]
    return keep


def decode_output(
    output: np.ndarray,
    *,
    confidence: float,
    iou_threshold: float,
    scale: float,
    pad_left: int,
    pad_top: int,
    source_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Decode raw Ultralytics v8 output and restore source-image coordinates."""
    pred = np.asarray(output)
    while pred.ndim > 2 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2:
        raise RuntimeError(f"unexpected head detector output shape: {pred.shape}")
    if (
        pred.shape[0] == 5
        or (pred.shape[1] < 5 <= pred.shape[0])
        or (pred.shape[0] <= 16 and pred.shape[1] > pred.shape[0])
    ):
        pred = pred.T
    if pred.shape[1] < 5:
        raise RuntimeError(f"unexpected head detector output shape: {pred.shape}")

    # Raw one-class YOLOv8: xywh + class confidence. Some exporters bake NMS and
    # return xyxy + confidence + class; accept that layout as well.
    nms_export = False
    if pred.shape[1] == 6 and len(pred):
        valid_xyxy = np.mean(
            (pred[:, 2] > pred[:, 0]) & (pred[:, 3] > pred[:, 1])
        )
        nms_export = bool(valid_xyxy > 0.8 and np.max(pred[:, 5]) <= 10)
    if nms_export:
        boxes = pred[:, :4].astype(np.float32)
        scores = pred[:, 4].astype(np.float32)
    else:
        xywh = pred[:, :4].astype(np.float32)
        scores = np.max(pred[:, 4:], axis=1).astype(np.float32)
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    if boxes.size and float(np.max(np.abs(boxes))) <= 2.0:
        boxes *= INPUT_SIZE
    chosen = scores >= float(confidence)
    boxes, scores = boxes[chosen], scores[chosen]
    if not len(boxes):
        return []
    keep = nms(boxes, scores, float(iou_threshold))
    width, height = source_size
    result: list[dict[str, Any]] = []
    for idx in keep:
        raw = boxes[idx]
        x1 = max(0.0, min(float(width), (float(raw[0]) - pad_left) / scale))
        y1 = max(0.0, min(float(height), (float(raw[1]) - pad_top) / scale))
        x2 = max(0.0, min(float(width), (float(raw[2]) - pad_left) / scale))
        y2 = max(0.0, min(float(height), (float(raw[3]) - pad_top) / scale))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        result.append({
            "score": round(float(scores[idx]), 6),
            "box": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        })
    return result


class HeadDetector:
    """One ONNX session with CUDA -> DirectML -> CPU creation/inference fallback."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or head_detector_target()
        self.session: Any = None
        self.input_name = ""
        self.provider = ""
        self._create_session()

    def _create_session(self, *, cpu_only: bool = False) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed; repair it under Settings -> Environment"
            ) from exc
        available = list(ort.get_available_providers())
        accelerator = None
        if not cpu_only and "CUDAExecutionProvider" in available:
            accelerator = "CUDAExecutionProvider"
        elif not cpu_only and "DmlExecutionProvider" in available:
            accelerator = "DmlExecutionProvider"
        providers = (
            [accelerator, "CPUExecutionProvider"]
            if accelerator else ["CPUExecutionProvider"]
        )
        ctx = (
            silenced_fd_stderr()
            if accelerator == "CUDAExecutionProvider" else contextlib.nullcontext()
        )
        try:
            with ctx:
                self.session = ort.InferenceSession(
                    str(self.model_path), providers=providers
                )
        except Exception:
            if not accelerator:
                raise
            logger.warning(
                "Creating the %s head-detector session failed; retrying on CPU",
                accelerator,
                exc_info=True,
            )
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        self.input_name = self.session.get_inputs()[0].name
        actual = list(self.session.get_providers())
        self.provider = actual[0] if actual else "CPUExecutionProvider"

    def run(self, tensor: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(self.session.run(None, {self.input_name: tensor})[0])
        except Exception:
            if self.provider != "CUDAExecutionProvider":
                raise
            logger.warning("CUDA head detection failed; retrying this job on CPU")
            self._create_session(cpu_only=True)
            return np.asarray(self.session.run(None, {self.input_name: tensor})[0])

    def detect(
        self,
        image_path: Path,
        *,
        confidence: float,
        iou_threshold: float,
    ) -> tuple[tuple[int, int], list[dict[str, Any]]]:
        from PIL import Image

        with Image.open(image_path) as raw:
            raw.load()
            image = raw.convert("RGB")
        tensor, scale, left, top = _letterbox(image)
        detections = decode_output(
            self.run(tensor),
            confidence=confidence,
            iou_threshold=iou_threshold,
            scale=scale,
            pad_left=left,
            pad_top=top,
            source_size=image.size,
        )
        return image.size, detections


def expand_detection(
    detection: dict[str, Any],
    image_size: tuple[int, int],
    *,
    padding_ratio: float,
    feather_ratio: float,
) -> dict[str, Any]:
    width, height = image_size
    x1, y1, x2, y2 = (float(v) for v in detection["box"])
    box_w, box_h = x2 - x1, y2 - y1
    px, py = box_w * padding_ratio, box_h * padding_ratio
    left = max(0, int(np.floor(x1 - px)))
    top = max(0, int(np.floor(y1 - py)))
    right = min(width, int(np.ceil(x2 + px)))
    bottom = min(height, int(np.ceil(y2 + py)))
    feather_x = max(0, int(round(box_w * feather_ratio)))
    feather_y = max(0, int(round(box_h * feather_ratio)))
    identity = hashlib.sha1(
        f"{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}".encode()
    ).hexdigest()[:12]
    return {
        "id": identity,
        "score": detection["score"],
        "box": detection["box"],
        "mask_region": {
            "x1": left, "y1": top, "x2": right, "y2": bottom,
            "feather_x": feather_x, "feather_y": feather_y,
        },
    }


def make_image_proposal(
    name: str,
    path: Path,
    size: tuple[int, int],
    detections: Iterable[dict[str, Any]],
    *,
    padding_ratio: float,
    feather_ratio: float,
) -> dict[str, Any]:
    snap = source_snapshot(path)
    regions = [
        expand_detection(
            det, size,
            padding_ratio=padding_ratio,
            feather_ratio=feather_ratio,
        )
        for det in detections
    ]
    # Equal boxes are unusual but possible after export rounding; keep IDs unique.
    for index, region in enumerate(regions):
        region["id"] = f"{index}-{region['id']}"
    return {
        "name": name,
        "size": [int(size[0]), int(size[1])],
        "source_mtime_ns": snap["mtime_ns"],
        "source_file_size": snap["file_size"],
        "regions": regions,
    }


def render_auto_mask(
    size: tuple[int, int], regions: Iterable[dict[str, Any]],
) -> np.ndarray:
    """Render selected rectangles as grayscale loss weights (255 learn, 0 ignore)."""
    width, height = size
    out = np.full((height, width), 255, dtype=np.uint8)
    for proposal in regions:
        region = proposal["mask_region"]
        x1 = max(0, min(width, int(region["x1"])))
        y1 = max(0, min(height, int(region["y1"])))
        x2 = max(x1, min(width, int(region["x2"])))
        y2 = max(y1, min(height, int(region["y2"])))
        fx = max(0, int(region.get("feather_x") or 0))
        fy = max(0, int(region.get("feather_y") or 0))
        if x2 <= x1 or y2 <= y1:
            continue
        out[y1:y2, x1:x2] = 0
        if not fx and not fy:
            continue
        ox1, oy1 = max(0, x1 - fx), max(0, y1 - fy)
        ox2, oy2 = min(width, x2 + fx), min(height, y2 + fy)
        yy, xx = np.ogrid[oy1:oy2, ox1:ox2]
        dx = np.maximum(np.maximum(x1 - xx, xx - (x2 - 1)), 0)
        dy = np.maximum(np.maximum(y1 - yy, yy - (y2 - 1)), 0)
        nx = dx / max(fx, 1) if fx else np.where(dx > 0, 1.0, 0.0)
        ny = dy / max(fy, 1) if fy else np.where(dy > 0, 1.0, 0.0)
        feather = np.clip(np.maximum(nx, ny), 0.0, 1.0)
        values = np.rint(feather * 255).astype(np.uint8)
        out[oy1:oy2, ox1:ox2] = np.minimum(out[oy1:oy2, ox1:ox2], values)
    return out


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_existing_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if not path.is_file():
        return np.full((size[1], size[0]), 255, dtype=np.uint8)
    try:
        with Image.open(path) as raw:
            raw.load()
            if raw.size != size:
                raise ValueError(f"mask size {raw.size} != image size {size}")
            return np.asarray(raw.convert("L"), dtype=np.uint8).copy()
    except (OSError, ValueError) as exc:
        raise ConflictError(
            "Existing training mask is unreadable or has the wrong dimensions",
            code="preprocess.head_mask_existing_mask_invalid",
            details={"path": str(path), "reason": str(exc)},
        ) from exc


def _restore_records(records: list[dict[str, Any]], train_dir: Path, backup_dir: Path) -> None:
    for record in records:
        target = train_masks.mask_path_for(train_dir, record["name"])
        if record["before_exists"]:
            source = backup_dir / record["backup_rel"]
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".rollback")
            shutil.copy2(source, tmp)
            os.replace(tmp, target)
        else:
            target.unlink(missing_ok=True)


def apply_proposals(
    job_id: int,
    train_dir: Path,
    selections: dict[str, list[str]],
) -> dict[str, Any]:
    """Validate every proposal, then atomically merge reviewed regions into masks."""
    from PIL import Image

    result = load_result(job_id)
    by_name = {str(item["name"]): item for item in result["images"]}
    unknown_names = sorted(set(selections) - set(by_name))
    if unknown_names:
        raise ValidationError(
            "Selection contains images outside this proposal",
            code="preprocess.head_mask_selection_invalid",
            details={"names": unknown_names}, http_status=400,
        )
    stale = [
        {"name": item["name"], "reason": reason}
        for item in result["images"]
        if (reason := proposal_stale_reason(item, train_dir)) is not None
    ]
    if stale:
        raise ConflictError(
            "One or more images changed after detection; run detection again",
            code="preprocess.head_mask_proposals_stale",
            details={"images": stale},
        )

    selected: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for name, ids in selections.items():
        item = by_name[name]
        regions = {str(region["id"]): region for region in item["regions"]}
        unknown_ids = sorted(set(ids) - set(regions))
        if unknown_ids:
            raise ValidationError(
                "Selection contains unknown head regions",
                code="preprocess.head_mask_selection_invalid",
                details={"name": name, "region_ids": unknown_ids}, http_status=400,
            )
        chosen = [regions[region_id] for region_id in dict.fromkeys(ids)]
        if chosen:
            selected.append((item, chosen))

    apply_id = str(time.time_ns())
    root = task_dir(job_id) / "head-mask"
    staging = root / "staging" / apply_id
    backup_dir = root / "undo" / apply_id
    staging.mkdir(parents=True, exist_ok=False)
    backup_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        for item, regions in selected:
            name = str(item["name"])
            size = (int(item["size"][0]), int(item["size"][1]))
            mask_path = train_masks.mask_path_for(train_dir, name)
            existing = _load_existing_mask(mask_path, size)
            merged = np.minimum(existing, render_auto_mask(size, regions))
            if np.array_equal(existing, merged) and mask_path.is_file():
                continue
            rel = f"{Path(name).parent.as_posix()}/{Path(name).stem}.mask"
            backup_rel = rel + ".before"
            before_exists = mask_path.is_file()
            if before_exists:
                backup = backup_dir / backup_rel
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mask_path, backup)
            staged = staging / rel
            staged.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(merged, mode="L").save(staged, format="PNG", optimize=False)
            records.append({
                "name": name,
                "staged_rel": rel,
                "backup_rel": backup_rel,
                "before_exists": before_exists,
                "selected_region_ids": [region["id"] for region in regions],
            })

        committed: list[dict[str, Any]] = []
        try:
            for record in records:
                source = staging / record["staged_rel"]
                target = train_masks.mask_path_for(train_dir, record["name"])
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                record["applied_sha256"] = _file_sha256(target)
                committed.append(record)
            state = {
                "schema_version": 1,
                "job_id": job_id,
                "apply_id": apply_id,
                "applied_at": time.time(),
                "backup_dir": str(backup_dir),
                "records": records,
                "undone": False,
            }
            _write_json_atomic(apply_state_path(job_id), state)
        except Exception:
            _restore_records(committed, train_dir, backup_dir)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "job_id": job_id,
        "applied": len(records),
        "images": [record["name"] for record in records],
        "undo_available": bool(records),
    }


def undo_apply(job_id: int, train_dir: Path) -> dict[str, Any]:
    path = apply_state_path(job_id)
    if not path.is_file():
        raise NotFoundError(
            "No automatic head-mask application can be undone",
            code="preprocess.head_mask_undo_missing",
            details={"job_id": job_id},
        )
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConflictError(
            "Automatic head-mask undo data is unreadable",
            code="preprocess.head_mask_undo_invalid",
            details={"job_id": job_id},
        ) from exc
    if state.get("undone"):
        raise ConflictError(
            "This automatic head-mask application was already undone",
            code="preprocess.head_mask_already_undone",
            details={"job_id": job_id},
        )
    records = list(state.get("records") or [])
    changed = []
    for record in records:
        current = train_masks.mask_path_for(train_dir, record["name"])
        if _file_sha256(current) != record.get("applied_sha256"):
            changed.append(record["name"])
    if changed:
        raise ConflictError(
            "Masks were edited after automatic masking; undo was refused",
            code="preprocess.head_mask_undo_modified",
            details={"images": changed},
        )

    backup_dir = Path(str(state["backup_dir"]))
    safety_dir = task_dir(job_id) / "head-mask" / "undo-staging" / str(time.time_ns())
    safety_dir.mkdir(parents=True, exist_ok=False)
    safety_records: list[dict[str, Any]] = []
    try:
        for record in records:
            current = train_masks.mask_path_for(train_dir, record["name"])
            if current.is_file():
                rel = record["staged_rel"] + ".current"
                saved = safety_dir / rel
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current, saved)
                safety_records.append({**record, "backup_rel": rel, "before_exists": True})
            else:
                safety_records.append({**record, "before_exists": False})
        try:
            _restore_records(records, train_dir, backup_dir)
            state["undone"] = True
            state["undone_at"] = time.time()
            _write_json_atomic(path, state)
        except Exception:
            _restore_records(safety_records, train_dir, safety_dir)
            raise
    finally:
        shutil.rmtree(safety_dir, ignore_errors=True)
    return {
        "job_id": job_id,
        "undone": len(records),
        "images": [record["name"] for record in records],
    }


def new_result(
    job_id: int,
    *,
    confidence: float,
    iou_threshold: float,
    padding_ratio: float,
    feather_ratio: float,
    provider: str,
    images: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "job_id": job_id,
        "model": {
            "revision": HEAD_DETECTOR_REVISION,
            "path": str(head_detector_target()),
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "provider": provider,
        },
        "parameters": {
            "confidence": confidence,
            "iou_threshold": iou_threshold,
            "padding_ratio": padding_ratio,
            "feather_ratio": feather_ratio,
        },
        "created_at": time.time(),
        "images": images,
    }
