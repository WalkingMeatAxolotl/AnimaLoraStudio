"""Primary training-region sidecars (``{stem}.regions.json``).

The sidecar stores a single normalized rectangle plus the small amount of
captioning metadata needed by the region editor.  Coordinates are normalized
so ordinary resize operations only need to refresh ``image_size``; crop
operations intersect and re-normalize the rectangle explicitly.

This is deliberately separate from ``{stem}.mask``.  A mask says where the
trainer must *not* learn, while a region says where the early training phase
should learn more strongly.  Conflating the two makes it impossible to anneal
back to whole-image training without changing the user's ignore mask.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from studio.domain.errors import ValidationError


REGION_SUFFIX = ".regions.json"
FORMAT_VERSION = 1


def region_path_for(train_dir: Path, rel_name: str) -> Path:
    image = Path(rel_name)
    return train_dir / image.parent / f"{image.stem}{REGION_SUFFIX}"


def _clean_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit]


def normalize_document(
    raw: Any,
    *,
    expected_size: tuple[int, int] | None = None,
    name: str = "",
) -> dict[str, Any]:
    """Validate and normalize a v1 single-primary-region document."""
    if not isinstance(raw, dict):
        raise ValidationError(
            "Region annotation must be a JSON object",
            code="preprocess.region_invalid",
            details={"name": name}, http_status=400,
        )
    try:
        version = int(raw.get("version", FORMAT_VERSION) or FORMAT_VERSION)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "Region annotation version must be an integer",
            code="preprocess.region_version_invalid",
            details={"name": name}, http_status=400,
        ) from exc
    if version != FORMAT_VERSION:
        raise ValidationError(
            f"Unsupported region annotation version: {version}",
            code="preprocess.region_version_unsupported",
            details={"name": name, "version": version}, http_status=400,
        )
    regions = raw.get("regions")
    if not isinstance(regions, list) or len(regions) != 1:
        raise ValidationError(
            "Exactly one primary region is required",
            code="preprocess.region_primary_required",
            details={"name": name}, http_status=400,
        )
    region = regions[0]
    if not isinstance(region, dict):
        raise ValidationError(
            "Primary region must be an object",
            code="preprocess.region_invalid",
            details={"name": name}, http_status=400,
        )
    box = region.get("box")
    if not isinstance(box, dict):
        raise ValidationError(
            "Primary region box is required",
            code="preprocess.region_box_required",
            details={"name": name}, http_status=400,
        )
    try:
        x = float(box["x"])
        y = float(box["y"])
        w = float(box["w"])
        h = float(box["h"])
        weight = float(region.get("weight", 1.0) or 1.0)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(
            "Region box and weight must be numeric",
            code="preprocess.region_numeric_invalid",
            details={"name": name}, http_status=400,
        ) from exc
    if (
        x < 0 or y < 0 or w <= 0 or h <= 0
        or x >= 1 or y >= 1 or x + w > 1.000001 or y + h > 1.000001
    ):
        raise ValidationError(
            "Region box must fit inside normalized image coordinates",
            code="preprocess.region_box_out_of_bounds",
            details={"name": name, "box": {"x": x, "y": y, "w": w, "h": h}},
            http_status=400,
        )
    if not (0.1 <= weight <= 10.0):
        raise ValidationError(
            "Region weight must be between 0.1 and 10",
            code="preprocess.region_weight_out_of_range",
            details={"name": name, "weight": weight}, http_status=400,
        )

    if expected_size is not None:
        iw, ih = int(expected_size[0]), int(expected_size[1])
    else:
        size = raw.get("image_size") or {}
        try:
            iw, ih = int(size["w"]), int(size["h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                "Region annotation image_size is invalid",
                code="preprocess.region_image_size_invalid",
                details={"name": name}, http_status=400,
            ) from exc
    if iw <= 0 or ih <= 0:
        raise ValidationError(
            "Region annotation image_size must be positive",
            code="preprocess.region_image_size_invalid",
            details={"name": name}, http_status=400,
        )

    return {
        "version": FORMAT_VERSION,
        "image_size": {"w": iw, "h": ih},
        "regions": [{
            "id": "primary",
            "label": _clean_text(region.get("label") or "primary", limit=80),
            "class_word": _clean_text(region.get("class_word"), limit=80),
            "caption": _clean_text(region.get("caption"), limit=1000),
            "weight": weight,
            "box": {"x": x, "y": y, "w": w, "h": h},
        }],
    }


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_region(
    train_dir: Path,
    rel_name: str,
    raw: Any,
    *,
    expected_size: tuple[int, int],
) -> dict[str, Any]:
    document = normalize_document(raw, expected_size=expected_size, name=rel_name)
    path = region_path_for(train_dir, rel_name)
    _atomic_write(path, document)
    st = path.stat()
    return {"name": rel_name, "mtime": st.st_mtime, "size": st.st_size, **document}


def read_region(train_dir: Path, rel_name: str) -> Optional[dict[str, Any]]:
    path = region_path_for(train_dir, rel_name)
    if not path.is_file():
        return None
    try:
        return normalize_document(
            json.loads(path.read_text(encoding="utf-8-sig")), name=rel_name,
        )
    except (OSError, json.JSONDecodeError, ValidationError):
        return None


def delete_region(train_dir: Path, rel_name: str) -> bool:
    path = region_path_for(train_dir, rel_name)
    if not path.is_file():
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def delete_regions_for(train_dir: Path, rel_names: Iterable[str]) -> int:
    return sum(1 for name in rel_names if delete_region(train_dir, name))


def region_stat(train_dir: Path, rel_name: str) -> Optional[dict[str, Any]]:
    path = region_path_for(train_dir, rel_name)
    try:
        st = path.stat()
    except OSError:
        return None
    return {"mtime": st.st_mtime, "size": st.st_size}


def resize_region_like(
    train_dir: Path, rel_name: str, size: tuple[int, int],
) -> None:
    document = read_region(train_dir, rel_name)
    if document is None:
        return
    document["image_size"] = {"w": int(size[0]), "h": int(size[1])}
    _atomic_write(region_path_for(train_dir, rel_name), document)


def crop_region_like(
    train_dir: Path,
    src_rel: str,
    boxes: list[tuple[int, int, int, int]],
    out_rels: list[str],
) -> None:
    """Intersect the primary rectangle with each crop and re-normalize it."""
    document = read_region(train_dir, src_rel)
    if document is None:
        return
    size = document["image_size"]
    iw, ih = int(size["w"]), int(size["h"])
    region = document["regions"][0]
    b = region["box"]
    rx0, ry0 = b["x"] * iw, b["y"] * ih
    rx1, ry1 = (b["x"] + b["w"]) * iw, (b["y"] + b["h"]) * ih

    written: list[Path] = []
    for crop, out_rel in zip(boxes, out_rels):
        cx0, cy0, cx1, cy1 = map(float, crop)
        cw, ch = cx1 - cx0, cy1 - cy0
        ix0, iy0 = max(rx0, cx0), max(ry0, cy0)
        ix1, iy1 = min(rx1, cx1), min(ry1, cy1)
        out_path = region_path_for(train_dir, out_rel)
        if cw <= 0 or ch <= 0 or ix1 <= ix0 or iy1 <= iy0:
            out_path.unlink(missing_ok=True)
            continue
        out_doc = {
            "version": FORMAT_VERSION,
            "image_size": {"w": int(round(cw)), "h": int(round(ch))},
            "regions": [{
                **{k: v for k, v in region.items() if k != "box"},
                "box": {
                    "x": (ix0 - cx0) / cw,
                    "y": (iy0 - cy0) / ch,
                    "w": (ix1 - ix0) / cw,
                    "h": (iy1 - iy0) / ch,
                },
            }],
        }
        _atomic_write(out_path, out_doc)
        written.append(out_path)

    src_path = region_path_for(train_dir, src_rel)
    if src_path not in written:
        src_path.unlink(missing_ok=True)
