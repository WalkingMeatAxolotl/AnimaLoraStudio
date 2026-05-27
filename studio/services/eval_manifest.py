"""Version-scoped LoRA evaluation manifest helpers.

The manifest is intentionally small in this PR: it records the fixed inputs
for future eval runners, but does not generate samples or compute metrics.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .dataset.scan import IMAGE_EXTS
from .tagging.caption_format import caption_json_to_text

SCHEMA_VERSION = 1
EVAL_DIRNAME = "eval"
MANIFEST_NAME = "manifest.json"
DEFAULT_HELDOUT_LIMIT = 32
DEFAULT_SEEDS = [12345]
DEFAULT_GENERATION = {
    "width": 1024,
    "height": 1024,
    "steps": 24,
    "guidance_scale": 3.5,
    "lora_scale": 1.0,
}


class EvalManifestError(Exception):
    """Business error for invalid eval manifests."""


def eval_dir(version_dir: Path) -> Path:
    return version_dir / EVAL_DIRNAME


def manifest_path(version_dir: Path) -> Path:
    return eval_dir(version_dir) / MANIFEST_NAME


def load_manifest(version_dir: Path) -> dict[str, Any] | None:
    path = manifest_path(version_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalManifestError(f"eval manifest 读取失败: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalManifestError("eval manifest 必须是 JSON object")
    return data


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


def _text(value: Any, *, default: str = "", limit: int = 8000) -> str:
    if value is None:
        return default
    return " ".join(str(value).strip().split())[:limit]


def _normalize_rel_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalManifestError(f"{field} 不能为空")
    raw = value.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(":" in part for part in path.parts)
    ):
        raise EvalManifestError(f"{field} 必须是 train/ 内的相对路径: {value!r}")
    return path.as_posix()


def _train_file(version_dir: Path, rel: str) -> Path:
    base = (version_dir / "train").resolve()
    path = (base / Path(*PurePosixPath(rel).parts)).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise EvalManifestError(f"路径逃出 train/: {rel}") from exc
    return path


def _caption_path_for(image: Path) -> Path | None:
    js = image.with_suffix(".json")
    if js.exists():
        return js
    txt = image.with_suffix(".txt")
    if txt.exists():
        return txt
    return None


def _caption_prompt(caption_path: Path | None) -> str:
    if caption_path is None:
        return ""
    if caption_path.suffix.lower() == ".json":
        try:
            data = json.loads(caption_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        return caption_json_to_text(data if isinstance(data, dict) else None)
    try:
        return caption_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _iter_train_images(version_dir: Path):
    train_dir = version_dir / "train"
    if not train_dir.exists():
        return
    for folder in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        for image in sorted(folder.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                yield image


def _default_heldout_item(
    image: Path, train_dir: Path, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    rel = image.relative_to(train_dir).as_posix()
    folder = image.parent.relative_to(train_dir).as_posix()
    caption_path = _caption_path_for(image)
    caption_rel = (
        caption_path.relative_to(train_dir).as_posix() if caption_path else None
    )
    prompt = _caption_prompt(caption_path)
    item = {
        "id": rel,
        "folder": folder,
        "image": rel,
        "caption": caption_rel,
        "prompt": prompt,
        "seed": seed,
        "metadata": {
            "image_mtime": image.stat().st_mtime,
            "image_size": image.stat().st_size,
        },
    }
    prompt_item = {
        "id": f"caption:{rel}",
        "text": prompt,
        "source": "caption" if caption_rel else "empty",
        "heldout_id": rel,
    }
    return item, prompt_item


def create_default_manifest(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    ts = time.time() if now is None else float(now)
    train_dir = version_dir / "train"
    heldout: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    image_count = 0
    for idx, image in enumerate(_iter_train_images(version_dir) or []):
        image_count += 1
        if len(heldout) >= DEFAULT_HELDOUT_LIMIT:
            continue
        seed = DEFAULT_SEEDS[idx % len(DEFAULT_SEEDS)]
        item, prompt = _default_heldout_item(image, train_dir, seed)
        heldout.append(item)
        prompts.append(prompt)

    return {
        "schema_version": SCHEMA_VERSION,
        "project_id": int(project["id"]),
        "project_slug": str(project.get("slug") or ""),
        "version_id": int(version["id"]),
        "version_label": str(version.get("label") or ""),
        "created_at": ts,
        "updated_at": ts,
        "source": "default_from_train",
        "heldout": heldout,
        "prompts": prompts,
        "seeds": list(DEFAULT_SEEDS),
        "generation": dict(DEFAULT_GENERATION),
        "metadata": {
            "train_image_count": image_count,
            "heldout_limit": DEFAULT_HELDOUT_LIMIT,
            "selection": "sorted_train_images",
        },
    }


def _normalize_seed_list(raw: Any) -> list[int]:
    values = raw if isinstance(raw, list) else DEFAULT_SEEDS
    seeds: list[int] = []
    for value in values:
        seed = _clamp_int(value, DEFAULT_SEEDS[0], 0, 2**32 - 1)
        if seed not in seeds:
            seeds.append(seed)
    return seeds or list(DEFAULT_SEEDS)


def _normalize_generation(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    out = dict(DEFAULT_GENERATION)
    out["width"] = _clamp_int(data.get("width"), out["width"], 64, 4096)
    out["height"] = _clamp_int(data.get("height"), out["height"], 64, 4096)
    out["steps"] = _clamp_int(data.get("steps"), out["steps"], 1, 200)
    out["guidance_scale"] = _clamp_float(
        data.get("guidance_scale"), out["guidance_scale"], 0.0, 50.0
    )
    out["lora_scale"] = _clamp_float(
        data.get("lora_scale"), out["lora_scale"], 0.0, 2.0
    )
    if isinstance(data.get("negative_prompt"), str):
        out["negative_prompt"] = _text(data["negative_prompt"], limit=8000)
    return out


def _normalize_heldout_item(
    item: Any, version_dir: Path, fallback_seed: int
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise EvalManifestError("heldout item 必须是 JSON object")
    image_rel = _normalize_rel_path(item.get("image"), "heldout.image")
    if PurePosixPath(image_rel).suffix.lower() not in IMAGE_EXTS:
        raise EvalManifestError(f"heldout.image 不是支持的图片格式: {image_rel}")
    image_path = _train_file(version_dir, image_rel)
    if not image_path.is_file():
        raise EvalManifestError(f"heldout.image 不存在: {image_rel}")

    caption_rel = None
    if item.get("caption"):
        caption_rel = _normalize_rel_path(item.get("caption"), "heldout.caption")
        if PurePosixPath(caption_rel).suffix.lower() not in {".txt", ".json"}:
            raise EvalManifestError(f"heldout.caption 不是 .txt/.json: {caption_rel}")
        if not _train_file(version_dir, caption_rel).is_file():
            raise EvalManifestError(f"heldout.caption 不存在: {caption_rel}")

    folder = item.get("folder")
    if isinstance(folder, str) and folder.strip():
        folder = _normalize_rel_path(folder, "heldout.folder")
    else:
        folder = PurePosixPath(image_rel).parent.as_posix()

    out = {
        "id": _text(item.get("id"), default=image_rel, limit=512),
        "folder": folder,
        "image": image_rel,
        "caption": caption_rel,
        "prompt": _text(item.get("prompt"), limit=8000),
        "seed": _clamp_int(item.get("seed"), fallback_seed, 0, 2**32 - 1),
    }
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        out["metadata"] = metadata
    return out


def _normalize_prompts(raw: Any, heldout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(raw, list) and raw:
        prompts: list[dict[str, Any]] = []
        for idx, prompt in enumerate(raw):
            if not isinstance(prompt, dict):
                raise EvalManifestError("prompts item 必须是 JSON object")
            text = _text(prompt.get("text"), limit=8000)
            prompts.append({
                "id": _text(prompt.get("id"), default=f"prompt:{idx + 1}", limit=512),
                "text": text,
                "source": _text(prompt.get("source"), default="manual", limit=64),
            })
        return prompts
    return [
        {
            "id": f"caption:{item['image']}",
            "text": item.get("prompt", ""),
            "source": "heldout",
            "heldout_id": item["id"],
        }
        for item in heldout
    ]


def normalize_manifest(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    manifest: dict[str, Any],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise EvalManifestError("eval manifest 必须是 JSON object")
    ts = time.time() if now is None else float(now)
    seeds = _normalize_seed_list(manifest.get("seeds"))
    heldout = [
        _normalize_heldout_item(item, version_dir, seeds[idx % len(seeds)])
        for idx, item in enumerate(manifest.get("heldout") or [])
    ]
    prompts = _normalize_prompts(manifest.get("prompts"), heldout)
    metadata = (
        manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "project_id": int(project["id"]),
        "project_slug": str(project.get("slug") or ""),
        "version_id": int(version["id"]),
        "version_label": str(version.get("label") or ""),
        "created_at": _clamp_float(manifest.get("created_at"), ts, 0.0, ts),
        "updated_at": ts,
        "source": _text(manifest.get("source"), default="manual", limit=64),
        "heldout": heldout,
        "prompts": prompts,
        "seeds": seeds,
        "generation": _normalize_generation(manifest.get("generation")),
        "metadata": metadata,
    }


def save_manifest(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    manifest: dict[str, Any],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    normalized = normalize_manifest(project, version, version_dir, manifest, now=now)
    _atomic_write(manifest_path(version_dir), normalized)
    return normalized


def save_default_manifest(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    manifest = create_default_manifest(project, version, version_dir, now=now)
    _atomic_write(manifest_path(version_dir), manifest)
    return manifest
