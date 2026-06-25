"""CLIP-T / CLIP-I metric runner for eval sample runs.

This module is intentionally scoped to the first concrete metric family. It
loads CLIP lazily inside the default scorer so API and tests can exercise the
job/result contract without importing torch or transformers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from . import eval_metrics, eval_samples
from .projects import jobs as project_jobs

JOB_KIND = "eval_clip"
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
CACHE_KEY = "clip"

ClipScorer = Callable[
    [dict[str, Any], Path, str, Callable[[str], None]],
    dict[str, Any],
]


class EvalClipError(Exception):
    """Business error for CLIP metric jobs."""


def start_job(
    conn,
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
    *,
    model_name: str | None = None,
    eval_root: Path | None = None,
    task_id: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Queue a CLIP metric job and mark CLIP states as pending."""
    model = _normalize_model_name(model_name)
    run = _load_scored_run(project, version, version_dir, run_id, eval_root)
    scoped_root = _run_eval_root(run, eval_root)
    inferred_task_id = _run_task_id(run, task_id)
    params: dict[str, Any] = {
        "version_id": int(version["id"]),
        "run_id": str(run["run_id"]),
        "model_name": model,
    }
    if inferred_task_id:
        params["task_id"] = int(inferred_task_id)
    job = project_jobs.create_job(
        conn,
        project_id=int(project["id"]),
        version_id=int(version["id"]),
        kind=JOB_KIND,
        params=params,
    )
    result = _save_clip_states(
        version_dir,
        str(run["run_id"]),
        {
            "clip_t": _metric_state(
                "clip_t",
                "pending",
                reason="eval_clip job queued",
                model_name=model,
                job_id=int(job["id"]),
            ),
            "clip_i": _metric_state(
                "clip_i",
                "pending",
                reason="eval_clip job queued",
                model_name=model,
                job_id=int(job["id"]),
            ),
        },
        clear_values=True,
        eval_root=scoped_root,
    )
    return job, result


def run_clip_job(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
    *,
    scorer: ClipScorer | None = None,
    model_name: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    """Compute CLIP-T / CLIP-I for one completed eval sample run."""
    progress = on_progress or (lambda _line: None)
    model = _normalize_model_name(model_name)
    run = _load_scored_run(project, version, version_dir, run_id, eval_root)
    eval_root = _run_eval_root(run, eval_root)
    _done_image_items(run, version_dir, eval_root)
    _save_clip_states(
        version_dir,
        str(run["run_id"]),
        {
            "clip_t": _metric_state(
                "clip_t", "running", reason="eval_clip job running", model_name=model
            ),
            "clip_i": _metric_state(
                "clip_i", "running", reason="eval_clip job running", model_name=model
            ),
        },
        clear_values=True,
        eval_root=eval_root,
    )

    try:
        progress(f"[eval-clip] scoring run={run['run_id']} model={model}")
        scored = (scorer or _default_scorer)(run, version_dir, model, progress)
        result = _result_from_scores(scored, model)
        saved = eval_metrics.save_result(
            version_dir,
            str(run["run_id"]),
            result,
            eval_root=eval_root,
        )
        progress(
            "[eval-clip] done "
            f"clip_t={saved['metric_states']['clip_t']['status']} "
            f"clip_i={saved['metric_states']['clip_i']['status']}"
        )
        return saved
    except Exception as exc:
        _save_failed(version_dir, str(run["run_id"]), model, str(exc), eval_root)
        raise


def _normalize_model_name(model_name: str | None) -> str:
    text = str(model_name or DEFAULT_MODEL_NAME).strip()
    if not text:
        return DEFAULT_MODEL_NAME
    if len(text) > 256:
        raise EvalClipError("model_name is too long")
    return text


def _load_scored_run(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    run = eval_samples.load_run(version_dir, run_id, eval_root)
    if run is None:
        raise EvalClipError(f"eval sample run not found: {run_id}")
    if int(run.get("project_id") or 0) != int(project["id"]):
        raise EvalClipError("eval sample run does not belong to this project")
    if int(run.get("version_id") or 0) != int(version["id"]):
        raise EvalClipError("eval sample run does not belong to this version")
    if str(run.get("status") or "") != "done":
        raise EvalClipError("eval sample run must be done before scoring CLIP metrics")
    return run


def _run_eval_root(
    run: dict[str, Any],
    eval_root: Path | None = None,
) -> Path | None:
    if eval_root is not None:
        return eval_root
    if str(run.get("storage_scope") or "") == "task" and run.get("eval_root"):
        return Path(str(run["eval_root"]))
    return None


def _run_task_id(run: dict[str, Any], task_id: int | None = None) -> int | None:
    if task_id:
        return int(task_id)
    source = run.get("auto_source") if isinstance(run.get("auto_source"), dict) else {}
    value = int(source.get("task_id") or 0)
    return value or None


def _done_image_items(
    run: dict[str, Any],
    version_dir: Path,
    eval_root: Path | None = None,
) -> list[dict[str, Any]]:
    items = run.get("items") if isinstance(run.get("items"), list) else []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("status") != "done":
            continue
        filename = str(item.get("filename") or "")
        path = eval_samples.sample_image_path(
            version_dir,
            str(run["run_id"]),
            filename,
            eval_root,
        )
        if not path.is_file():
            continue
        out.append({**item, "_image_path": path})
    if not out:
        raise EvalClipError("eval sample run has no completed image files")
    return out


def _metric_state(
    key: str,
    status: str,
    *,
    value: float | None = None,
    reason: str = "",
    model_name: str,
    count: int | None = None,
    job_id: int | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "key": key,
        "status": status,
        "value": value,
        "reason": reason,
        "model_name": model_name,
    }
    if count is not None:
        state["count"] = int(count)
    if job_id is not None:
        state["job_id"] = int(job_id)
    return state


def _save_clip_states(
    version_dir: Path,
    run_id: str,
    states: dict[str, dict[str, Any]],
    *,
    clear_values: bool,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if clear_values:
        metrics = {"clip_t": None, "clip_i": None}
    return eval_metrics.save_result(
        version_dir,
        run_id,
        {"metrics": metrics, "metric_states": states},
        eval_root=eval_root,
    )


def _save_failed(
    version_dir: Path,
    run_id: str,
    model_name: str,
    reason: str,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    return _save_clip_states(
        version_dir,
        run_id,
        {
            "clip_t": _metric_state(
                "clip_t", "failed", reason=reason, model_name=model_name
            ),
            "clip_i": _metric_state(
                "clip_i", "failed", reason=reason, model_name=model_name
            ),
        },
        clear_values=True,
        eval_root=eval_root,
    )


def _result_from_scores(scored: dict[str, Any], model_name: str) -> dict[str, Any]:
    clip_t = _float_or_none(scored.get("clip_t"))
    clip_i = _float_or_none(scored.get("clip_i"))
    clip_t_count = _int_or_none(scored.get("clip_t_count"))
    clip_i_count = _int_or_none(scored.get("clip_i_count"))
    metrics: dict[str, Any] = {"clip_t": clip_t, "clip_i": clip_i}
    return {
        "metrics": metrics,
        "metric_states": {
            "clip_t": _metric_state(
                "clip_t",
                "done" if clip_t is not None else "unavailable",
                value=clip_t,
                reason=str(
                    scored.get("clip_t_reason")
                    or ("computed" if clip_t is not None else "no prompt/image pairs")
                ),
                model_name=str(scored.get("model_name") or model_name),
                count=clip_t_count,
            ),
            "clip_i": _metric_state(
                "clip_i",
                "done" if clip_i is not None else "unavailable",
                value=clip_i,
                reason=str(
                    scored.get("clip_i_reason")
                    or (
                        "computed"
                        if clip_i is not None
                        else "no paired reference/image pairs"
                    )
                ),
                model_name=str(scored.get("model_name") or model_name),
                count=clip_i_count,
            ),
        },
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_scorer(
    run: dict[str, Any],
    version_dir: Path,
    model_name: str,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor

    eval_root = _run_eval_root(run)
    items = _done_image_items(run, version_dir, eval_root)
    references = _reference_paths(run, version_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    progress(f"[eval-clip] loading CLIP on {device}")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    image_paths = [item["_image_path"] for item in items]
    prompts = [str(item.get("prompt") or "").strip() for item in items]
    ref_paths = [
        references.get(str(item.get("prompt_id") or ""))
        for item in items
    ]

    with torch.inference_mode():
        image_emb = _encode_images(model, processor, image_paths, device, progress)
        text_emb = _encode_texts(model, processor, prompts, device)
        ref_emb = (
            _encode_images(
                model,
                processor,
                [path for path in ref_paths if path is not None],
                device,
                progress,
                label="reference",
            )
            if any(path is not None for path in ref_paths)
            else None
        )

    clip_t_values: list[float] = []
    for idx, prompt in enumerate(prompts):
        if not prompt:
            continue
        clip_t_values.append(float((image_emb[idx] * text_emb[idx]).sum().item()))

    clip_i_values: list[float] = []
    ref_idx = 0
    for idx, ref_path in enumerate(ref_paths):
        if ref_path is None or ref_emb is None:
            continue
        clip_i_values.append(float((image_emb[idx] * ref_emb[ref_idx]).sum().item()))
        ref_idx += 1

    cache = eval_metrics.ensure_embeddings_cache_dir(version_dir, eval_root) / CACHE_KEY
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "generated.npy", image_emb.cpu().numpy())
    np.save(cache / "text.npy", text_emb.cpu().numpy())
    if ref_emb is not None:
        np.save(cache / "reference.npy", ref_emb.cpu().numpy())
    _write_cache_metadata(cache, version_dir, run, model_name, items, ref_paths)

    return {
        "model_name": model_name,
        "clip_t": _mean(clip_t_values),
        "clip_i": _mean(clip_i_values),
        "clip_t_count": len(clip_t_values),
        "clip_i_count": len(clip_i_values),
        "clip_t_reason": (
            "mean cosine similarity over generated image and prompt text embeddings"
            if clip_t_values else "no non-empty prompt/image pairs"
        ),
        "clip_i_reason": (
            "mean cosine similarity over generated image and paired reference embeddings"
            if clip_i_values else "no paired reference/image pairs"
        ),
    }


def _encode_images(
    model,
    processor,
    paths: list[Path],
    device: str,
    progress: Callable[[str], None],
    *,
    label: str = "generated",
):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    chunks = []
    batch_size = 8
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        end = start + len(batch_paths)
        progress(f"[eval-clip] encoding {label} images {start + 1}-{end}")
        images = []
        for path in batch_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        feats = _feature_tensor(
            model.get_image_features(**inputs),
            projection=getattr(model, "visual_projection", None),
        )
        chunks.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def _encode_texts(model, processor, texts: list[str], device: str):
    import torch
    import torch.nn.functional as F

    chunks = []
    batch_size = 32
    safe_texts = [text if text else " " for text in texts]
    for start in range(0, len(safe_texts), batch_size):
        batch = safe_texts[start:start + batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        feats = _feature_tensor(
            model.get_text_features(**inputs),
            projection=getattr(model, "text_projection", None),
        )
        chunks.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def _feature_tensor(output, *, projection=None):
    """Return the embedding tensor from Tensor or ModelOutput-like values."""
    if hasattr(output, "float"):
        return output
    text_embeds = getattr(output, "text_embeds", None)
    if text_embeds is not None:
        return text_embeds
    image_embeds = getattr(output, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds
    pooler = getattr(output, "pooler_output", None)
    if pooler is not None:
        return _maybe_project_feature(pooler, projection)
    if isinstance(output, dict):
        for key in ("text_embeds", "image_embeds"):
            value = output.get(key)
            if value is not None:
                return value
        pooler_value = output.get("pooler_output")
        if pooler_value is not None:
            return _maybe_project_feature(pooler_value, projection)
        hidden = output.get("last_hidden_state")
        if hidden is not None:
            return hidden
    if isinstance(output, (list, tuple)):
        for idx in (1, 0):
            if len(output) > idx and output[idx] is not None:
                return output[idx]
    raise EvalClipError(
        f"CLIP feature output is not tensor-like: {type(output).__name__}"
    )


def _maybe_project_feature(feature, projection):
    if projection is None:
        return feature
    in_features = getattr(projection, "in_features", None)
    shape = getattr(feature, "shape", None)
    last_dim = shape[-1] if shape is not None and len(shape) else None
    if in_features is not None and last_dim is not None and int(in_features) != int(last_dim):
        return feature
    return projection(feature)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _reference_paths(run: dict[str, Any], version_dir: Path) -> dict[str, Path]:
    """Map each item's prompt_id → its held-out reference image (validation/).

    Items without a ``reference_image`` (CLIP-T-only fallback when validation/ is
    empty) or whose reference file is missing are skipped — those score CLIP-T
    only, never CLIP-I.
    """
    out: dict[str, Path] = {}
    items = run.get("items") if isinstance(run.get("items"), list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        rel = item.get("reference_image")
        prompt_id = str(item.get("prompt_id") or "")
        if not rel or not prompt_id:
            continue
        try:
            out[prompt_id] = _reference_rel_path(version_dir, str(rel))
        except EvalClipError:
            continue
    return out


def _reference_rel_path(version_dir: Path, rel: str) -> Path:
    raw = rel.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(":" in part for part in path.parts)
    ):
        raise EvalClipError(f"invalid reference image path: {rel!r}")
    base = version_dir.resolve()
    resolved = (base / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise EvalClipError(f"reference image path escapes version dir: {rel!r}") from exc
    if not resolved.is_file():
        raise EvalClipError(f"reference image not found: {rel}")
    return resolved


def _write_cache_metadata(
    cache: Path,
    version_dir: Path,
    run: dict[str, Any],
    model_name: str,
    items: list[dict[str, Any]],
    ref_paths: list[Path | None],
) -> None:
    refs = [
        _rel_to_version(version_dir, path)
        if path is not None else None
        for path in ref_paths
    ]
    payload = {
        "schema_version": 1,
        "run_id": run.get("run_id"),
        "model_name": model_name,
        "created_at": time.time(),
        "generated_count": len(items),
        "reference_count": sum(1 for path in ref_paths if path is not None),
        "items": [
            {
                "id": item.get("id"),
                "prompt_id": item.get("prompt_id"),
                "filename": item.get("filename"),
                "reference": refs[idx],
            }
            for idx, item in enumerate(items)
        ],
    }
    (cache / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _rel_to_version(version_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(version_dir.resolve()).as_posix()
    except ValueError:
        return str(path)
