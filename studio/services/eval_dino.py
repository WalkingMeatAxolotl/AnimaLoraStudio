"""DINO-I metric runner for eval sample runs.

This module is scoped to subject/style fidelity over generated/reference image
pairs. Heavy dependencies are imported only inside the default scorer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from . import eval_metrics, eval_samples
from .projects import jobs as project_jobs

JOB_KIND = "eval_dino"
DEFAULT_MODEL_NAME = "facebook/dinov2-small"
CACHE_KEY = "dino"

DinoScorer = Callable[
    [dict[str, Any], Path, str, Callable[[str], None]],
    dict[str, Any],
]


class EvalDinoError(Exception):
    """Business error for DINO metric jobs."""


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
    """Queue a DINO-I metric job and mark DINO-I as pending."""
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
    result = _save_dino_state(
        version_dir,
        str(run["run_id"]),
        _metric_state(
            "pending",
            reason="eval_dino job queued",
            model_name=model,
            job_id=int(job["id"]),
        ),
        clear_value=True,
        eval_root=scoped_root,
    )
    return job, result


def run_dino_job(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
    *,
    scorer: DinoScorer | None = None,
    model_name: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    """Compute DINO-I for one completed eval sample run."""
    progress = on_progress or (lambda _line: None)
    model = _normalize_model_name(model_name)
    run = _load_scored_run(project, version, version_dir, run_id, eval_root)
    eval_root = _run_eval_root(run, eval_root)
    _done_image_items(run, version_dir, eval_root)
    _save_dino_state(
        version_dir,
        str(run["run_id"]),
        _metric_state("running", reason="eval_dino job running", model_name=model),
        clear_value=True,
        eval_root=eval_root,
    )

    try:
        progress(f"[eval-dino] scoring run={run['run_id']} model={model}")
        scored = (scorer or _default_scorer)(run, version_dir, model, progress)
        result = _result_from_scores(scored, model)
        saved = eval_metrics.save_result(
            version_dir,
            str(run["run_id"]),
            result,
            eval_root=eval_root,
        )
        state = saved["metric_states"]["dino_i"]
        progress(f"[eval-dino] done dino_i={state['status']}")
        return saved
    except Exception as exc:
        _save_failed(version_dir, str(run["run_id"]), model, str(exc), eval_root)
        raise


def _normalize_model_name(model_name: str | None) -> str:
    text = str(model_name or DEFAULT_MODEL_NAME).strip()
    if not text:
        return DEFAULT_MODEL_NAME
    if len(text) > 256:
        raise EvalDinoError("model_name is too long")
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
        raise EvalDinoError(f"eval sample run not found: {run_id}")
    if int(run.get("project_id") or 0) != int(project["id"]):
        raise EvalDinoError("eval sample run does not belong to this project")
    if int(run.get("version_id") or 0) != int(version["id"]):
        raise EvalDinoError("eval sample run does not belong to this version")
    if str(run.get("status") or "") != "done":
        raise EvalDinoError("eval sample run must be done before scoring DINO-I")
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
        raise EvalDinoError("eval sample run has no completed image files")
    return out


def _metric_state(
    status: str,
    *,
    value: float | None = None,
    reason: str = "",
    model_name: str,
    count: int | None = None,
    job_id: int | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "key": "dino_i",
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


def _save_dino_state(
    version_dir: Path,
    run_id: str,
    state: dict[str, Any],
    *,
    clear_value: bool,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if clear_value:
        metrics = {"dino_i": None}
    return eval_metrics.save_result(
        version_dir,
        run_id,
        {"metrics": metrics, "metric_states": {"dino_i": state}},
        eval_root=eval_root,
    )


def _save_failed(
    version_dir: Path,
    run_id: str,
    model_name: str,
    reason: str,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    return _save_dino_state(
        version_dir,
        run_id,
        _metric_state("failed", reason=reason, model_name=model_name),
        clear_value=True,
        eval_root=eval_root,
    )


def _result_from_scores(scored: dict[str, Any], model_name: str) -> dict[str, Any]:
    dino_i = _float_or_none(scored.get("dino_i"))
    count = _int_or_none(scored.get("dino_i_count"))
    return {
        "metrics": {"dino_i": dino_i},
        "metric_states": {
            "dino_i": _metric_state(
                "done" if dino_i is not None else "unavailable",
                value=dino_i,
                reason=str(
                    scored.get("dino_i_reason")
                    or (
                        "computed"
                        if dino_i is not None
                        else "no paired reference/image pairs"
                    )
                ),
                model_name=str(scored.get("model_name") or model_name),
                count=count,
            )
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
    from transformers import AutoImageProcessor, AutoModel

    eval_root = _run_eval_root(run)
    items = _done_image_items(run, version_dir, eval_root)
    references = _reference_paths(run, version_dir)
    paired_items: list[dict[str, Any]] = []
    ref_paths: list[Path] = []
    for item in items:
        ref_path = references.get(str(item.get("prompt_id") or ""))
        if ref_path is None:
            continue
        paired_items.append(item)
        ref_paths.append(ref_path)

    if not paired_items:
        return {
            "model_name": model_name,
            "dino_i": None,
            "dino_i_count": 0,
            "dino_i_reason": "no paired reference/image pairs",
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    progress(f"[eval-dino] loading DINO on {device}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    image_paths = [item["_image_path"] for item in paired_items]
    with torch.inference_mode():
        image_emb = _encode_images(model, processor, image_paths, device, progress)
        ref_emb = _encode_images(
            model,
            processor,
            ref_paths,
            device,
            progress,
            label="reference",
        )

    values = [
        float((image_emb[idx] * ref_emb[idx]).sum().item())
        for idx in range(len(paired_items))
    ]

    cache = eval_metrics.ensure_embeddings_cache_dir(version_dir, eval_root) / CACHE_KEY
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "generated.npy", image_emb.cpu().numpy())
    np.save(cache / "reference.npy", ref_emb.cpu().numpy())
    _write_cache_metadata(cache, version_dir, run, model_name, paired_items, ref_paths)

    return {
        "model_name": model_name,
        "dino_i": _mean(values),
        "dino_i_count": len(values),
        "dino_i_reason": (
            "mean cosine similarity over generated image and paired reference DINO embeddings"
            if values else "no paired reference/image pairs"
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
        progress(f"[eval-dino] encoding {label} images {start + 1}-{end}")
        images = []
        for path in batch_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        feats = _feature_tensor(model(**inputs))
        chunks.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def _feature_tensor(output):
    """Return an image-level embedding tensor from common DINO model outputs."""
    pooler = getattr(output, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0]
    if isinstance(output, dict):
        pooler_value = output.get("pooler_output")
        if pooler_value is not None:
            return pooler_value
        hidden_value = output.get("last_hidden_state")
        if hidden_value is not None:
            return hidden_value[:, 0]
    if isinstance(output, (list, tuple)):
        if len(output) > 1 and output[1] is not None:
            return output[1]
        if output and output[0] is not None:
            return output[0][:, 0]
    raise EvalDinoError(
        f"DINO feature output is not tensor-like: {type(output).__name__}"
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _reference_paths(run: dict[str, Any], version_dir: Path) -> dict[str, Path]:
    manifest = (
        run.get("manifest_snapshot")
        if isinstance(run.get("manifest_snapshot"), dict)
        else {}
    )
    heldout_items = (
        manifest.get("heldout") if isinstance(manifest.get("heldout"), list) else []
    )
    prompts = (
        manifest.get("prompts") if isinstance(manifest.get("prompts"), list) else []
    )
    heldout_by_id: dict[str, dict[str, Any]] = {}
    heldout_by_image: dict[str, dict[str, Any]] = {}
    for item in heldout_items:
        if not isinstance(item, dict):
            continue
        image = str(item.get("image") or "")
        heldout_by_id[str(item.get("id") or image)] = item
        heldout_by_image[image] = item

    out: dict[str, Path] = {}
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        prompt_id = str(prompt.get("id") or "")
        heldout_id = str(prompt.get("heldout_id") or "")
        item = heldout_by_id.get(heldout_id)
        if item is None and prompt_id.startswith("caption:"):
            item = heldout_by_image.get(prompt_id[len("caption:"):])
        if item is None:
            continue
        image = str(item.get("image") or "")
        if image:
            out[prompt_id] = _train_rel_path(version_dir, image)
    return out


def _train_rel_path(version_dir: Path, rel: str) -> Path:
    raw = rel.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(":" in part for part in path.parts)
    ):
        raise EvalDinoError(f"invalid reference image path: {rel!r}")
    base = (version_dir / "train").resolve()
    resolved = (base / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise EvalDinoError(f"reference image path escapes train/: {rel!r}") from exc
    if not resolved.is_file():
        raise EvalDinoError(f"reference image not found: {rel}")
    return resolved


def _write_cache_metadata(
    cache: Path,
    version_dir: Path,
    run: dict[str, Any],
    model_name: str,
    items: list[dict[str, Any]],
    ref_paths: list[Path],
) -> None:
    payload = {
        "schema_version": 1,
        "run_id": run.get("run_id"),
        "model_name": model_name,
        "created_at": time.time(),
        "generated_count": len(items),
        "reference_count": len(ref_paths),
        "items": [
            {
                "id": item.get("id"),
                "prompt_id": item.get("prompt_id"),
                "filename": item.get("filename"),
                "reference": _rel_to_version(version_dir, ref_paths[idx]),
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
