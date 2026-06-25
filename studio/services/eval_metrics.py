"""Metric result contract for ADR-0010 eval sample runs.

This module defines where metric runners will write their outputs and how the
API should represent empty/not-yet-computed states. It intentionally does not
compute CLIP, DINO, SSCD, CMMD, or any other metric.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from . import eval_manifest, eval_samples

SCHEMA_VERSION = 1
METRICS_FILE = "metrics.json"
CACHE_DIRNAME = "cache"
EMBEDDINGS_DIRNAME = "embeddings"

DEFAULT_METRICS: tuple[dict[str, Any], ...] = (
    {
        "key": "clip_t",
        "label": "CLIP-T",
        "question": "Do generated images match the prompt?",
        "requires": ["generated_images", "prompts"],
        "higher_is_better": True,
    },
    {
        "key": "clip_i",
        "label": "CLIP-I",
        "question": "Are generated images visually similar to references?",
        "requires": ["generated_images", "reference_images"],
        "higher_is_better": True,
    },
    {
        "key": "dino_i",
        "label": "DINO-I",
        "question": "Did the model learn subject or style features?",
        "requires": ["generated_images", "reference_images"],
        "higher_is_better": True,
    },
    {
        "key": "diversity",
        "label": "Diversity",
        "question": "Do same-prompt multi-seed samples avoid collapse?",
        "requires": ["generated_images", "prompt_groups"],
        "higher_is_better": True,
    },
    {
        "key": "sscd_nn",
        "label": "SSCD nearest neighbor",
        "question": "Are generated images suspiciously close to training images?",
        "requires": ["generated_images", "training_images"],
        "higher_is_better": False,
    },
    {
        "key": "paired_cmmd2",
        "label": "paired CMMD^2",
        "question": "How far is the generated set from the reference set?",
        "requires": ["generated_images", "reference_images"],
        "higher_is_better": False,
    },
)


class EvalMetricsError(Exception):
    """Business error for eval metric result files."""


def metric_specs() -> list[dict[str, Any]]:
    return [
        {
            "key": str(spec["key"]),
            "label": str(spec["label"]),
            "question": str(spec["question"]),
            "requires": list(spec["requires"]),
            "higher_is_better": bool(spec["higher_is_better"]),
        }
        for spec in DEFAULT_METRICS
    ]


def metrics_path(version_dir: Path, run_id: str, eval_root: Path | None = None) -> Path:
    return eval_samples.run_dir(version_dir, run_id, eval_root) / METRICS_FILE


def cache_dir(version_dir: Path, eval_root: Path | None = None) -> Path:
    root = eval_root if eval_root is not None else eval_manifest.eval_dir(version_dir)
    return root / CACHE_DIRNAME


def embeddings_cache_dir(version_dir: Path, eval_root: Path | None = None) -> Path:
    return cache_dir(version_dir, eval_root) / EMBEDDINGS_DIRNAME


def ensure_embeddings_cache_dir(version_dir: Path, eval_root: Path | None = None) -> Path:
    path = embeddings_cache_dir(version_dir, eval_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _rel_to_version(version_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(version_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _file_count_and_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    count = 0
    size = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        count += 1
        try:
            size += item.stat().st_size
        except OSError:
            pass
    return count, size


def cache_layout(version_dir: Path, eval_root: Path | None = None) -> dict[str, Any]:
    root = embeddings_cache_dir(version_dir, eval_root)
    entries: list[dict[str, Any]] = []
    if root.exists():
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            count, size = _file_count_and_size(child)
            entries.append({
                "key": child.name,
                "path": _rel_to_version(version_dir, child),
                "file_count": count,
                "size_bytes": size,
            })
    return {
        "embeddings_dir": _rel_to_version(version_dir, root),
        "entries": entries,
    }


def _default_metric_states() -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for spec in DEFAULT_METRICS:
        key = str(spec["key"])
        states[key] = {
            "key": key,
            "label": spec["label"],
            "status": "not_run",
            "value": None,
            "reason": "metric runner has not produced this metric yet",
            "question": spec["question"],
            "requires": list(spec["requires"]),
            "higher_is_better": bool(spec["higher_is_better"]),
        }
    return states


def _status_summary(states: dict[str, dict[str, Any]]) -> dict[str, int]:
    out = {
        "total": len(states),
        "not_run": 0,
        "pending": 0,
        "running": 0,
        "done": 0,
        "failed": 0,
        "unavailable": 0,
    }
    for state in states.values():
        status = str(state.get("status") or "not_run")
        if status not in out:
            out[status] = 0
        out[status] += 1
    return out


def _overall_status(states: dict[str, dict[str, Any]]) -> str:
    summary = _status_summary(states)
    if summary.get("running") or summary.get("pending"):
        return "running"
    if summary.get("failed"):
        return "failed"
    if summary.get("done") and summary.get("done") == summary.get("total"):
        return "done"
    if summary.get("done"):
        return "partial"
    if summary.get("unavailable"):
        return "partial"
    return "empty"


def _sample_run_ref(version_dir: Path, run: dict[str, Any], eval_root: Path | None = None) -> dict[str, Any]:
    run_id = str(run.get("run_id") or "")
    return {
        "run_id": run_id,
        "path": _rel_to_version(version_dir, eval_samples.run_path(version_dir, run_id, eval_root)),
        "status": run.get("status") or "unknown",
        "summary": run.get("summary") if isinstance(run.get("summary"), dict) else {},
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
    }


def _merge_metric_states(raw: Any, metrics: Any) -> dict[str, dict[str, Any]]:
    states = _default_metric_states()
    metric_values = metrics if isinstance(metrics, dict) else {}
    for key, value in metric_values.items():
        state = dict(states.get(str(key)) or {
            "key": str(key),
            "label": str(key),
            "question": "",
            "requires": [],
            "higher_is_better": None,
        })
        if isinstance(value, dict):
            state.update(value)
            if "status" not in value:
                state["status"] = "done"
        else:
            state["value"] = value
            state["status"] = "done"
        states[str(key)] = state

    raw_states = raw if isinstance(raw, dict) else {}
    for key, value in raw_states.items():
        if not isinstance(value, dict):
            continue
        state = dict(states.get(str(key)) or {"key": str(key), "label": str(key)})
        state.update(value)
        state.setdefault("key", str(key))
        state.setdefault("status", "not_run")
        states[str(key)] = state
    return states


def empty_result(version_dir: Path, run: dict[str, Any], eval_root: Path | None = None) -> dict[str, Any]:
    states = _default_metric_states()
    run_id = str(run.get("run_id") or "")
    return {
        "schema_version": SCHEMA_VERSION,
        "has_metrics": False,
        "status": "empty",
        "run_id": run_id,
        "project_id": run.get("project_id"),
        "project_slug": run.get("project_slug"),
        "version_id": run.get("version_id"),
        "version_label": run.get("version_label"),
        "created_at": None,
        "updated_at": None,
        "metrics_path": _rel_to_version(version_dir, metrics_path(version_dir, run_id, eval_root)),
        "sample_run": _sample_run_ref(version_dir, run, eval_root),
        "manifest_digest": run.get("manifest_digest"),
        "checkpoint": run.get("checkpoint") if isinstance(run.get("checkpoint"), dict) else {},
        "metrics": {},
        "metric_states": states,
        "summary": _status_summary(states),
        "cache": cache_layout(version_dir, eval_root),
    }


def _normalize_result(
    version_dir: Path,
    run: dict[str, Any],
    data: dict[str, Any],
    *,
    has_metrics: bool,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise EvalMetricsError("metrics result must be a JSON object")
    run_id = str(run.get("run_id") or "")
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    states = _merge_metric_states(data.get("metric_states"), metrics)
    status = str(data.get("status") or _overall_status(states))
    return {
        "schema_version": int(data.get("schema_version") or SCHEMA_VERSION),
        "has_metrics": has_metrics,
        "status": status,
        "run_id": run_id,
        "project_id": run.get("project_id"),
        "project_slug": run.get("project_slug"),
        "version_id": run.get("version_id"),
        "version_label": run.get("version_label"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "metrics_path": _rel_to_version(version_dir, metrics_path(version_dir, run_id, eval_root)),
        "sample_run": _sample_run_ref(version_dir, run, eval_root),
        "manifest_digest": run.get("manifest_digest"),
        "checkpoint": run.get("checkpoint") if isinstance(run.get("checkpoint"), dict) else {},
        "metrics": metrics,
        "metric_states": states,
        "summary": _status_summary(states),
        "cache": cache_layout(version_dir, eval_root),
    }


def load_result(version_dir: Path, run_id: str, eval_root: Path | None = None) -> dict[str, Any] | None:
    run = eval_samples.load_run(version_dir, run_id, eval_root)
    if run is None:
        return None
    path = metrics_path(version_dir, run_id, eval_root)
    if not path.exists():
        return empty_result(version_dir, run, eval_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalMetricsError(f"metrics result read failed: {exc}") from exc
    return _normalize_result(version_dir, run, data, has_metrics=True, eval_root=eval_root)


def save_result(
    version_dir: Path,
    run_id: str,
    result: dict[str, Any],
    *,
    eval_root: Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    run = eval_samples.load_run(version_dir, run_id, eval_root)
    if run is None:
        raise EvalMetricsError(f"eval sample run not found: {run_id}")
    ts = time.time() if now is None else float(now)
    path = metrics_path(version_dir, run_id, eval_root)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = {}
    payload = {
        key: value for key, value in existing.items()
        if key not in {"status", "summary", "updated_at"}
    }
    existing_metrics = (
        existing.get("metrics") if isinstance(existing.get("metrics"), dict) else {}
    )
    incoming_metrics = (
        result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    )
    if existing_metrics or incoming_metrics:
        metrics = dict(existing_metrics)
        for key, value in incoming_metrics.items():
            if value is None:
                metrics.pop(str(key), None)
            else:
                metrics[str(key)] = value
        payload["metrics"] = metrics
    existing_states = (
        existing.get("metric_states")
        if isinstance(existing.get("metric_states"), dict)
        else {}
    )
    incoming_states = (
        result.get("metric_states")
        if isinstance(result.get("metric_states"), dict)
        else {}
    )
    if existing_states or incoming_states:
        payload["metric_states"] = {**existing_states, **incoming_states}
    for key, value in result.items():
        if key not in {"metrics", "metric_states"}:
            payload[key] = value
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("created_at", existing.get("created_at") or ts)
    payload["updated_at"] = ts
    normalized = _normalize_result(version_dir, run, payload, has_metrics=True, eval_root=eval_root)
    to_write = {
        "schema_version": normalized["schema_version"],
        "status": normalized["status"],
        "run_id": normalized["run_id"],
        "created_at": normalized["created_at"],
        "updated_at": normalized["updated_at"],
        "manifest_digest": normalized["manifest_digest"],
        "checkpoint": normalized["checkpoint"],
        "metrics": normalized["metrics"],
        "metric_states": normalized["metric_states"],
        "summary": normalized["summary"],
    }
    _atomic_write(metrics_path(version_dir, run_id, eval_root), to_write)
    return load_result(version_dir, run_id, eval_root) or normalized


def list_results(version_dir: Path, eval_root: Path | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for run in eval_samples.list_runs(version_dir, eval_root):
        run_id = str(run.get("run_id") or "")
        if not run_id:
            continue
        result = load_result(version_dir, run_id, eval_root)
        if result is not None:
            results.append(result)
    return results
