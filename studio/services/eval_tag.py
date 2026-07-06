"""Tag-Recall metric runner —— 动漫原生的 prompt-following 替代 CLIP-T。

对每张生成图跑现有 WD14 tagger 回标，看 prompt 里的 booru tag 被召回多少：
``recall = |生成图 tag ∩ prompt tag| / |prompt tag|``。只统计 WD14 词表里有的 tag
（触发词、非 danbooru token 既无法被召回，也不计入分母），仅对 booru-tag 形态的
caption 有意义。复用项目已下载的 WD14（零新模型），无参考图。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import eval_metrics, eval_samples
from .projects import jobs as project_jobs

JOB_KIND = "eval_tag"
DEFAULT_MODEL_NAME = "wd14"
METRIC_KEY = "tag_recall"

TagScorer = Callable[
    [dict[str, Any], Path, str, Callable[[str], None]],
    dict[str, Any],
]


class EvalTagError(Exception):
    """Business error for tag-recall metric jobs."""


# ---------------------------------------------------------------------------
# tag 归一 + prompt 解析
# ---------------------------------------------------------------------------


def _norm_tag(tag: str) -> str:
    """归一键：下划线↔空格、大小写、首尾空格都不敏感（同 wd14 blacklist 口径）。"""
    return tag.replace("_", " ").strip().lower()


def parse_booru_tags(prompt: str | None) -> list[str]:
    """逗号分隔的 caption → 去重归一的 tag 列表（保序）。"""
    if not prompt:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in str(prompt).split(","):
        n = _norm_tag(raw)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# job lifecycle（与 eval_dino 同构）
# ---------------------------------------------------------------------------


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
    """Queue a tag-recall metric job and mark tag_recall pending."""
    model = DEFAULT_MODEL_NAME
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
    result = _save_state(
        version_dir,
        str(run["run_id"]),
        _metric_state(
            "pending",
            reason="eval_tag job queued",
            model_name=model,
            job_id=int(job["id"]),
        ),
        clear_value=True,
        eval_root=scoped_root,
    )
    return job, result


def run_tag_job(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
    *,
    scorer: TagScorer | None = None,
    model_name: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    """Compute tag-recall for one completed eval sample run."""
    progress = on_progress or (lambda _line: None)
    model = DEFAULT_MODEL_NAME
    run = _load_scored_run(project, version, version_dir, run_id, eval_root)
    eval_root = _run_eval_root(run, eval_root)
    _done_image_items(run, version_dir, eval_root)
    _save_state(
        version_dir,
        str(run["run_id"]),
        _metric_state("running", reason="eval_tag job running", model_name=model),
        clear_value=True,
        eval_root=eval_root,
    )
    try:
        progress(f"[eval-tag] scoring run={run['run_id']} (WD14 tag-recall)")
        scored = (scorer or _default_scorer)(run, version_dir, model, progress)
        result = _result_from_scores(scored, model)
        saved = eval_metrics.save_result(
            version_dir, str(run["run_id"]), result, eval_root=eval_root,
        )
        state = saved["metric_states"][METRIC_KEY]
        progress(f"[eval-tag] done tag_recall={state['status']}")
        return saved
    except Exception as exc:
        _save_failed(version_dir, str(run["run_id"]), model, str(exc), eval_root)
        raise


# ---------------------------------------------------------------------------
# helpers（与 eval_dino 同构，metric key = tag_recall）
# ---------------------------------------------------------------------------


def _load_scored_run(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    run = eval_samples.load_run(version_dir, run_id, eval_root)
    if run is None:
        raise EvalTagError(f"eval sample run not found: {run_id}")
    if int(run.get("project_id") or 0) != int(project["id"]):
        raise EvalTagError("eval sample run does not belong to this project")
    if int(run.get("version_id") or 0) != int(version["id"]):
        raise EvalTagError("eval sample run does not belong to this version")
    if str(run.get("status") or "") != "done":
        raise EvalTagError("eval sample run must be done before scoring tag-recall")
    return run


def _run_eval_root(run: dict[str, Any], eval_root: Path | None = None) -> Path | None:
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
    run: dict[str, Any], version_dir: Path, eval_root: Path | None = None,
) -> list[dict[str, Any]]:
    items = run.get("items") if isinstance(run.get("items"), list) else []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("status") != "done":
            continue
        path = eval_samples.sample_image_path(
            version_dir, str(run["run_id"]), str(item.get("filename") or ""), eval_root,
        )
        if not path.is_file():
            continue
        out.append({**item, "_image_path": path})
    if not out:
        raise EvalTagError("eval sample run has no completed image files")
    return out


def _metric_state(
    status: str, *, value: float | None = None, reason: str = "",
    model_name: str, count: int | None = None, job_id: int | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "key": METRIC_KEY, "status": status, "value": value,
        "reason": reason, "model_name": model_name,
    }
    if count is not None:
        state["count"] = int(count)
    if job_id is not None:
        state["job_id"] = int(job_id)
    return state


def _save_state(
    version_dir: Path, run_id: str, state: dict[str, Any], *,
    clear_value: bool, eval_root: Path | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {METRIC_KEY: None} if clear_value else {}
    return eval_metrics.save_result(
        version_dir, run_id,
        {"metrics": metrics, "metric_states": {METRIC_KEY: state}},
        eval_root=eval_root,
    )


def _save_failed(
    version_dir: Path, run_id: str, model_name: str, reason: str,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    return _save_state(
        version_dir, run_id,
        _metric_state("failed", reason=reason, model_name=model_name),
        clear_value=True, eval_root=eval_root,
    )


def _result_from_scores(scored: dict[str, Any], model_name: str) -> dict[str, Any]:
    value = _float_or_none(scored.get("tag_recall"))
    count = _int_or_none(scored.get("tag_recall_count"))
    return {
        "metrics": {METRIC_KEY: value},
        "metric_states": {
            METRIC_KEY: _metric_state(
                "done" if value is not None else "unavailable",
                value=value,
                reason=str(
                    scored.get("tag_recall_reason")
                    or ("computed" if value is not None else "no booru-tag prompts")
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


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------
# default scorer —— 复用 WD14 tagger
# ---------------------------------------------------------------------------


def _default_scorer(
    run: dict[str, Any],
    version_dir: Path,
    model_name: str,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    from studio.services.tagging.wd14 import WD14Tagger

    eval_root = _run_eval_root(run)
    items = _done_image_items(run, version_dir, eval_root)

    tagger = WD14Tagger()
    ok, msg = tagger.is_available()
    if not ok:
        raise EvalTagError(f"WD14 模型不可用：{msg}")
    tagger.prepare()
    known = {_norm_tag(t) for t in tagger.known_tags()}

    # 只保留有「WD14 词表内 prompt tag」的样本（触发词等不在词表→不计分母）
    scored_items: list[tuple[dict[str, Any], set[str]]] = []
    for item in items:
        prompt_tags = {t for t in parse_booru_tags(item.get("prompt")) if t in known}
        if prompt_tags:
            scored_items.append((item, prompt_tags))
    if not scored_items:
        return {
            "model_name": model_name, "tag_recall": None, "tag_recall_count": 0,
            "tag_recall_reason": "no in-vocabulary booru-tag prompts",
        }

    progress(f"[eval-tag] tagging {len(scored_items)} generated images with WD14")
    paths = [it["_image_path"] for it, _ in scored_items]
    preds = list(tagger.tag(
        paths, on_progress=lambda d, t: progress(f"[eval-tag] {d}/{t}"),
    ))

    recalls: list[float] = []
    for (_item, prompt_tags), pred in zip(scored_items, preds):
        if pred.get("error"):
            continue
        pred_set = {_norm_tag(t) for t in pred.get("tags", [])}
        hits = sum(1 for t in prompt_tags if t in pred_set)
        recalls.append(hits / len(prompt_tags))

    return {
        "model_name": model_name,
        "tag_recall": _mean(recalls),
        "tag_recall_count": len(recalls),
        "tag_recall_reason": (
            "mean booru-tag recall over generated images"
            if recalls else "no taggable generated images"
        ),
    }
