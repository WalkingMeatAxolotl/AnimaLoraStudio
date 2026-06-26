"""CCIP-I metric runner —— anime 域角色身份保真（DINO-I 的动漫替代）。

对每张生成图与其 held-out 参考图，用 CCIP（deepghs/ccip_onnx）判是不是同一个
动漫角色，``ccip_i = 判为同角色的配对比例 ∈ [0,1]``（越高越好）。CCIP 走两套 ONNX：
``model_feat.onnx``（CAFormer 特征塔 → 768d）+ ``model_metrics.onnx``（learned
metric head，出成对 difference 矩阵），按变体 ``metrics.json`` 的 threshold 判同/异
（diff ≤ threshold = 同角色）。纯 onnxruntime，复用项目下载中心，不引 imgutils。

局限：仅单角色图有意义（多角色/画风 LoRA 不适用，靠 Settings 复选框门控）；对发色/
肤色不敏感。模型缺失时由 ``ensure_ccip_model`` 懒加载下载到 models/eval/ccip/。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import eval_metrics, eval_samples
from .projects import jobs as project_jobs

JOB_KIND = "eval_ccip"
DEFAULT_MODEL_NAME = "ccip-caformer-24-randaug-pruned"
METRIC_KEY = "ccip_i"

# CCIP 预处理（imgutils 同款）：384×384 BILINEAR、CHW、/255、CLIP mean/std。
_CCIP_SIZE = 384
_CCIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CCIP_STD = (0.26862954, 0.26130258, 0.27577711)

CcipScorer = Callable[
    [dict[str, Any], Path, str, Callable[[str], None]],
    dict[str, Any],
]


class EvalCcipError(Exception):
    """Business error for CCIP metric jobs."""


# ---------------------------------------------------------------------------
# job lifecycle（与 eval_dino 同构，metric key = ccip_i）
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
    """Queue a CCIP-I metric job and mark ccip_i pending."""
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
    result = _save_state(
        version_dir,
        str(run["run_id"]),
        _metric_state(
            "pending", reason="eval_ccip job queued",
            model_name=model, job_id=int(job["id"]),
        ),
        clear_value=True,
        eval_root=scoped_root,
    )
    return job, result


def run_ccip_job(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
    *,
    scorer: CcipScorer | None = None,
    model_name: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    """Compute CCIP-I for one completed eval sample run."""
    progress = on_progress or (lambda _line: None)
    model = _normalize_model_name(model_name)
    run = _load_scored_run(project, version, version_dir, run_id, eval_root)
    eval_root = _run_eval_root(run, eval_root)
    _done_image_items(run, version_dir, eval_root)
    _save_state(
        version_dir, str(run["run_id"]),
        _metric_state("running", reason="eval_ccip job running", model_name=model),
        clear_value=True, eval_root=eval_root,
    )
    try:
        progress(f"[eval-ccip] scoring run={run['run_id']} model={model}")
        scored = (scorer or _default_scorer)(run, version_dir, model, progress)
        result = _result_from_scores(scored, model)
        saved = eval_metrics.save_result(
            version_dir, str(run["run_id"]), result, eval_root=eval_root,
        )
        state = saved["metric_states"][METRIC_KEY]
        progress(f"[eval-ccip] done ccip_i={state['status']}")
        return saved
    except Exception as exc:
        _save_failed(version_dir, str(run["run_id"]), model, str(exc), eval_root)
        raise


def _normalize_model_name(model_name: str | None) -> str:
    text = str(model_name or DEFAULT_MODEL_NAME).strip()
    if not text:
        return DEFAULT_MODEL_NAME
    if len(text) > 256:
        raise EvalCcipError("model_name is too long")
    return text


# ---------------------------------------------------------------------------
# helpers（与 eval_dino 同构）
# ---------------------------------------------------------------------------


def _load_scored_run(project, version, version_dir, run_id, eval_root=None):
    run = eval_samples.load_run(version_dir, run_id, eval_root)
    if run is None:
        raise EvalCcipError(f"eval sample run not found: {run_id}")
    if int(run.get("project_id") or 0) != int(project["id"]):
        raise EvalCcipError("eval sample run does not belong to this project")
    if int(run.get("version_id") or 0) != int(version["id"]):
        raise EvalCcipError("eval sample run does not belong to this version")
    if str(run.get("status") or "") != "done":
        raise EvalCcipError("eval sample run must be done before scoring CCIP-I")
    return run


def _run_eval_root(run, eval_root=None):
    if eval_root is not None:
        return eval_root
    if str(run.get("storage_scope") or "") == "task" and run.get("eval_root"):
        return Path(str(run["eval_root"]))
    return None


def _run_task_id(run, task_id=None):
    if task_id:
        return int(task_id)
    source = run.get("auto_source") if isinstance(run.get("auto_source"), dict) else {}
    value = int(source.get("task_id") or 0)
    return value or None


def _done_image_items(run, version_dir, eval_root=None):
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
        raise EvalCcipError("eval sample run has no completed image files")
    return out


def _reference_paths(run: dict[str, Any], version_dir: Path) -> dict[str, Path]:
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
        except EvalCcipError:
            continue
    return out


def _reference_rel_path(version_dir: Path, rel: str) -> Path:
    from pathlib import PurePosixPath
    raw = rel.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(":" in part for part in path.parts)
    ):
        raise EvalCcipError(f"invalid reference image path: {rel!r}")
    base = version_dir.resolve()
    resolved = (base / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise EvalCcipError(f"reference image path escapes version dir: {rel!r}") from exc
    if not resolved.is_file():
        raise EvalCcipError(f"reference image not found: {rel}")
    return resolved


def _metric_state(status, *, value=None, reason="", model_name, count=None, job_id=None):
    state: dict[str, Any] = {
        "key": METRIC_KEY, "status": status, "value": value,
        "reason": reason, "model_name": model_name,
    }
    if count is not None:
        state["count"] = int(count)
    if job_id is not None:
        state["job_id"] = int(job_id)
    return state


def _save_state(version_dir, run_id, state, *, clear_value, eval_root=None):
    metrics: dict[str, Any] = {METRIC_KEY: None} if clear_value else {}
    return eval_metrics.save_result(
        version_dir, run_id,
        {"metrics": metrics, "metric_states": {METRIC_KEY: state}},
        eval_root=eval_root,
    )


def _save_failed(version_dir, run_id, model_name, reason, eval_root=None):
    return _save_state(
        version_dir, run_id,
        _metric_state("failed", reason=reason, model_name=model_name),
        clear_value=True, eval_root=eval_root,
    )


def _result_from_scores(scored: dict[str, Any], model_name: str) -> dict[str, Any]:
    value = _float_or_none(scored.get("ccip_i"))
    count = _int_or_none(scored.get("ccip_i_count"))
    return {
        "metrics": {METRIC_KEY: value},
        "metric_states": {
            METRIC_KEY: _metric_state(
                "done" if value is not None else "unavailable",
                value=value,
                reason=str(
                    scored.get("ccip_i_reason")
                    or ("computed" if value is not None else "no paired reference/image pairs")
                ),
                model_name=str(scored.get("model_name") or model_name),
                count=count,
            )
        },
    }


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------
# default scorer —— CCIP feat + metric ONNX（纯 onnxruntime）
# ---------------------------------------------------------------------------


def _make_session(model_path: Path):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
    )
    try:
        return ort.InferenceSession(str(model_path), providers=providers)
    except Exception:  # noqa: BLE001 —— GPU EP 创建失败降 CPU
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _ccip_preprocess(path: Path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as raw:
        img = raw.convert("RGB").resize((_CCIP_SIZE, _CCIP_SIZE), Image.BILINEAR)
    arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    mean = np.array(_CCIP_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(_CCIP_STD, dtype=np.float32)[:, None, None]
    return (arr - mean) / std  # (3, 384, 384)


def _default_scorer(run, version_dir, model_name, progress):
    import json
    import numpy as np
    from studio.services.models.downloader import ensure_ccip_model

    eval_root = _run_eval_root(run)
    items = _done_image_items(run, version_dir, eval_root)
    references = _reference_paths(run, version_dir)
    pairs: list[tuple[Path, Path]] = []
    for item in items:
        ref = references.get(str(item.get("prompt_id") or ""))
        if ref is not None:
            pairs.append((item["_image_path"], ref))
    if not pairs:
        return {
            "model_name": model_name, "ccip_i": None, "ccip_i_count": 0,
            "ccip_i_reason": "no paired reference/image pairs",
        }

    model_dir = ensure_ccip_model(model_name, on_log=progress)
    threshold = float(
        json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["threshold"]
    )
    progress(f"[eval-ccip] loading CCIP onnx (threshold={threshold:.4f})")
    feat_sess = _make_session(model_dir / "model_feat.onnx")
    metric_sess = _make_session(model_dir / "model_metrics.onnx")
    feat_in = feat_sess.get_inputs()[0].name
    metric_in = metric_sess.get_inputs()[0].name

    def _feat(path: Path):
        data = _ccip_preprocess(path)[None].astype(np.float32)  # (1,3,384,384)
        out = feat_sess.run(None, {feat_in: data})[0]
        return np.asarray(out).reshape(-1).astype(np.float32)  # (768,)

    n = len(pairs)
    progress(f"[eval-ccip] extracting CCIP features for {n} pairs")
    gen_feats = [_feat(g) for g, _ in pairs]
    ref_feats = [_feat(r) for _, r in pairs]
    stacked = np.stack(gen_feats + ref_feats, axis=0).astype(np.float32)  # (2n,768)
    diff = np.asarray(metric_sess.run(None, {metric_in: stacked})[0])  # (2n,2n)

    same = [1.0 if float(diff[i, n + i]) <= threshold else 0.0 for i in range(n)]
    return {
        "model_name": model_name,
        "ccip_i": _mean(same),
        "ccip_i_count": n,
        "ccip_i_reason": (
            "fraction of generated/reference pairs judged same character (CCIP)"
            if same else "no paired reference/image pairs"
        ),
    }
