"""Version eval metric result endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ._shared import _version_dir_or_404
from ....services import eval_metrics, eval_samples

router = APIRouter()


@router.get("/api/projects/{pid}/versions/{vid}/eval/metrics")
def list_eval_metric_results_endpoint(pid: int, vid: int) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    try:
        results = eval_metrics.list_results(vdir)
    except (eval_metrics.EvalMetricsError, eval_samples.EvalSamplesError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "metric_specs": eval_metrics.metric_specs(),
        "cache": eval_metrics.cache_layout(vdir),
        "results": results,
    }


@router.get("/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}/metrics")
def get_eval_metric_result_endpoint(
    pid: int, vid: int, run_id: str
) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    try:
        result = eval_metrics.load_result(vdir, run_id)
    except (eval_metrics.EvalMetricsError, eval_samples.EvalSamplesError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if result is None:
        raise HTTPException(404, f"eval sample run 不存在: {run_id}")
    return {"metric_specs": eval_metrics.metric_specs(), "result": result}
