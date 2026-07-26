"""EvalSession worker —— 一次评估跑在一个进程里（issue #465）。

旧模型下一次评估 fan-out 成 `(候选数) × (1 出图 + N 指标)` 个作业，各自一个 tasks 行
和一个日志目录。这里全部收进一个 `eval_session` 作业，内部按阶段顺序跑：

    generate            → 逐候选出图（含 baseline 纯底模对照）
    metric:<runner>     → 每个启用的指标 runner 一个阶段，批量过所有候选
    aggregate           → rollup 状态 + 写 report.json

**断点续跑**：阶段状态落 DB（`eval_candidates.status` / `eval_metric_results.status`），
重跑时已 done 的候选跳过出图、已 done 的指标跳过重算。作业被 kill / 机器断电后重新
入队，不会从头再来。

**失败隔离**：一个候选出图失败只标它自己 failed，继续下一个；一个指标对某候选失败只
标那一格。整体状态由 `eval_session.rollup_status` 汇总成 done / partial / failed ——
部分结果仍然可看，不会因为一个 checkpoint 崩了就报整体失败。

**出图走测试出图那条常驻 daemon**：整个出图阶段共用**一个** daemon 实例
（`eval_generation.DaemonSampleGenerator`），底模只加载一次，候选之间由 daemon 的
`ModelCache` 热换 LoRA 权重。指标算法仍直接调 `eval_*.run_*_job`，每个候选内部仍是
一个 eval_samples run，只是 `eval_root` 指向 `eval/sessions/<id>/samples/`。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from studio import db, secrets
from studio.services import (
    eval_ccip,
    eval_clip,
    eval_dino,
    eval_generation,
    eval_registry,
    eval_samples,
    eval_session,
    eval_tag,
)
from studio.services.projects import projects, versions

logger = logging.getLogger(__name__)

# runner key → (跑它的函数, 从 secrets 取默认模型名的取值器, 阶段级共享 scorer)
#
# 第三项是「模型只加载一次」的关键：`_stage_metric` 本来就是「一个指标跑完所有候选
# 再换下一个」，但 run_*_job 每次调用都自己加载一遍模型（旧模型每候选一个子进程时
# 的正确形状）。`shared_scorer` 把模型的生命周期提到阶段上，跑完即释放 —— 跟出图那
# 刀（#470）同构，只是小一号。
_RUNNERS: dict[
    str,
    tuple[Callable[..., dict[str, Any]], Callable[[Any], str], Callable[..., Any]],
] = {
    "clip": (
        eval_clip.run_clip_job, lambda cfg: cfg.clip_model_name,
        eval_clip.shared_scorer,
    ),
    "dino": (
        eval_dino.run_dino_job, lambda cfg: cfg.dino_model_name,
        eval_dino.shared_scorer,
    ),
    "tag": (
        eval_tag.run_tag_job, lambda _cfg: eval_tag.DEFAULT_MODEL_NAME,
        eval_tag.shared_scorer,
    ),
    "ccip": (
        eval_ccip.run_ccip_job, lambda cfg: cfg.ccip_model_name,
        eval_ccip.shared_scorer,
    ),
}


def run(task_id: int) -> int:
    def progress(line: str) -> None:
        print(line, flush=True)

    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
        if not task:
            progress(f"[error] task {task_id} not found")
            return 1
        params = task.get("params_decoded") or {}
        session_id = int(params.get("session_id") or 0)
        if not session_id:
            progress(f"[error] task {task_id} has no session_id")
            return 1
        session = eval_session.get_session(conn, session_id)
        if session is None:
            progress(f"[error] eval session {session_id} not found")
            return 1
        project = projects.get_project(conn, int(session["project_id"] or 0))
        version = versions.get_version(conn, int(session["version_id"] or 0))

    if not project or not version:
        with db.connection_for() as conn:
            _fail(conn, session_id, "project / version 不存在")
        progress("[error] project or version missing")
        return 1

    vdir = versions.version_dir(
        int(project["id"]), str(project["slug"]), str(version["label"])
    )
    plan = session.get("plan") or {}
    runners: list[str] = list(plan.get("metrics", {}).get("runners") or [])
    if not runners:
        # plan 是创建时冻结的；万一是老 plan 或空集合，退回当前 Settings 的启用集
        runners = eval_registry.enabled_runners(secrets.load().eval_metrics.enabled_metrics)

    with db.connection_for() as conn:
        eval_session.update_session(
            conn, session_id,
            status=eval_session.STATUS_RUNNING,
            stage=eval_session.STAGE_GENERATE,
            started_at=time.time(),
            error=None,
        )

    progress(
        f"[start] eval session={session_id} candidates="
        f"{len(plan.get('candidates') or [])}+baseline stages=1+{len(runners)}"
    )

    try:
        _stage_generate(session_id, task_id, project, version, vdir, progress)
        for runner in runners:
            _stage_metric(session_id, runner, project, version, vdir, progress)
        return _stage_aggregate(session_id, progress)
    except Exception as exc:  # noqa: BLE001
        logger.exception("eval session worker crashed (session=%s)", session_id)
        with db.connection_for() as conn:
            _fail(conn, session_id, str(exc))
        progress(f"[error] {exc}")
        return 1


# ---------------------------------------------------------------------------
# Stage 1：出图
# ---------------------------------------------------------------------------

def _stage_generate(
    session_id: int,
    task_id: int,
    project: dict[str, Any],
    version: dict[str, Any],
    vdir: Path,
    progress: Callable[[str], None],
) -> None:
    eval_root = eval_session.samples_root(session_id)
    with db.connection_for() as conn:
        eval_session.update_session(conn, session_id, stage=eval_session.STAGE_GENERATE)
        candidates = eval_session.list_candidates(conn, session_id)

    pending = [c for c in candidates if not _generation_complete(c, vdir, eval_root)]
    if not pending:
        progress("[generate] 全部候选已完成，跳过出图阶段")
        return

    # daemon 的生命周期是**整个出图阶段**，不是单个候选 —— 底模常驻的收益全在这里。
    # 断点续跑时只为真正要跑的候选起 daemon（上面先算 pending）。
    with eval_generation.DaemonSampleGenerator(progress, task_id=task_id) as generate:
        _generate_candidates(
            session_id, candidates, generate, project, version, vdir, eval_root, progress,
        )


def _generate_candidates(
    session_id: int,
    candidates: list[dict[str, Any]],
    generate: Any,
    project: dict[str, Any],
    version: dict[str, Any],
    vdir: Path,
    eval_root: Path,
    progress: Callable[[str], None],
) -> None:
    for cand in candidates:
        cid = int(cand["id"])
        label = f"{cand['role']}#{cand['ordinal']}"
        if _generation_complete(cand, vdir, eval_root):
            progress(f"[generate] {label} 已完成，跳过")
            continue

        with db.connection_for() as conn:
            eval_session.update_candidate(
                conn, cid, status=eval_session.STATUS_RUNNING, error=None
            )
        try:
            run_id = str(cand.get("run_id") or "")
            if run_id and _load_run(vdir, run_id, eval_root) is None:
                # run 文件没了（被清理 / 磁盘问题），DB 里的 id 已经失效。不丢弃它就会
                # 拿着这个 id 反复调 run_sample_job 撞「run 不存在」，永远恢复不了。
                progress(f"[generate] {label} run={run_id} 已丢失，重新出图")
                run_id = ""
            if not run_id:
                stored = str(cand.get("checkpoint_path") or "")
                # DB 存的是相对 version 目录的便携路径；create_run 只接受绝对路径
                # （它把相对路径解释成 version_dir/output/<raw>）。
                ckpt_abs = (
                    str(eval_session.resolve_candidate_path(vdir, stored))
                    if stored else None
                )
                run = eval_samples.create_run(
                    project, version, vdir,
                    checkpoint_path=ckpt_abs,
                    # Session 自己编排指标阶段，绝不能触发旧的「出图完自动排指标作业」链路
                    auto_metrics=False,
                    auto_source={"eval_session_id": session_id, "candidate_id": cid},
                    eval_root=eval_root,
                    baseline=cand.get("role") == "baseline",
                )
                run_id = str(run["run_id"])
                with db.connection_for() as conn:
                    eval_session.update_candidate(
                        conn, cid, run_id=run_id,
                        samples_total=int(run["summary"]["total"]),
                    )
            progress(f"[generate] {label} run={run_id}")
            result = eval_samples.run_sample_job(
                project, version, vdir, run_id,
                generator=generate,
                on_progress=progress, eval_root=eval_root,
            )
            summary = result.get("summary") or {}
            done = int(summary.get("done") or 0)
            ok = str(result.get("status") or "") == "done"
            with db.connection_for() as conn:
                eval_session.update_candidate(
                    conn, cid,
                    status=eval_session.STATUS_DONE if ok else eval_session.STATUS_FAILED,
                    samples_done=done,
                    error=None if ok else str(result.get("error") or "出图未完成"),
                )
            progress(
                f"[generate] {label} status={result.get('status')} "
                f"done={done}/{summary.get('total')}"
            )
        except Exception as exc:  # noqa: BLE001
            # 单个候选失败不中断整个 Session —— 其余候选仍然值得跑完
            logger.exception("candidate generation failed (candidate=%s)", cid)
            with db.connection_for() as conn:
                eval_session.update_candidate(
                    conn, cid, status=eval_session.STATUS_FAILED, error=str(exc)
                )
            progress(f"[generate] {label} 失败：{exc}")


def _load_run(vdir: Path, run_id: str, eval_root: Path) -> dict[str, Any] | None:
    """读一个 eval_samples run；不存在 / 读不动都返回 None（调用方据此重建）。"""
    try:
        return eval_samples.load_run(vdir, run_id, eval_root)
    except Exception:
        return None


def _generation_complete(
    cand: dict[str, Any], vdir: Path, eval_root: Path
) -> bool:
    """该候选的出图是否已经完成（断点续跑判据）。

    只信「DB 标 done **且** run 文件确实是 done」—— 光看 DB 会在 run 目录被清掉后
    误判成完成，后面的指标阶段会拿不到图。
    """
    if cand.get("status") != eval_session.STATUS_DONE:
        return False
    run_id = str(cand.get("run_id") or "")
    if not run_id:
        return False
    run = _load_run(vdir, run_id, eval_root)
    return bool(run) and str(run.get("status") or "") == "done"


# ---------------------------------------------------------------------------
# Stage 2..M+1：指标
# ---------------------------------------------------------------------------

def _stage_metric(
    session_id: int,
    runner: str,
    project: dict[str, Any],
    version: dict[str, Any],
    vdir: Path,
    progress: Callable[[str], None],
) -> None:
    spec = _RUNNERS.get(runner)
    if spec is None:
        progress(f"[metric:{runner}] 未知 runner，跳过")
        return
    run_fn, model_getter, shared_scorer = spec
    model_name = model_getter(secrets.load().eval_metrics)
    metric_keys = eval_registry.runner_metrics(runner)
    eval_root = eval_session.samples_root(session_id)
    stage = eval_session.metric_stage(runner)

    with db.connection_for() as conn:
        eval_session.update_session(conn, session_id, stage=stage)
        candidates = eval_session.list_candidates(conn, session_id)
        existing = eval_session.list_metric_results(conn, session_id)

    # 模型在这里加载一次、跑完全部候选后释放（惰性：真正要打分时才 load，所以
    # 「候选全跳过」的情形一次都不加载）
    with shared_scorer(progress) as scorer:
        _score_candidates(
            session_id, runner, candidates, existing, scorer, run_fn,
            project, version, vdir, eval_root, model_name, metric_keys, stage, progress,
        )


def _score_candidates(
    session_id: int,
    runner: str,
    candidates: list[dict[str, Any]],
    existing: dict[int, list[dict[str, Any]]],
    scorer: Any,
    run_fn: Callable[..., dict[str, Any]],
    project: dict[str, Any],
    version: dict[str, Any],
    vdir: Path,
    eval_root: Path,
    model_name: str,
    metric_keys: list[str],
    stage: str,
    progress: Callable[[str], None],
) -> None:
    for cand in candidates:
        cid = int(cand["id"])
        label = f"{cand['role']}#{cand['ordinal']}"
        run_id = str(cand.get("run_id") or "")
        by_key = {str(r["metric_key"]): r for r in existing.get(cid, [])}

        if cand.get("status") != eval_session.STATUS_DONE or not run_id:
            _skip_metrics(
                cid, metric_keys, model_name, reason="出图未完成，跳过指标",
            )
            progress(f"[{stage}] {label} 无可用出图，跳过")
            continue
        if all(
            by_key.get(k, {}).get("status") == eval_session.STATUS_DONE
            for k in metric_keys
        ):
            progress(f"[{stage}] {label} 已算过，跳过")
            continue

        try:
            progress(f"[{stage}] {label} run={run_id}")
            saved = run_fn(
                project, version, vdir, run_id,
                scorer=scorer,
                model_name=model_name, on_progress=progress, eval_root=eval_root,
            )
            _record_metrics(cid, metric_keys, saved, model_name)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "metric runner failed (session=%s candidate=%s runner=%s)",
                session_id, cid, runner,
            )
            with db.connection_for() as conn:
                for key in metric_keys:
                    eval_session.set_metric_result(
                        conn, cid, key,
                        status=eval_session.STATUS_FAILED,
                        model_ref=model_name, reason=str(exc),
                    )
            progress(f"[{stage}] {label} 失败：{exc}")


def _record_metrics(
    candidate_id: int,
    metric_keys: list[str],
    saved: dict[str, Any],
    model_name: str,
) -> None:
    """把 runner 写进 metrics.json 的状态搬进 DB（DB 才是运行状态的真相）。"""
    states = saved.get("metric_states") or {}
    with db.connection_for() as conn:
        for key in metric_keys:
            state = states.get(key) or {}
            raw_value = state.get("value")
            try:
                value = float(raw_value) if raw_value is not None else None
            except (TypeError, ValueError):
                value = None
            status = str(state.get("status") or "")
            # runner 侧的 not_run / skipped 都不是失败（比如 tag_recall 遇上自然语言
            # caption、ccip 遇上多角色图）—— 照原状记下来，别粉饰成 done 也别报错
            mapped = (
                eval_session.STATUS_DONE if status == "done"
                else eval_session.STATUS_FAILED if status == "failed"
                else status or "not_run"
            )
            eval_session.set_metric_result(
                conn, candidate_id, key,
                status=mapped,
                value=value,
                model_ref=str(state.get("model_name") or model_name),
                sample_count=_int_or_none(state.get("sample_count")),
                reason=str(state.get("reason")) if state.get("reason") else None,
            )


def _skip_metrics(
    candidate_id: int,
    metric_keys: list[str],
    model_name: str,
    *,
    reason: str,
) -> None:
    with db.connection_for() as conn:
        for key in metric_keys:
            eval_session.set_metric_result(
                conn, candidate_id, key,
                status="skipped", model_ref=model_name, reason=reason,
            )


def _int_or_none(raw: Any) -> int | None:
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Stage 3：聚合
# ---------------------------------------------------------------------------

def _stage_aggregate(session_id: int, progress: Callable[[str], None]) -> int:
    with db.connection_for() as conn:
        eval_session.update_session(conn, session_id, stage=eval_session.STAGE_AGGREGATE)
        candidates = eval_session.list_candidates(conn, session_id)
        results = eval_session.list_metric_results(conn, session_id)
        status = eval_session.rollup_status(candidates, results)
        eval_session.update_session(
            conn, session_id, status=status, stage=None, finished_at=time.time()
        )
        report = eval_session.write_report(conn, session_id)

    n_metrics = sum(
        1 for rows in results.values()
        for r in rows if r.get("status") == eval_session.STATUS_DONE
    )
    progress(
        f"[done] session={session_id} status={status} "
        f"candidates={len(candidates)} metrics_done={n_metrics}"
    )
    if report is None:
        progress("[warn] report 生成失败（结果仍在数据库里）")
    # partial 仍算作业成功：有结果可看，队列不该标红
    return 0 if status in (eval_session.STATUS_DONE, eval_session.STATUS_PARTIAL) else 1


def _fail(conn, session_id: int, message: str) -> None:
    eval_session.update_session(
        conn, session_id,
        status=eval_session.STATUS_FAILED,
        stage=None,
        finished_at=time.time(),
        error=message[:2000],
    )


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
