"""EvalSession —— 一次评估作为一等领域对象（issue #465 的根治）。

## 为什么

0.21 及以前，一次评估是「一堆普通队列作业隐式串联」出来的工作流：每个 checkpoint 排
一个出图作业，出图完成再按启用指标各排一个指标作业。`(checkpoint 数 + baseline) ×
(1 + 指标 runner 数)` 个作业，每个一行 tasks + 一个 `studio_data/tasks/<id>/` 日志目录
—— 200 个 checkpoint 就是 603 个只含一个 run.log 的目录。运行状态还散在三处（tasks 行 /
run.json / metrics.json），多个指标 worker 并发改写同一个 metrics.json。

## 模型

    EvalSession       一次完整评估，拥有不可变 EvalPlan 和整体生命周期
    EvalCandidate     本次评估的一个被测对象（某个 checkpoint，或 baseline 纯底模对照）
    EvalMetricResult  某个候选在某个指标上的结果

一个 Session ↔ **一个** `eval_session` 类型的 tasks 行 ↔ 一个 `tasks/<id>/run.log`。
`1 + M` 个执行阶段（出图 + 每个指标 runner）跑在这一个 worker 进程内部，阶段状态落
本模块这几张表 —— **DB 是运行状态的唯一真相**，文件只存 artifacts 与导出。

"Candidate" 只是技术名称，不表示 Session 会替用户选优或晋级 —— 评估提供量化信号，
选哪个 checkpoint 仍由用户在测试页的 XY 对比里凭视觉判断。

## 历史保留

每次评估创建一个新 Session，**旧 Session 全部留档**（推翻 ADR 0011 Addendum §5 的
「每次运行评估先清空上一轮、永远只显示当次」）。理由：一次评估一个 task 之后历史本身
就是干净可读的，而「改了配置重训后指标怎么变」是评估的核心用途之一，删掉就白费了。
UI 默认显示最新 Session，可翻历史。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from studio import db, secrets
from studio.infrastructure.paths import (
    eval_session_dir,
    eval_session_plan_path,
    eval_session_report_path,
    eval_session_samples_dir,
)
from studio.services import eval_registry, eval_samples, eval_validation
from studio.services.projects import jobs as project_jobs, versions

logger = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = 1
TASK_TYPE = "eval_session"

# Session 状态。partial = 跑完了但有候选 / 指标失败（有部分结果可看，不算整体失败）。
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_CANCELED = "canceled"
TERMINAL_STATUSES = frozenset({STATUS_DONE, STATUS_PARTIAL, STATUS_FAILED, STATUS_CANCELED})

# 阶段名。`metric:<runner>` 是每个指标 runner 一个阶段。
STAGE_GENERATE = "generate"
STAGE_AGGREGATE = "aggregate"


def metric_stage(runner: str) -> str:
    return f"metric:{runner}"


class EvalSessionError(Exception):
    """Session 业务错误。"""


# ---------------------------------------------------------------------------
# EvalPlan —— 创建时冻结，之后只读
# ---------------------------------------------------------------------------

def checkpoint_digest(path: Path) -> str:
    """Checkpoint 的**廉价指纹**：size + mtime_ns 的短 hash。

    用途是「这个文件后来被换掉了吗」，不是内容校验 —— LoRA safetensors 动辄几百 MB
    到几 GB，创建 Session 时全文件 hash 会让入队卡住数十秒。真要内容级校验是另一件事，
    到时候单独加字段，不要偷偷把这里的语义改掉。
    """
    try:
        st = path.stat()
    except OSError:
        return ""
    raw = f"{st.st_size}:{st.st_mtime_ns}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _rel_to_version(version_dir: Path, path: Path) -> str:
    """绝对路径 → 相对 version 目录的 posix 路径（落不进 version 目录时退回文件名）。

    checkpoint / 验证图在 plan 和 DB 里一律存这种形式：studio_data 换位置、项目目录
    整体搬走之后仍然解析得到。`resolve_candidate_path` 是反向。
    """
    try:
        return path.resolve().relative_to(version_dir.resolve()).as_posix()
    except (ValueError, OSError):
        return path.name


def resolve_candidate_path(version_dir: Path, stored: str) -> Path:
    """DB / plan 里存的相对路径 → 绝对路径。

    传给 `eval_samples.create_run` 的必须是**绝对**路径：它对相对路径的解释是
    `version_dir / "output" / <raw>`，把 `output/x.safetensors` 喂进去会拼成
    `output/output/x.safetensors`。
    """
    p = Path(stored)
    return p if p.is_absolute() else (version_dir / stored)


def _reference_manifest(version_dir: Path) -> dict[str, Any]:
    """冻结验证集清单：参与评估的图 + 它的 caption。

    Session 创建时定下来，之后用户往 validation/ 里加删图都不影响这次评估的口径 ——
    「同一个 plan 跑出来的结果可比」是 EvalPlan 存在的意义。
    """
    entries: list[dict[str, Any]] = []
    for folder, image in eval_validation.iter_images(
        eval_validation.validation_dir(version_dir)
    ):
        entries.append({
            "folder": folder,
            "image": _rel_to_version(version_dir, image),
            "digest": checkpoint_digest(image),
        })
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False).encode()
    return {
        "count": len(entries),
        "digest": hashlib.sha256(payload).hexdigest()[:16],
        "entries": entries,
    }


def build_plan(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    *,
    checkpoints: list[dict[str, Any]],
    trigger: str,
    baseline: bool,
    metric_keys: Iterable[str],
    skip_count: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """构造不可变 EvalPlan。

    `checkpoints` 是已经按策略挑好的 `list_lora_ckpts` 条目（调用方负责选，见
    `eval_auto.select_checkpoints`）—— plan 只负责把结果连同来龙去脉一起冻住。
    """
    ts = time.time() if now is None else float(now)
    cfg = eval_samples._read_config(project, version)
    generation = eval_samples._generation_from_cfg(cfg)
    runners = eval_registry.enabled_runners(metric_keys)
    active_metrics = sorted(eval_registry.normalize_enabled(metric_keys))
    reference = _reference_manifest(version_dir)

    planned: list[dict[str, Any]] = []
    for ordinal, ckpt in enumerate(checkpoints):
        raw = str(ckpt.get("path") or "")
        if not raw:
            continue
        p = Path(raw)
        planned.append({
            "role": "checkpoint",
            "ordinal": ordinal,
            # 存**相对 version 目录**的便携路径（`output/xxx.safetensors`），跟
            # eval_samples 的 run.json 口径一致。`list_lora_ckpts` 给的是绝对路径，
            # 直接存下来会在 studio_data 迁移 / 换机器后全部失效。
            "path": _rel_to_version(version_dir, p),
            "label": str(ckpt.get("label") or p.stem),
            "kind": str(ckpt.get("kind") or "other"),
            "epoch": int(ckpt["value"]) if ckpt.get("kind") == "epoch" else None,
            "step": int(ckpt["value"]) if ckpt.get("kind") == "step" else None,
            "digest": checkpoint_digest(p),
        })
    if not planned:
        raise EvalSessionError("没有可评估的 checkpoint")

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at": ts,
        "trigger": trigger,
        "project": {"id": int(project["id"]), "slug": str(project.get("slug") or "")},
        "version": {"id": int(version["id"]), "label": str(version.get("label") or "")},
        "checkpoint_sampling": {"skip_count": skip_count},
        "candidates": planned,
        "baseline": {
            "enabled": bool(baseline),
            # baseline = 同 prompt / 同 seed / 纯底模（lora_scale=0），给各 checkpoint
            # 算 Δ。复用首个候选的路径只是为了走通同一条出图链路，LoRA 实际不生效。
            "checkpoint_path": planned[0]["path"] if baseline else None,
        },
        "generation": generation,
        "reference_manifest": reference,
        "metrics": {"keys": active_metrics, "runners": runners},
    }


# ---------------------------------------------------------------------------
# 创建 —— 一个事务里建 session + candidates + metric placeholders + task 行
# ---------------------------------------------------------------------------

def create_session(
    conn: sqlite3.Connection,
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    *,
    checkpoints: list[dict[str, Any]],
    trigger: str,
    parent_task_id: int | None = None,
    baseline: bool | None = None,
    metric_keys: Iterable[str] | None = None,
    skip_count: int | None = None,
) -> dict[str, Any]:
    """建一个 Session（含 candidates / metric placeholders）+ 一个 eval_session task。

    自动评估和手动评估共用这一条创建路径（设计稿 §0.3 第 5/6 项）。
    """
    cfg = secrets.load().eval_metrics
    use_baseline = bool(cfg.eval_baseline_enabled) if baseline is None else bool(baseline)
    keys = cfg.enabled_metrics if metric_keys is None else list(metric_keys)

    plan = build_plan(
        project, version, version_dir,
        checkpoints=checkpoints,
        trigger=trigger,
        baseline=use_baseline,
        metric_keys=keys,
        skip_count=skip_count,
    )

    ts = float(plan["created_at"])
    cur = conn.execute(
        "INSERT INTO eval_sessions"
        "(parent_task_id, project_id, version_id, trigger, status, plan_json, created_at) "
        "VALUES(?, ?, ?, ?, ?, ?, ?)",
        (
            int(parent_task_id) if parent_task_id else None,
            int(project["id"]),
            int(version["id"]),
            trigger,
            STATUS_PENDING,
            json.dumps(plan, ensure_ascii=False),
            ts,
        ),
    )
    session_id = int(cur.lastrowid)

    metric_keys_planned = list(plan["metrics"]["keys"])
    rows = list(plan["candidates"])
    if plan["baseline"]["enabled"]:
        rows.append({
            "role": "baseline",
            "ordinal": len(rows),
            "path": plan["baseline"]["checkpoint_path"],
            "epoch": None,
            "step": None,
            "digest": "",
        })
    for row in rows:
        cand = conn.execute(
            "INSERT INTO eval_candidates"
            "(session_id, role, checkpoint_path, checkpoint_digest, epoch, step,"
            " ordinal, status, samples_total) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, row["role"], str(row.get("path") or ""),
                row.get("digest") or None, row.get("epoch"), row.get("step"),
                int(row["ordinal"]), STATUS_PENDING,
                int(plan["reference_manifest"]["count"]),
            ),
        )
        cand_id = int(cand.lastrowid)
        for key in metric_keys_planned:
            conn.execute(
                "INSERT INTO eval_metric_results(candidate_id, metric_key, status) "
                "VALUES(?, ?, ?)",
                (cand_id, key, STATUS_PENDING),
            )

    # task 行：worker 模块由 kind 自动派生（supervisor/cmd_builder），
    # 所以 eval_session kind 直接落到 studio.workers.eval_session_worker。
    task_id = db.create_task(
        conn,
        name=TASK_TYPE,
        config_name=TASK_TYPE,
        task_type=TASK_TYPE,
        params={"session_id": session_id},
        project_id=int(project["id"]),
        version_id=int(version["id"]),
    )
    conn.execute(
        "UPDATE eval_sessions SET task_id = ? WHERE id = ?", (int(task_id), session_id)
    )
    conn.commit()

    _write_plan_file(session_id, plan)
    logger.info(
        "created eval session=%s task=%s trigger=%s candidates=%s metrics=%s",
        session_id, task_id, trigger, len(rows), metric_keys_planned,
    )
    return get_session(conn, session_id) or {}


def _write_plan_file(session_id: int, plan: dict[str, Any]) -> None:
    """plan.json 是 DB plan_json 的人类可读副本，写失败不影响 Session 运行。"""
    try:
        path = eval_session_plan_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        logger.warning("failed writing plan.json for session=%s", session_id, exc_info=True)


# ---------------------------------------------------------------------------
# 读
# ---------------------------------------------------------------------------

def _row(row: sqlite3.Row | None) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    out = dict(row)
    if "plan_json" in out:
        try:
            out["plan"] = json.loads(out["plan_json"] or "{}")
        except (TypeError, ValueError):
            out["plan"] = {}
    return out


def get_session(conn: sqlite3.Connection, session_id: int) -> Optional[dict[str, Any]]:
    return _row(
        conn.execute("SELECT * FROM eval_sessions WHERE id = ?", (int(session_id),)).fetchone()
    )


def get_session_by_task(conn: sqlite3.Connection, task_id: int) -> Optional[dict[str, Any]]:
    return _row(
        conn.execute(
            "SELECT * FROM eval_sessions WHERE task_id = ?", (int(task_id),)
        ).fetchone()
    )


def list_sessions(
    conn: sqlite3.Connection,
    *,
    project_id: int | None = None,
    version_id: int | None = None,
    parent_task_id: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """按创建时间倒序列 Session（最新在前）——历史全部保留，UI 默认取第一条。"""
    sql = "SELECT * FROM eval_sessions WHERE 1=1"
    params: list[Any] = []
    if project_id is not None:
        sql += " AND project_id = ?"
        params.append(int(project_id))
    if version_id is not None:
        sql += " AND version_id = ?"
        params.append(int(version_id))
    if parent_task_id is not None:
        sql += " AND parent_task_id = ?"
        params.append(int(parent_task_id))
    sql += " ORDER BY created_at DESC, id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [_row(r) or {} for r in conn.execute(sql, params)]


def list_candidates(conn: sqlite3.Connection, session_id: int) -> list[dict[str, Any]]:
    return [
        dict(r) for r in conn.execute(
            "SELECT * FROM eval_candidates WHERE session_id = ? ORDER BY ordinal, id",
            (int(session_id),),
        )
    ]


def list_metric_results(
    conn: sqlite3.Connection, session_id: int
) -> dict[int, list[dict[str, Any]]]:
    """candidate_id → 该候选的指标结果列表。"""
    out: dict[int, list[dict[str, Any]]] = {}
    for r in conn.execute(
        "SELECT m.* FROM eval_metric_results m "
        "JOIN eval_candidates c ON c.id = m.candidate_id "
        "WHERE c.session_id = ? ORDER BY c.ordinal, m.metric_key",
        (int(session_id),),
    ):
        row = dict(r)
        out.setdefault(int(row["candidate_id"]), []).append(row)
    return out


# ---------------------------------------------------------------------------
# 写
# ---------------------------------------------------------------------------

_SESSION_FIELDS = frozenset({
    "task_id", "status", "stage", "started_at", "finished_at", "error",
})
_CANDIDATE_FIELDS = frozenset({
    "status", "samples_done", "samples_total", "run_id", "error",
})


def update_session(conn: sqlite3.Connection, session_id: int, **fields: Any) -> None:
    unknown = set(fields) - _SESSION_FIELDS
    if unknown:
        raise EvalSessionError(f"未知 session 字段: {sorted(unknown)}")
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE eval_sessions SET {sets} WHERE id = ?",
        (*fields.values(), int(session_id)),
    )
    conn.commit()


def update_candidate(conn: sqlite3.Connection, candidate_id: int, **fields: Any) -> None:
    unknown = set(fields) - _CANDIDATE_FIELDS
    if unknown:
        raise EvalSessionError(f"未知 candidate 字段: {sorted(unknown)}")
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE eval_candidates SET {sets} WHERE id = ?",
        (*fields.values(), int(candidate_id)),
    )
    conn.commit()


def set_metric_result(
    conn: sqlite3.Connection,
    candidate_id: int,
    metric_key: str,
    *,
    status: str,
    value: float | None = None,
    model_ref: str | None = None,
    sample_count: int | None = None,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """写一个候选 × 指标的结果。(candidate_id, metric_key) 上有唯一索引 → upsert。"""
    conn.execute(
        "INSERT INTO eval_metric_results"
        "(candidate_id, metric_key, status, value, model_ref, sample_count, reason, details_json) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(candidate_id, metric_key) DO UPDATE SET "
        "status = excluded.status, value = excluded.value, model_ref = excluded.model_ref, "
        "sample_count = excluded.sample_count, reason = excluded.reason, "
        "details_json = excluded.details_json",
        (
            int(candidate_id), metric_key, status, value, model_ref, sample_count, reason,
            json.dumps(details, ensure_ascii=False) if details else None,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------

def session_dir(session_id: int) -> Path:
    return eval_session_dir(session_id)


def samples_root(session_id: int) -> Path:
    """出图根 —— 传给复用的 `eval_samples` 作为 eval_root。"""
    return eval_session_samples_dir(session_id)


def delete_session(conn: sqlite3.Connection, session_id: int) -> bool:
    """删一个 Session：DB 行（candidates / metric_results 走 ON DELETE CASCADE）+ 目录。

    不删对应的 tasks 行 —— 队列历史归队列管，用户删 task 时 task_dir 一并清（见
    queue/lifecycle）。checkpoint 更不动：plan 里只存路径引用。
    """
    row = conn.execute(
        "SELECT id FROM eval_sessions WHERE id = ?", (int(session_id),)
    ).fetchone()
    if row is None:
        return False
    conn.execute("DELETE FROM eval_sessions WHERE id = ?", (int(session_id),))
    conn.commit()
    import shutil
    d = session_dir(session_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def rollup_status(
    candidates: list[dict[str, Any]], results: dict[int, list[dict[str, Any]]]
) -> str:
    """候选 / 指标状态 → Session 整体状态。

    全成 → done；全败 → failed；有成有败 → partial（有结果可看，不该报成整体失败）。
    """
    if not candidates:
        return STATUS_FAILED
    if any(c.get("status") == STATUS_CANCELED for c in candidates):
        if not any(c.get("status") == STATUS_DONE for c in candidates):
            return STATUS_CANCELED
    ok = 0
    bad = 0
    for cand in candidates:
        cid = int(cand["id"])
        cand_failed = cand.get("status") in (STATUS_FAILED, STATUS_CANCELED)
        metric_states = [r.get("status") for r in results.get(cid, [])]
        if cand_failed or any(s == STATUS_FAILED for s in metric_states):
            bad += 1
        if not cand_failed and any(s == STATUS_DONE for s in metric_states):
            ok += 1
    if bad and not ok:
        return STATUS_FAILED
    if bad:
        return STATUS_PARTIAL
    return STATUS_DONE


def build_report(
    conn: sqlite3.Connection, session_id: int
) -> Optional[dict[str, Any]]:
    """从 DB 生成完成态报告（含 baseline Δ）。可随时重新生成，不是第二真相。"""
    session = get_session(conn, session_id)
    if session is None:
        return None
    candidates = list_candidates(conn, session_id)
    results = list_metric_results(conn, session_id)

    baseline_metrics: dict[str, float] = {}
    for cand in candidates:
        if cand.get("role") != "baseline":
            continue
        for r in results.get(int(cand["id"]), []):
            if r.get("status") == STATUS_DONE and r.get("value") is not None:
                baseline_metrics[str(r["metric_key"])] = float(r["value"])

    rows: list[dict[str, Any]] = []
    for cand in candidates:
        metrics: dict[str, Any] = {}
        for r in results.get(int(cand["id"]), []):
            key = str(r["metric_key"])
            value = float(r["value"]) if r.get("value") is not None else None
            base = baseline_metrics.get(key)
            metrics[key] = {
                "status": r.get("status"),
                "value": value,
                "delta": (value - base) if (value is not None and base is not None) else None,
                "reason": r.get("reason"),
                "model_ref": r.get("model_ref"),
                "sample_count": r.get("sample_count"),
            }
        rows.append({
            "candidate_id": int(cand["id"]),
            "role": cand.get("role"),
            "ordinal": cand.get("ordinal"),
            "checkpoint_path": cand.get("checkpoint_path"),
            "epoch": cand.get("epoch"),
            "step": cand.get("step"),
            "status": cand.get("status"),
            "samples_done": cand.get("samples_done"),
            "samples_total": cand.get("samples_total"),
            "run_id": cand.get("run_id"),
            "error": cand.get("error"),
            "metrics": metrics,
        })

    return {
        "session_id": int(session_id),
        "task_id": session.get("task_id"),
        "status": session.get("status"),
        "trigger": session.get("trigger"),
        "created_at": session.get("created_at"),
        "started_at": session.get("started_at"),
        "finished_at": session.get("finished_at"),
        "error": session.get("error"),
        "plan": session.get("plan") or {},
        "baseline_metrics": baseline_metrics,
        "candidates": rows,
    }


def session_results(
    conn: sqlite3.Connection, session_id: int
) -> Optional[list[dict[str, Any]]]:
    """Session 的候选 × 指标结果 → 前端既有的 `EvalMetricResult` 形状。

    评估面板的指标卡 / 曲线 / 表格早就按那个形状写好了（每个 checkpoint 一条，带
    metric_states / metrics / delta / sample_run）。映射一层比重写前端便宜，也让
    存量旧结果和新 Session 结果在同一个组件里显示一致。

    候选顺序即 plan 顺序（展示序）；`baseline` 条目照旧带 `baseline=True`，前端会把
    它从 checkpoint 列表里过滤掉、只用来画参考线。
    """
    session = get_session(conn, session_id)
    if session is None:
        return None
    plan = session.get("plan") or {}
    plan_by_ordinal = {
        int(c.get("ordinal", i)): c for i, c in enumerate(plan.get("candidates") or [])
    }
    candidates = list_candidates(conn, session_id)
    results = list_metric_results(conn, session_id)

    baseline_metrics: dict[str, float] = {}
    for cand in candidates:
        if cand.get("role") != "baseline":
            continue
        for r in results.get(int(cand["id"]), []):
            if r.get("status") == STATUS_DONE and r.get("value") is not None:
                baseline_metrics[str(r["metric_key"])] = float(r["value"])

    out: list[dict[str, Any]] = []
    for cand in candidates:
        cid = int(cand["id"])
        is_baseline = cand.get("role") == "baseline"
        meta = plan_by_ordinal.get(int(cand.get("ordinal") or 0)) or {}
        metric_states: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        delta: dict[str, float] = {}
        for r in results.get(cid, []):
            key = str(r["metric_key"])
            value = float(r["value"]) if r.get("value") is not None else None
            metric_states[key] = {
                "key": key,
                "status": r.get("status") or "not_run",
                "value": value,
                "reason": r.get("reason"),
                "model_name": r.get("model_ref"),
                "count": r.get("sample_count"),
            }
            if value is not None:
                metrics[key] = value
                base = baseline_metrics.get(key)
                if base is not None and not is_baseline:
                    delta[key] = value - base
        total = int(cand.get("samples_total") or 0)
        done = int(cand.get("samples_done") or 0)
        out.append({
            "schema_version": 1,
            "has_metrics": bool(metrics),
            "status": cand.get("status") or STATUS_PENDING,
            "run_id": str(cand.get("run_id") or f"candidate-{cid}"),
            "candidate_id": cid,
            "session_id": int(session_id),
            "project_id": session.get("project_id"),
            "version_id": session.get("version_id"),
            "created_at": session.get("created_at"),
            "updated_at": session.get("finished_at") or session.get("started_at"),
            "checkpoint": {
                "kind": meta.get("kind") or ("baseline" if is_baseline else "other"),
                "label": meta.get("label") or ("baseline" if is_baseline else ""),
                "path": cand.get("checkpoint_path"),
                "value": int(
                    cand.get("step") or cand.get("epoch") or meta.get("step")
                    or meta.get("epoch") or 0
                ),
            },
            "epoch": cand.get("epoch"),
            "step": cand.get("step"),
            "metrics": metrics,
            "metric_states": metric_states,
            "baseline": is_baseline,
            "delta": delta,
            "baseline_metrics": dict(baseline_metrics),
            "error": cand.get("error"),
            "sample_run": {
                "run_id": str(cand.get("run_id") or ""),
                "status": cand.get("status") or STATUS_PENDING,
                "summary": {"total": total, "done": done},
            },
        })
    return out


def write_report(conn: sqlite3.Connection, session_id: int) -> Optional[dict[str, Any]]:
    report = build_report(conn, session_id)
    if report is None:
        return None
    try:
        path = eval_session_report_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        logger.warning("failed writing report.json for session=%s", session_id, exc_info=True)
    return report


def cancel_active_sessions_for_task(conn: sqlite3.Connection, task_id: int) -> int:
    """取消某训练 task 名下未完成的 Session（对应的 task 行由调用方 cancel）。"""
    rows = conn.execute(
        "SELECT id FROM eval_sessions WHERE parent_task_id = ? AND status IN (?, ?)",
        (int(task_id), STATUS_PENDING, STATUS_RUNNING),
    ).fetchall()
    for row in rows:
        update_session(
            conn, int(row["id"]), status=STATUS_CANCELED, finished_at=time.time()
        )
    return len(rows)


def resource_summary(
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    *,
    selected_count: int,
    metric_keys: Iterable[str] | None = None,
    baseline: bool | None = None,
) -> dict[str, Any]:
    """创建前的规模摘要：出多少图、跑几个阶段（设计稿 §3.3「Review & Run」）。

    取代旧的「将创建 N 个后台任务」—— Session 模型下永远只有 1 个 task，成本改用
    出图数和阶段数表达。
    """
    cfg = secrets.load().eval_metrics
    keys = cfg.enabled_metrics if metric_keys is None else list(metric_keys)
    runners = eval_registry.enabled_runners(keys)
    use_baseline = bool(cfg.eval_baseline_enabled) if baseline is None else bool(baseline)
    validation_images = eval_validation.count_images(
        eval_validation.validation_dir(version_dir)
    )
    selected = max(0, int(selected_count))
    has_baseline = use_baseline and selected > 0
    candidates = selected + (1 if has_baseline else 0)
    return {
        "checkpoints_selected": selected,
        "baseline": has_baseline,
        "baseline_enabled": use_baseline,
        "candidates": candidates,
        "validation_images": validation_images,
        "images": candidates * validation_images,
        "metric_runners": runners,
        "metric_keys": sorted(eval_registry.normalize_enabled(keys)),
        # 1 个出图阶段 + 每个 runner 一个阶段，全在同一个 task 里顺序跑
        "stages": 1 + len(runners),
        "tasks": 1,
    }
