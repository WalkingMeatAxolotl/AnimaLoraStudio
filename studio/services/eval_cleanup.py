"""旧模型 eval 作业行的一次性清理 —— issue #465 的存量收尾。

## 清什么

**代际判据**：0.21 及以前，一次评估被拆成 `(候选数) × (1 出图 + N 指标)` 个子作业，
每条一行 tasks + 一个 `studio_data/tasks/<id>/` 目录（里面只有一个 run.log）。200 个
checkpoint 就是 603 条。这批就是清理对象。

    旧模型  task_type ∈ (eval_samples, eval_clip, eval_dino, eval_tag, eval_ccip)  → 清
    新模型  task_type = eval_session                                              → 留

新模型一次评估只产生一条 `eval_session` 行 —— 那是**正常的历史记录**，永久保留，
绝不在这里碰。

判据刻意**不看**「run 文件还在不在」：那个口径会把正常的历史记录当垃圾清掉（旧设计
每次重跑会删上一轮 run 文件、故意保留作业行）。按代际分才不会误伤。

## 为什么是一次性的

`eval_session` 上线后旧那五种 kind 再也不会产生，所以这件事有明确终点：升级后跑一次，
写个标记，之后不再扫。工具本身也该在存量清完的版本之后整体删掉 —— 那时这个文件、
它的标记键、和 lifespan 里的调用一起走。

## 一个已知取舍

旧评估的**指标数据**存在 `tasks/<训练 task id>/eval/samples/<run_id>/metrics.json` 里，
不依赖作业行，所以清理后旧结果的指标仍然读得到。但日志关联（`/eval/jobs`）依赖作业行，
清理后**旧结果看不到日志了**。历史结果的日志价值很低，而这批日志正是 #465 抱怨的东西。
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

from studio.infrastructure import db
from studio.infrastructure.paths import task_dir

logger = logging.getLogger(__name__)

# 代际判据：旧模型（0.21 及以前）一次评估散成几百条子作业行，每条一个日志目录。
LEGACY_EVAL_TASK_TYPES = db.LEGACY_EVAL_TASK_TYPES
# 只清终态。_v19 迁移已把升级瞬间残留的 pending/running 收成 canceled；万一还有非终态
# 的，留着让用户看见比默默删掉好。
TERMINAL_STATUSES = ("done", "failed", "canceled")
# SQLite 默认变量上限 999 —— DELETE 分批走，几千条也不会撞上限。
_DELETE_CHUNK = 400
# 「存量已清」标记。queue_settings 是既有的跨重启 kv 表（ADR 0006 PR-2 引入）。
_FLAG_KEY = "eval.legacy_cleanup_done"


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def scan_legacy_jobs(
    conn: sqlite3.Connection, *, with_size: bool = True
) -> dict[str, Any]:
    """扫出所有旧模型 eval 作业行。纯读，不改任何东西。

    `with_size=False` 跳过目录大小统计 —— 那是递归 stat，几千个目录会明显拖慢；
    启动期自动清理不需要报数给用户，只有交互式确认才需要。
    """
    kind_ph = ",".join("?" for _ in LEGACY_EVAL_TASK_TYPES)
    status_ph = ",".join("?" for _ in TERMINAL_STATUSES)
    rows = conn.execute(
        f"SELECT id, task_type, status, project_id, version_id FROM tasks "
        f"WHERE task_type IN ({kind_ph}) AND status IN ({status_ph}) ORDER BY id",
        (*LEGACY_EVAL_TASK_TYPES, *TERMINAL_STATUSES),
    ).fetchall()

    jobs: list[dict[str, Any]] = []
    for row in rows:
        jid = int(row["id"])
        d = task_dir(jid)
        exists = d.exists()
        jobs.append({
            "id": jid,
            "kind": row["task_type"],
            "status": row["status"],
            "project_id": int(row["project_id"]) if row["project_id"] else None,
            "version_id": int(row["version_id"]) if row["version_id"] else None,
            "dir": str(d),
            "dir_exists": exists,
            "bytes": _dir_size(d) if (exists and with_size) else 0,
        })

    return {
        "count": len(jobs),
        "bytes": sum(int(j["bytes"]) for j in jobs),
        "dirs": sum(1 for j in jobs if j["dir_exists"]),
        "jobs": jobs,
    }


def purge_legacy_jobs(
    conn: sqlite3.Connection,
    ids: Optional[Iterable[int]] = None,
    *,
    with_size: bool = True,
) -> dict[str, Any]:
    """删旧模型 eval 作业的 `tasks/<id>/` 目录 + tasks 表行。

    `ids` 给定时只清其中确实属于旧模型的那些（交集）—— 永远**重新扫描**再取交集，
    调用方传过期或伪造的 id 都碰不到 `eval_session` 行。
    """
    scan = scan_legacy_jobs(conn, with_size=with_size)
    targets: list[dict[str, Any]] = scan["jobs"]
    if ids is not None:
        allow = {int(i) for i in ids}
        targets = [j for j in targets if int(j["id"]) in allow]
    if not targets:
        return {"removed_rows": 0, "removed_dirs": 0, "freed_bytes": 0, "ids": []}

    removed_dirs = 0
    freed = 0
    for job in targets:
        d = Path(str(job["dir"]))
        if not d.exists():
            continue
        shutil.rmtree(d, ignore_errors=True)
        if d.exists():
            logger.warning("legacy eval job dir not fully removed: path=%s", d)
            continue
        removed_dirs += 1
        freed += int(job["bytes"])

    ids_to_delete = [int(j["id"]) for j in targets]
    removed_rows = 0
    for start in range(0, len(ids_to_delete), _DELETE_CHUNK):
        chunk = ids_to_delete[start:start + _DELETE_CHUNK]
        ph = ",".join("?" for _ in chunk)
        cur = conn.execute(f"DELETE FROM tasks WHERE id IN ({ph})", chunk)
        removed_rows += cur.rowcount
    conn.commit()

    logger.info(
        "legacy eval cleanup: job_rows=%s log_dirs=%s freed=%.1f MB",
        removed_rows, removed_dirs, freed / 1_048_576,
    )
    return {
        "removed_rows": removed_rows,
        "removed_dirs": removed_dirs,
        "freed_bytes": freed,
        "ids": ids_to_delete,
    }


# ---------------------------------------------------------------------------
# 一次性执行标记
# ---------------------------------------------------------------------------

def already_done(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT value FROM queue_settings WHERE key = ?", (_FLAG_KEY,)
    ).fetchone()
    return row is not None and str(row[0]).lower() == "true"


def mark_done(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO queue_settings(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (_FLAG_KEY, "true"),
    )
    conn.commit()


def cleanup_legacy_eval_on_startup(conn: sqlite3.Connection) -> dict[str, Any]:
    """启动期清一次旧模型 eval 作业存量，然后写标记不再重复。

    只有从 0.21 及以前升级上来的库才有存量；干净库跑一次扫描（很快）就写标记退场。
    标记写在扫描**之后**：中途崩了下次启动会重来，不会留一半。
    """
    if already_done(conn):
        return {"skipped": True, "reason": "already done"}
    result = purge_legacy_jobs(conn, with_size=False)
    mark_done(conn)
    return {"skipped": False, **result}
