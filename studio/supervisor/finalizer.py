"""task 终态 → version.status 映射（PR-4 从 supervisor.py 抽出）。

ADR-0007 §11.3-B：task 终态独立映射，不撒谎。done/failed/canceled 三种
task_status 各映射到对应 VersionStatus；paused 不进此函数（§11.3-A：
task=paused 时 version 仍 training，UI 派生显示）。
"""
from __future__ import annotations

from typing import Any

from .. import db


def _maybe_finalize_version(
    conn: Any, task_id: int, task_status: str = "done"
) -> None:
    """task 终态 → 推 version.status（ADR-0007 §11.3-B）。

    task_status 映射：
    - done → completed（+ output_lora_path 回填）
    - failed → failed（+ last_failure_reason 写入 task.error_msg）
    - canceled → canceled

    paused 不进此函数（§11.3-A：task=paused 时 version 仍 training，UI 派生显示）。
    """
    from ..services.projects import versions as _versions
    task_row = db.get_task(conn, task_id)
    if not task_row:
        return
    vid = task_row.get("version_id")
    pid = task_row.get("project_id")
    if not (vid and pid):
        return
    v = _versions.get_version(conn, int(vid))
    if not v:
        return
    from ..services.projects import projects as _projects
    p = _projects.get_project(conn, int(pid))
    if not p:
        return

    # ADR-0007 §11.3-B：task 终态独立映射，不撒谎
    new_status_map = {
        "done":     _versions.VersionStatus.COMPLETED,
        "failed":   _versions.VersionStatus.FAILED,
        "canceled": _versions.VersionStatus.CANCELED,
    }
    new_status = new_status_map.get(task_status)
    if new_status is None:
        return  # 未知 task_status（如 paused / running）不动 version

    fields: dict[str, Any] = {"status": new_status}

    if task_status == "done":
        output_name = f"{p['slug']}_{v['label']}"
        vdir = _versions.version_dir(int(pid), p["slug"], v["label"])
        candidate = vdir / "output" / f"{output_name}_final.safetensors"
        if candidate.exists():
            fields["output_lora_path"] = str(candidate)
    elif task_status == "failed":
        err = task_row.get("error_msg")
        if err:
            fields["last_failure_reason"] = str(err)

    _versions.update_version(conn, int(vid), **fields)
