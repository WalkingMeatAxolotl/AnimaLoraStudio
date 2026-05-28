"""projects/ 子包共用 helpers（PR-6.5 从 server.py 抽出）。

只服务 projects 子包内部的各 sub-router；非 projects domain 的 router
不应该 import 这里（避免反向 / 横向耦合）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .... import db
from ....services.projects import projects, versions
from ....event_bus import bus
from ....services import version_config


def _project_err_code(exc: Exception) -> int:
    """ProjectError / VersionError → HTTP 状态码。"""
    msg = str(exc)
    if "不存在" in msg:
        return 404
    if "已存在" in msg or "非法" in msg or "不能为空" in msg:
        return 400
    return 422


def _project_payload(p: dict[str, Any]) -> dict[str, Any]:
    """对外详情 payload：项目本身 + versions[] 含 stats + download stats。"""
    out = dict(p)
    out.update(projects.stats_for_project(p))
    with db.connection_for() as conn:
        vs = versions.list_versions(conn, p["id"])
    out["versions"] = [
        {**v, "stats": versions.stats_for_version(p, v)} for v in vs
    ]
    return out


def _publish_project_state(p: dict[str, Any]) -> None:
    bus.publish({
        "type": "project_state_changed",
        "project_id": p["id"],
    })


def _publish_version_state(v: dict[str, Any]) -> None:
    bus.publish({
        "type": "version_state_changed",
        "project_id": v["project_id"],
        "version_id": v["id"],
        "status": versions.get_status(v),
        "phase": versions.get_phase(v),
    })


def _publish_job_state(job: dict[str, Any]) -> None:
    bus.publish({
        "type": "job_state_changed",
        "job_id": job["id"],
        "project_id": job["project_id"],
        "version_id": job.get("version_id"),
        "kind": job["kind"],
        "status": job["status"],
    })


def _version_dir_or_404(pid: int, vid: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """返回 (project, version, version_dir)。"""
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        p = projects.get_project(conn, pid)
    assert p is not None
    return p, v, versions.version_dir(p["id"], p["slug"], v["label"])


def _version_train_dir_or_404(pid: int, vid: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """返回 (project, version, version_dir/train)。"""
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        p = projects.get_project(conn, pid)
    assert p is not None
    return p, v, versions.version_dir(p["id"], p["slug"], v["label"]) / "train"


def _project_and_version_or_404(
    pid: int, vid: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """version_config domain 的双对象解析（含 VersionConfigError → 404 包装）。"""
    with db.connection_for() as conn:
        try:
            return version_config.get_project_and_version(conn, pid, vid)
        except version_config.VersionConfigError as exc:
            raise HTTPException(404, str(exc)) from exc


def _reg_dir(vdir: Path) -> Path:
    """reg 根目录 — 子目录直接镜像 train 子文件夹（与源脚本一致，无 1_general 中间层）。"""
    return vdir / "reg"
