"""Projects + Versions CRUD + phase 推进 + ckpts（PR-6.5 commit 1 从 server.py 抽出）。

18 routes：
    GET    /api/projects                                       list（含 active version enrich；含已归档行）
    POST   /api/projects                                       create（可选 initial version）
    GET    /api/projects/{pid}                                 get（含 versions 列表）
    PATCH  /api/projects/{pid}                                 update
    DELETE /api/projects/{pid}                                 delete
    POST   /api/projects/{pid}/archive                         归档（软隐藏，可逆）
    POST   /api/projects/{pid}/unarchive                       取消归档
    GET    /api/projects/{pid}/versions                        list versions
    POST   /api/projects/{pid}/versions                        create version
    GET    /api/projects/{pid}/versions/{vid}                  get version
    PATCH  /api/projects/{pid}/versions/{vid}                  update version
    DELETE /api/projects/{pid}/versions/{vid}                  delete version
    POST   /api/projects/{pid}/versions/{vid}/activate         activate
    POST   /api/projects/{pid}/versions/{vid}/advance-phase    ADR-0007 §11.5-A
    POST   /api/projects/{pid}/versions/{vid}/skip-phase       ADR-0007 §11.5-B
    GET    /api/projects/{pid}/versions/{vid}/lora_ckpts       LoRA picker 第二层（XY ckpt 轴）
    GET    /api/projects/{pid}/state_ckpts                     resume_state picker（按 version 分组）
    GET    /api/projects/{pid}/lora_ckpts                      resume_lora picker（按 version 分组）
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from ...schemas.projects import (
    ProjectCreate,
    ProjectUpdate,
    VersionCreate,
    VersionUpdate,
)
from ._shared import (
    _project_err_code,
    _project_payload,
    _publish_project_state,
    _publish_version_state,
    _version_dir_or_404,
)
from .... import db
from ....services.projects import projects, versions, phase as versions_phase

router = APIRouter()


@router.get("/api/projects")
def list_projects_endpoint() -> dict[str, Any]:
    """ADR-0007 §11.8-E：enrich active version label + status，卡片右上角 badge 用。

    v12 起额外带 archived_at（归档行不在这里过滤 —— 项目量小，归档/活跃的
    切分交给前端，省一次切视图的往返）+ active_version_phase（preparing 时
    badge 显示当前 phase，如"准备中 · 打标"）。
    """
    with db.connection_for() as conn:
        rows = projects.list_projects(conn)
        enriched: list[dict[str, Any]] = []
        for r in projects.projects_with_stats(rows):
            r["active_version_label"] = None
            r["active_version_status"] = None
            r["active_version_phase"] = None
            avid = r.get("active_version_id")
            if avid:
                av = versions.get_version(conn, int(avid))
                if av:
                    r["active_version_label"] = av["label"]
                    r["active_version_status"] = versions.get_status(av)
                    r["active_version_phase"] = versions.get_phase(av)
            enriched.append(r)
    return {"items": enriched}


@router.post("/api/projects")
def create_project_endpoint(body: ProjectCreate) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            p = projects.create_project(
                conn, title=body.title, slug=body.slug, note=body.note
            )
        except projects.ProjectError as exc:
            _project_err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
        if body.initial_version_label:
            try:
                versions.create_version(
                    conn, project_id=p["id"], label=body.initial_version_label
                )
            except versions.VersionError as exc:
                # 项目已建好；版本失败给前端但保留项目
                _project_err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
        p = projects.get_project(conn, p["id"])
    assert p is not None
    _publish_project_state(p)
    return _project_payload(p)


@router.get("/api/projects/{pid}")
def get_project_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    return _project_payload(p)


@router.patch("/api/projects/{pid}")
def patch_project_endpoint(pid: int, body: ProjectUpdate) -> dict[str, Any]:
    fields = body.model_dump(exclude_unset=True)
    with db.connection_for() as conn:
        try:
            p = projects.update_project(conn, pid, **fields)
        except projects.ProjectError as exc:
            _project_err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    _publish_project_state(p)
    return _project_payload(p)


@router.post("/api/projects/{pid}/archive")
def archive_project_endpoint(pid: int) -> dict[str, Any]:
    """归档：列表软隐藏，目录 / versions / 任务全部原样，可随时取消。"""
    with db.connection_for() as conn:
        try:
            p = projects.set_archived(conn, pid, True)
        except projects.ProjectError as exc:
            _project_err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    _publish_project_state(p)
    return _project_payload(p)


@router.post("/api/projects/{pid}/unarchive")
def unarchive_project_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            p = projects.set_archived(conn, pid, False)
        except projects.ProjectError as exc:
            _project_err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    _publish_project_state(p)
    return _project_payload(p)


@router.delete("/api/projects/{pid}")
def delete_project_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        try:
            projects.delete_project(conn, pid)
        except projects.ProjectError as exc:
            _project_err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    return {"deleted": pid}


# Versions ------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions")
def list_versions_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        vs = versions.list_versions(conn, pid)
        p = projects.get_project(conn, pid)
    assert p is not None
    return {
        "items": [
            {**v, "stats": versions.stats_for_version(p, v)} for v in vs
        ]
    }


@router.get("/api/projects/{pid}/versions/{vid}/lora_ckpts")
def list_version_lora_ckpts(pid: int, vid: int) -> dict[str, Any]:
    """列出 version output/ 下所有 .safetensors（step / epoch / final），
    用于 LoRA picker 第二层（XY ckpt 轴 + 单图模式切 ckpt）。"""
    p, v, vdir = _version_dir_or_404(pid, vid)
    return {"items": versions.list_lora_ckpts(vdir)}


@router.get("/api/projects/{pid}/state_ckpts")
def list_project_state_ckpts(pid: int) -> dict[str, Any]:
    """列出项目所有 versions 的 training_state_step*.pt，按 version 分组。

    给 Train 页 resume_state 字段的「浏览本项目」picker 用：用户看 version
    分组的语义化文件列表，选中后前端把绝对路径写入字段。
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise HTTPException(404, f"项目不存在: id={pid}")
        return {"groups": versions.list_project_state_ckpts(conn, p)}


@router.get("/api/projects/{pid}/lora_ckpts")
def list_project_lora_ckpts(pid: int) -> dict[str, Any]:
    """列出项目所有 versions 的 LoRA ckpt（.safetensors），按 version 分组。

    给 Train 页 resume_lora 字段的「浏览本项目」picker 用。
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise HTTPException(404, f"项目不存在: id={pid}")
        return {"groups": versions.list_project_lora_ckpts(conn, p)}


@router.post("/api/projects/{pid}/versions")
def create_version_endpoint(pid: int, body: VersionCreate) -> dict[str, Any]:
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        try:
            v = versions.create_version(
                conn,
                project_id=pid,
                label=body.label,
                fork_from_version_id=body.fork_from_version_id,
                note=body.note,
            )
        except versions.VersionError as exc:
            _project_err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    _publish_version_state(v)
    return v


@router.get("/api/projects/{pid}/versions/{vid}")
def get_version_endpoint(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        p = projects.get_project(conn, pid)
    if not v or v["project_id"] != pid:
        raise HTTPException(404, f"版本不存在: id={vid}")
    assert p is not None
    return {**v, "stats": versions.stats_for_version(p, v)}


@router.patch("/api/projects/{pid}/versions/{vid}")
def patch_version_endpoint(
    pid: int, vid: int, body: VersionUpdate,
) -> dict[str, Any]:
    fields = body.model_dump(exclude_unset=True)
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        try:
            v = versions.update_version(conn, vid, **fields)
        except versions.VersionError as exc:
            _project_err_code(exc); raise  # PR-2 C4: DomainError handler 翻 envelope
    _publish_version_state(v)
    return v


@router.delete("/api/projects/{pid}/versions/{vid}")
def delete_version_endpoint(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        versions.delete_version(conn, vid)
    return {"deleted": vid}


@router.post("/api/projects/{pid}/versions/{vid}/activate")
def activate_version_endpoint(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        v = versions.activate_version(conn, vid)
        p = projects.get_project(conn, pid)
    assert p is not None
    _publish_project_state(p)
    return _project_payload(p)


# ---------------------------------------------------------------------------
# Phase cursor 推进 / 跳过 — ADR-0007 §11.5-A / §11.5-B
# ---------------------------------------------------------------------------


def _phase_advance_payload(
    advanced: bool, result: versions_phase.CheckResult,
    new_phase: Optional[str], version: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "advanced": advanced,
        "ok": result.ok,
        "reason": result.reason,
        "new_phase": new_phase,
        "version": version,
    }


@router.post("/api/projects/{pid}/versions/{vid}/advance-phase")
def advance_phase_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """phase cursor 推进 —— "下一步" 按钮调用（ADR-0007 §11.5-A）。

    成功 → cursor++ + 返回新 phase + publish version_state_changed。
    失败 → ok=False + reason（前端 toast），cursor 不动。
    """
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        advanced, result, new_phase = versions_phase.advance_phase(conn, vid)
        v_after = versions.get_version(conn, vid)
    if advanced and v_after is not None:
        _publish_version_state(v_after)
    return _phase_advance_payload(advanced, result, new_phase, v_after)


@router.post("/api/projects/{pid}/versions/{vid}/skip-phase")
def skip_phase_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """跳过可跳过的 phase（当前仅 regularizing；ADR-0007 §11.5-A）。"""
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        advanced, result, new_phase = versions_phase.skip_phase(conn, vid)
        v_after = versions.get_version(conn, vid)
    if advanced and v_after is not None:
        _publish_version_state(v_after)
    return _phase_advance_payload(advanced, result, new_phase, v_after)
