"""文件管理 + curation + 去重（PR-6.5 commit 4 从 server.py 抽出）。

routes：
    POST /api/projects/{pid}/files/delete                                download/ 文件 + caption metadata
    GET  /api/projects/{pid}/files                                       download/ 列表
    GET  /api/projects/{pid}/thumb                                       缩略图（含 manifest resolve）
    GET  /api/projects/{pid}/versions/{vid}/jobs/latest                  hydrate latest job + log
    GET  /api/projects/{pid}/versions/{vid}/curation                     curation_view（train/ 内容 + download/ 剩余）
    POST /api/projects/{pid}/versions/{vid}/preprocess/duplicates/scan   train scope 去重扫描 + SSE 进度
    POST /api/projects/{pid}/versions/{vid}/preprocess/duplicates/apply  标记 manifest duplicate_removed
    POST /api/projects/{pid}/versions/{vid}/curation/copy                download → train/{folder}
    POST /api/projects/{pid}/versions/{vid}/curation/remove              train/{folder} → 删
    POST /api/projects/{pid}/versions/{vid}/curation/folder              create/rename/delete folder
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...errors import _safe_join_or_400
from ...responses import _thumb_response
from ....domain.errors import DomainError, NotFoundError, ValidationError
from ...schemas.curation import (
    CopyRequest,
    CopyValidationRequest,
    DeleteFilesRequest,
    DuplicateApplyRequest,
    DuplicateScanRequest,
    FolderOp,
    RemoveRequest,
    RemoveValidationRequest,
)
from ._shared import _publish_project_state
from .... import db
from ....services.projects import jobs as project_jobs, projects, versions
from ....services.dataset import curation, scan as datasets
from ....infrastructure.event_bus import bus
from ....services.preprocess import duplicates as duplicate_finder, manifest as preprocess_manifest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/projects/{pid}/files/delete")
def delete_project_files(
    pid: int, body: DeleteFilesRequest,
) -> dict[str, Any]:
    """从项目 `download/` 删除指定文件（含同名 caption metadata）。

    metadata 命名约定：
    - booru 下载会写 `{stem}.booru.txt`
    - tag/caption 流程可能写 `{stem}.txt` 或 `{stem}.json`
    都一并清理；不存在的扩展静默跳过。
    """
    if not body.names:
        return {"deleted": [], "missing": []}
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found", details={"id": pid},
        )
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    if not pdir.exists():
        return {"deleted": [], "missing": list(body.names)}

    META_EXTS = (".booru.txt", ".txt", ".json")
    deleted: list[str] = []
    missing: list[str] = []
    for name in body.names:
        f = _safe_join_or_400(pdir, name)
        if not f.exists() or not f.is_file():
            missing.append(name)
            continue
        try:
            f.unlink()
        except OSError as exc:
            raise DomainError(
                f'Failed to delete "{name}": {exc}',
                code="dataset.delete_failed",
                details={"name": name, "reason": str(exc)},
                http_status=500,
            ) from exc
        # 清理同 stem 的 metadata（best-effort，失败仅日志）
        stem = f.stem
        for ext in META_EXTS:
            m = pdir / f"{stem}{ext}"
            if m.exists():
                try:
                    m.unlink()
                except OSError as exc:
                    logger.warning("删 metadata 失败 %s: %s", m, exc)
        deleted.append(name)
    return {"deleted": deleted, "missing": missing}


@router.get("/api/projects/{pid}/files")
def list_files(pid: int, bucket: str = "download") -> dict[str, Any]:
    if bucket != "download":
        raise ValidationError(
            "Unsupported image set",
            code="dataset.image_set_invalid",
            details={"bucket": bucket}, http_status=400,
        )
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found", details={"id": pid},
        )
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    items: list[dict[str, Any]] = []
    if pdir.exists():
        for f in sorted(pdir.iterdir()):
            if f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS:
                items.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "has_meta": f.with_suffix(".booru.txt").exists(),
                })
    return {"items": items, "count": len(items)}


@router.get("/api/projects/{pid}/thumb")
def project_thumb(
    pid: int,
    bucket: str = "download",
    name: str = "",
    size: int = 256,
    raw: int = 0,
) -> FileResponse:
    """缩略图：默认 256px JPEG（缓存）；size=0 → 原图。

    两种 bucket：
      - `bucket=download`（默认）：`name` 是 download/ 下的原始文件名。
        后端通过 `preprocess_manifest.resolve_origin()` 决定实际字节路径：
        未处理 → download/{name}，已处理 → preprocess/ 下第一个 origin 匹配
        的派生。前端"按 download 名"调用时不需要感知预处理。
      - `bucket=preprocess`：`name` 是 preprocess/ 下的**实际产物文件名**
        （含 multi-crop 派生的 _c0 / _c1 后缀）。直接按文件名取，**不走**
        resolve_origin —— multi-crop 后多个产物共享同一 origin，按 origin
        永远落到 [0] 是 bug。裁剪 / 总览页应该走这条来精确寻址。

    `raw=1`（仅 bucket=download）：跳过 resolve_origin，强制读 download/{name}
    原始字节。给「对比预览」场景用：左 pane 永远要 download 原图，不能被
    preprocess 派生 hijack。

    缓存路径：`studio_data/thumb_cache/{sha1(src+mtime+size)}.jpg`。
    源文件 mtime 变化会自动 invalidate（hash 变）。
    """
    if bucket not in ("download", "preprocess"):
        raise ValidationError(
            "Unsupported image set",
            code="dataset.image_set_invalid",
            details={"bucket": bucket}, http_status=400,
        )
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found", details={"id": pid},
        )
    pdir = projects.project_dir(p["id"], p["slug"])

    if bucket == "preprocess":
        # Direct addressing — no resolve. Path traversal guard against the
        # actual preprocess/ dir (any filename including _c0/_c1 derivatives).
        _safe_join_or_400(pdir / "preprocess", name)
        f = pdir / "preprocess" / name
    elif raw:
        # bucket=download + raw=1: bypass resolve_origin, hand back the
        # untouched download/{name} bytes. Used by the processed-tab compare
        # preview left pane (need the original, not the derivative).
        _safe_join_or_400(pdir / "download", name)
        f = pdir / "download" / name
    else:
        # bucket=download — historical behavior: address by download name,
        # resolve to first preprocess product if any (1:1 / multi-crop cases).
        # duplicate_removed origins: resolve_origin returns [] but the original
        # file in download/ still exists; the Download page must keep showing
        # it (软删除 ≠ 不可见). Fall back to download/{name} like any other
        # un-resolved origin.
        _safe_join_or_400(pdir / "download", name)
        candidates = preprocess_manifest.resolve_origin(pdir, name)
        f = candidates[0] if candidates else (pdir / "download" / name)
        # Curation passes multi-crop derivative names (X_c0.png) through this
        # endpoint with bucket=download. resolve_origin only matches by origin,
        # not by entry key, so derivatives miss → f points at a non-existent
        # download/X_c0.png. Fall back: if the name IS a preprocess entry key,
        # serve preprocess/{name} directly. Filename was already safety-checked
        # against download/, same validation applies to preprocess/.
        if not f.exists() and preprocess_manifest.get_entry(pdir, name) is not None:
            f = pdir / "preprocess" / name

    if not f.exists() or f.suffix.lower() not in datasets.IMAGE_EXTS:
        logger.info("thumb 404: pid=%s bucket=%s name=%s -> %s", pid, bucket, name, f)
        raise NotFoundError(
            "Thumbnail not found", code="dataset.thumbnail_not_found",
        )
    return _thumb_response(f, size)


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/jobs/latest（hydrate）
# /api/jobs/{jid} / log / cancel 在 PR-6 commit 2 抽到 api/routers/jobs.py；
# 本 endpoint 因为路径在 /api/projects/ 下，归 projects 子包。
# ---------------------------------------------------------------------------

_HYDRATABLE_JOB_KINDS = {
    "download",
    "tag",
    "reg_build",
    "eval_samples",
    "eval_clip",
    "eval_dino",
    "eval_tag",
}


@router.get("/api/projects/{pid}/versions/{vid}/jobs/latest")
def get_latest_version_job(pid: int, vid: int, kind: str) -> dict[str, Any]:
    """页面刷新 hydrate 用：返回该 version 下指定 kind 的最近一条 job + 全量日志。

    Tagging / Regularization 页之前只在本会话 startBuild 后才知道 jid，刷新一下
    就丢了；这里给个起点让前端 mount 时锁回 jid + 回放历史日志，SSE 继续接力。
    `job` 可能是 running / pending / 已完成；前端按 status 决定要不要继续等事件。
    """
    if kind not in _HYDRATABLE_JOB_KINDS:
        raise HTTPException(400, f"unknown kind: {kind}")
    with db.connection_for() as conn:
        job = project_jobs.latest_for(
            conn, project_id=pid, kind=kind, version_id=vid,
        )
    if not job:
        return {"job": None, "log": ""}
    log_path = Path(job.get("log_path") or "")
    log = ""
    if log_path.exists():
        try:
            log = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            log = ""
    return {"job": job, "log": log}


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/curation  (PP3)
# ---------------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions/{vid}/curation")
def get_curation(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        return curation.curation_view(conn, pid, vid)


# ---------------------------------------------------------------------------
# held-out 验证集手动维护（与 train curation 对称，但右栏扁平、无文件夹）
# ---------------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions/{vid}/curation/validation")
def get_curation_validation(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        return curation.curation_validation_view(conn, pid, vid)


@router.post("/api/projects/{pid}/versions/{vid}/curation/validation/copy")
def copy_to_validation(
    pid: int, vid: int, body: CopyValidationRequest,
) -> dict[str, Any]:
    """download → validation/1_data/ 复制（连 caption）；已在 train/validation
    的名字 skip 防泄漏。"""
    with db.connection_for() as conn:
        return curation.copy_download_to_validation(conn, pid, vid, body.files)


@router.post("/api/projects/{pid}/versions/{vid}/curation/validation/remove")
def remove_from_validation(
    pid: int, vid: int, body: RemoveValidationRequest,
) -> dict[str, Any]:
    with db.connection_for() as conn:
        return curation.remove_from_validation(
            conn, pid, vid, [item.model_dump() for item in body.items],
        )


# ---------------------------------------------------------------------------
# ADR 0010 — train-scope duplicates endpoint
#
# scope 收窄到 versions/{label}/train/，对应 scan_train_duplicates /
# apply_train_duplicate_removals。
# ---------------------------------------------------------------------------


def _resolve_pv_or_404_dup(
    pid: int, vid: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise NotFoundError(
                "Project not found", code="project.not_found", details={"id": pid},
            )
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found", details={"id": vid},
            )
    return p, v


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/duplicates/scan")
def scan_preprocess_duplicates_train(
    pid: int, vid: int, body: DuplicateScanRequest,
) -> dict[str, Any]:
    """ADR 0010 train scope 重复扫描。sources = versions/{label}/train/
    （跳过 manifest 已标 duplicate_removed 的）；返回结构跟老 endpoint 一致，
    `target` 字段从 `"preprocess"` 改成 `"train"`。"""
    _resolve_pv_or_404_dup(pid, vid)
    with db.connection_for() as conn:
        try:
            options = duplicate_finder.options_from_payload(body.model_dump())
            last_progress_at = 0.0

            def publish_progress(payload: dict[str, Any]) -> None:
                nonlocal last_progress_at
                now = time.monotonic()
                if now - last_progress_at < 1.0:
                    return
                last_progress_at = now
                bus.publish({
                    "type": "duplicate_scan_progress",
                    "project_id": pid,
                    "version_id": vid,
                    "status": "running",
                    **payload,
                })

            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "version_id": vid,
                "status": "running",
                "text": "Scanning duplicate candidates in train set...",
            })
            result = duplicate_finder.scan_train_duplicates(
                conn, pid, vid, options, on_progress=publish_progress,
            )
            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "version_id": vid,
                "status": "done",
                "total_images": result["total_images"],
                "group_count": result["group_count"],
                "candidate_count": result["candidate_count"],
                "blur_candidate_count": result.get("blur_candidate_count", 0),
                "crop_relation_count": result.get("crop_relation_count", 0),
                "elapsed_seconds": result["elapsed_seconds"],
                "text": (
                    f"Scanned {result['total_images']} train images; "
                    f"found {result['group_count']} groups / "
                    f"{result['candidate_count']} candidates, "
                    f"{result.get('blur_candidate_count', 0)} blur candidates, "
                    f"{result.get('crop_relation_count', 0)} crop relations."
                ),
            })
            return result
        except curation.CurationError as exc:
            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "version_id": vid,
                "status": "failed",
                "text": str(exc),
            })
            raise
        except duplicate_finder.DuplicateFinderError as exc:
            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "version_id": vid,
                "status": "failed",
                "text": str(exc),
            })
            raise


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/duplicates/apply")
def apply_preprocess_duplicates_train(
    pid: int, vid: int, body: DuplicateApplyRequest,
) -> dict[str, Any]:
    """ADR 0010 train scope: 标记 per-version 审核状态到 train manifest。
    `names` 是 train rel path（`"1_data/X.png"`）。物理文件不动。"""
    _resolve_pv_or_404_dup(pid, vid)
    with db.connection_for() as conn:
        result = duplicate_finder.apply_train_duplicate_removals(
            conn, pid, vid, names=body.names,
        )
        project = projects.get_project(conn, pid)
    if project:
        _publish_project_state(project)
    return result


@router.post("/api/projects/{pid}/versions/{vid}/curation/copy")
def copy_to_train(
    pid: int, vid: int, body: CopyRequest,
) -> dict[str, Any]:
    """ADR 0010 fixup（2026-06-04）：endpoint 切到 `copy_download_to_train`
    简化版——纯 download → train 复制 + 写 train manifest entry。不再走
    "preprocess 派生 vs download 原图"双分支 / 不再 check 老
    `duplicate_removed`（PR-4 决议 dedupe 下沉 train scope，老标记不影响
    Curation 操作）。前端 Curation 选 download 原图加入 train 直接 work。
    """
    with db.connection_for() as conn:
        result = curation.copy_download_to_train(
            conn, pid, vid, body.files, body.dest_folder,
        )
    return result


@router.post("/api/projects/{pid}/versions/{vid}/curation/remove")
def remove_from_train(
    pid: int, vid: int, body: RemoveRequest,
) -> dict[str, Any]:
    with db.connection_for() as conn:
        result = curation.remove_from_train(
            conn, pid, vid, body.folder, body.files,
        )
    return result


@router.post("/api/projects/{pid}/versions/{vid}/curation/folder")
def folder_op(
    pid: int, vid: int, body: FolderOp,
) -> dict[str, Any]:
    with db.connection_for() as conn:
        if body.op == "create":
            p = curation.create_folder(conn, pid, vid, body.name)
            return {"path": str(p)}
        if body.op == "rename":
            if not body.new_name:
                raise ValidationError(
                    "A new name is required to rename a folder",
                    code="curation.new_name_required", http_status=400,
                )
            p = curation.rename_folder(
                conn, pid, vid, body.name, body.new_name,
            )
            return {"path": str(p)}
        if body.op == "delete":
            curation.delete_folder(conn, pid, vid, body.name)
            return {"deleted": body.name}
        raise ValidationError(
            f"Unknown folder operation: {body.op}",
            code="curation.op_invalid", details={"op": body.op}, http_status=400,
        )
