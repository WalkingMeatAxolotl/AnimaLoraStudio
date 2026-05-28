"""图片获取 + 预处理（PR-6.5 commit 3 从 server.py 抽出）。

14 routes：

  下载 / 上传 (5)
    POST /api/projects/{pid}/download/estimate    booru count API 估算
    POST /api/projects/{pid}/download             启动 booru 下载 job
    POST /api/projects/{pid}/upload               多文件本地上传 (单图 / zip)
    POST /api/projects/{pid}/upload-from-path     服务端可见路径导入单图 / zip
    GET  /api/projects/{pid}/download/status      最近 download job + log_tail

  预处理 (9)
    POST /api/projects/{pid}/preprocess/start
    GET  /api/projects/{pid}/preprocess/status
    GET  /api/projects/{pid}/preprocess/files
    GET  /api/projects/{pid}/preprocess/duplicates/removed
    GET  /api/projects/{pid}/preprocess/crop/workspace
    POST /api/projects/{pid}/preprocess/crop
    POST /api/projects/{pid}/preprocess/files/reset
    POST /api/projects/{pid}/preprocess/files/restore
    GET  /api/projects/{pid}/preprocess/thumb     [Deprecated] 兼容旧 URL

注：duplicates scan / apply（preprocess 子域）属于 commit 4（curation），不在本文件。
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ...errors import _validate_component_or_400  # noqa: F401  reserved for future use
from ...responses import _thumb_response
from ...schemas.ingestion import (
    DownloadRequest,
    EstimateRequest,
    PreprocessCropRequest,
    PreprocessRestoreRequest,
    PreprocessStartRequest,
    UploadFromPathBody,
)
from ._shared import _publish_job_state, _publish_project_state
from .... import db, secrets
from ....services.projects import jobs as project_jobs, projects
from ....services.dataset import scan as datasets
from ....paths import REPO_ROOT
from .... import preprocess as preprocess_svc
from ....services import (
    downloader,
    model_downloader,
    preprocess_manifest,
    uploads as uploads_svc,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# /api/projects/{pid}/download + /api/projects/{pid}/files + /api/jobs/*  (PP2)
# ---------------------------------------------------------------------------


@router.post("/api/projects/{pid}/download/estimate")
def estimate_download(pid: int, body: EstimateRequest) -> dict[str, Any]:
    """先调 booru 的 count API 估算命中数，再让用户决定 count。

    返回 -1 表示未知（API 不支持精确计数）；前端按「下载全部」处理。
    """
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise HTTPException(400, f"不支持的 api_source: {body.api_source}")
    if not body.tag.strip():
        raise HTTPException(400, "tag 不能为空")
    if not secrets.has_credentials_for(body.api_source):
        raise HTTPException(
            400,
            f"未配置 {body.api_source} 凭据，请先到「设置」页填写",
        )
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
    sec = secrets.load()
    if body.api_source == "danbooru":
        opts = downloader.DownloadOptions(
            tag=body.tag.strip(),
            count=1,
            api_source="danbooru",
            username=sec.danbooru.username,
            api_key=sec.danbooru.api_key,
            exclude_tags=list(sec.download.exclude_tags),
        )
    else:
        opts = downloader.DownloadOptions(
            tag=body.tag.strip(),
            count=1,
            api_source="gelbooru",
            user_id=sec.gelbooru.user_id,
            api_key=sec.gelbooru.api_key,
            exclude_tags=list(sec.download.exclude_tags),
        )
    count = downloader.estimate(opts)
    return {
        "tag": body.tag.strip(),
        "api_source": body.api_source,
        "exclude_tags": list(sec.download.exclude_tags),
        "effective_query": opts.effective_tag_query(),
        "count": count,
    }


@router.post("/api/projects/{pid}/download")
def start_download(pid: int, body: DownloadRequest) -> dict[str, Any]:
    if not body.tag.strip():
        raise HTTPException(400, "tag 不能为空")
    if body.count < 1:
        raise HTTPException(400, "count 必须 >= 1")
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise HTTPException(400, f"不支持的 api_source: {body.api_source}")
    if not secrets.has_credentials_for(body.api_source):
        raise HTTPException(
            400,
            f"未配置 {body.api_source} 凭据，请先到「设置」页填写",
        )

    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            kind="download",
            params={
                "tag": body.tag.strip(),
                "count": body.count,
                "api_source": body.api_source,
            },
        )
    _publish_job_state(job)
    return job


def _apply_project_upload_result(pid: int, result: uploads_svc.UploadResult) -> dict[str, Any]:
    # ADR-0007 PR-5: project 无 stage 字段；upload 完成由前端实时扫 download/ 派生数字
    return result.as_dict()


@router.post("/api/projects/{pid}/upload")
async def upload_local_files(
    pid: int, files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """本地上传：单图（jpg/png）或 zip 包（自动解压）→ project 的 download/。

    与 booru 下载共用同一份「全量备份」目录；上传不走 job 系统，端点同步处理
    并返回 added / skipped 列表。任一文件成功即把项目 stage 推到 downloading。
    """
    if not files:
        raise HTTPException(400, "没有上传文件")
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"

    # 全量读入内存交给 service 解析；FastAPI 的 UploadFile 内部本就是 SpooledTemporaryFile，
    # 大文件会落临时盘，所以这里 read() 不会立即吃光内存。
    pairs: list[tuple[str, io.BytesIO]] = []
    for f in files:
        data = await f.read()
        pairs.append((f.filename or "", io.BytesIO(data)))
    result = uploads_svc.accept_many(pairs, pdir)
    return _apply_project_upload_result(pid, result)


@router.post("/api/projects/{pid}/upload-from-path")
def upload_local_file_from_path(pid: int, body: UploadFromPathBody) -> dict[str, Any]:
    """从 server 可见路径导入单图或 zip → project 的 download/。"""
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    src = Path(body.path)
    if not src.is_absolute():
        src = (REPO_ROOT / src).resolve()
    else:
        src = src.resolve()
    if not src.exists():
        raise HTTPException(404, f"文件不存在: {body.path}")
    if not src.is_file():
        raise HTTPException(400, "请选择文件")
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    with src.open("rb") as fh:
        result = uploads_svc.accept_many([(src.name, fh)], pdir)
    return _apply_project_upload_result(pid, result)


@router.get("/api/projects/{pid}/download/status")
def download_status(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        job = project_jobs.latest_for(conn, project_id=pid, kind="download")
    if not job:
        return {"job": None, "log_tail": ""}
    log_path = Path(job.get("log_path") or "")
    tail = ""
    if log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(text.splitlines()[-50:])
        except Exception:
            tail = ""
    return {"job": job, "log_tail": tail}


# ---------------------------------------------------------------------------
# /api/projects/{pid}/preprocess/* — 预处理阶段（下载与筛选之间）
# 第一阶段只做放大（spandrel + 4x-AnimeSharp）；裁剪 / 涂抹后续 PR。
# ---------------------------------------------------------------------------


@router.post("/api/projects/{pid}/preprocess/start")
def start_preprocess(pid: int, body: PreprocessStartRequest) -> dict[str, Any]:
    """开始预处理 job（当前只放大）。

    mode='all' 增量跳过已处理；'all_force' 全部重跑；'selected' 处理 names。
    返回新建的 job 行。
    """
    if body.mode not in ("all", "selected", "all_force"):
        raise HTTPException(400, f"未知 mode: {body.mode}")
    if body.tile_size <= 0:
        raise HTTPException(400, "tile_size 必须 > 0")
    if body.device not in ("auto", "cuda", "cpu"):
        raise HTTPException(400, f"未知 device: {body.device}")
    # 边界：合理面积区间 256² ~ 4096²（再大就该自己写脚本了），None 表示关闭智能模式
    if body.target_area is not None and (
        body.target_area < 256 * 256 or body.target_area > 4096 * 4096
    ):
        raise HTTPException(400, f"target_area 超出范围: {body.target_area}")

    # 模型权重必须先下载（避免 worker 启起来才报错）。
    # body.model 可以是预设 label 或 custom filename（带扩展名）；
    # upscaler_target 内部做穿越保护 + 扩展名白名单。
    try:
        target = model_downloader.upscaler_target(body.model)
    except ValueError as exc:
        raise HTTPException(400, f"未知放大器: {body.model}") from exc
    if not target.exists():
        raise HTTPException(
            409,
            f"放大器权重未下载: {body.model}（请先到「设置 → 预处理」下载）",
        )

    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        try:
            job = preprocess_svc.start_job(
                conn,
                project_id=pid,
                mode=body.mode,
                names=body.names,
                model=body.model,
                tile_size=body.tile_size,
                tile_pad=body.tile_pad,
                device=body.device,
                target_area=body.target_area,
            )
        except preprocess_svc.PreprocessError as exc:
            raise HTTPException(400, str(exc)) from exc
    _publish_job_state(job)
    return job


@router.get("/api/projects/{pid}/preprocess/status")
def preprocess_status(pid: int) -> dict[str, Any]:
    """返回最新 preprocess job + 日志尾 + 概要统计。"""
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise HTTPException(404, f"项目不存在: id={pid}")
        job = project_jobs.latest_for(
            conn, project_id=pid, kind=preprocess_svc.PREPROCESS_KIND
        )
    log_tail = ""
    if job:
        log_path = Path(job.get("log_path") or "")
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                log_tail = "\n".join(text.splitlines()[-50:])
            except Exception:
                log_tail = ""
    return {
        "job": job,
        "log_tail": log_tail,
        "summary": preprocess_svc.summary(p),
    }


@router.get("/api/projects/{pid}/preprocess/files")
def list_preprocess_files(pid: int) -> dict[str, Any]:
    """返回 preprocess/ 已处理产物 + download/ 里还没处理的源。"""
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    return {
        "processed": preprocess_svc.list_processed(p),
        "pending": preprocess_svc.list_pending(p),
        "summary": preprocess_svc.summary(p),
    }


@router.get("/api/projects/{pid}/preprocess/duplicates/removed")
def list_duplicate_removed(pid: int) -> dict[str, Any]:
    """总览页「已删除」tab：列出被去重审核标记的 manifest entries。

    返回 `{images: [{name, source, w, h, mtime, size}, ...]}`。物理图仍在
    `download/{source}`，缩略图按 download bucket + source 取。恢复走
    `POST /api/projects/{pid}/preprocess/files/restore`（restore() 对
    duplicate_removed entry 也 work：删 entry，没 PNG 时静默跳过）。
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    return {"images": preprocess_svc.list_duplicate_removed_workspace(p)}


@router.get("/api/projects/{pid}/preprocess/crop/workspace")
def list_crop_workspace(pid: int) -> dict[str, Any]:
    """裁剪页工作集：返回所有可裁剪的图 + 像素尺寸。

    包含两类：
    - preprocess/ 里已处理的图（origin 指 download/ 原图）
    - download/ 里未处理的图（裁剪页把"未放大"图当 1× pass-through）

    返回 `{images: [{name, source, w, h, mtime, size, processed}, ...]}`。
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    return {"images": preprocess_svc.list_crop_workspace(p)}


@router.post("/api/projects/{pid}/preprocess/crop")
def start_preprocess_crop(
    pid: int, body: PreprocessCropRequest,
) -> dict[str, Any]:
    """开始裁剪 job。

    `crops`: `{源文件名: [{x,y,w,h,label?}], ...}`，每条 rect 归一化 [0..1]。
    源文件名为 preprocess/ 下当前文件名（worker 兜底 download/）。

    返回新建的 job 行。worker 切 PNG + 更新 manifest（多裁剪走 fan-out 命名
    `{stem}_c{n}.png` 并删原 `{stem}.png`）。详见 docs/design/preprocess-crop-design.md。
    """
    if not body.crops:
        raise HTTPException(400, "crops 不能为空")
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"项目不存在: id={pid}")
        # Pydantic 模型转成 dict 喂业务层（业务层会再做一次校验 + clamp）
        crops_payload: dict[str, list[dict[str, Any]]] = {
            name: [r.model_dump() for r in rects]
            for name, rects in body.crops.items()
        }
        try:
            job = preprocess_svc.start_crop_job(
                conn, project_id=pid, crops=crops_payload,
            )
        except preprocess_svc.PreprocessError as exc:
            raise HTTPException(400, str(exc)) from exc
    _publish_job_state(job)
    return job


@router.post("/api/projects/{pid}/preprocess/files/reset")
def reset_preprocess_files(pid: int) -> dict[str, Any]:
    """整项目预处理状态归零：删 manifest 所有 entry + 删 preprocess/ 所有 PNG。

    工具栏「总览」tab 的「撤销全部」走这个；下游 resolver 回看 download/ 原图。
    `preprocess_manifest.clear_all` 已存在；这里只是 HTTP 入口 + 项目存在校验。
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    pdir = projects.project_dir(p["id"], p["slug"])
    preprocess_manifest.clear_all(pdir)
    _publish_project_state(p)
    return {"ok": True}


@router.post("/api/projects/{pid}/preprocess/files/restore")
def restore_preprocess_files(
    pid: int, body: PreprocessRestoreRequest,
) -> dict[str, Any]:
    """还原指定产物：删 manifest entry + 删 preprocess/{name} PNG。

    还原后图回到「未处理」（隐式 original）状态。下游 resolver 重新指向
    download/{原名}。见 ADR 0004。
    """
    if not body.names:
        return {"restored": [], "missing": []}
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    try:
        res = preprocess_svc.restore_products(p, body.names)
    except preprocess_svc.PreprocessError as exc:
        raise HTTPException(400, str(exc)) from exc
    if res["restored"]:
        _publish_project_state(p)
    return res


@router.get("/api/projects/{pid}/preprocess/thumb")
def preprocess_thumb(
    pid: int, name: str = "", size: int = 256,
) -> FileResponse:
    """[Deprecated] preprocess/ 目录的缩略图。

    ADR 0004 之后 `/api/projects/{pid}/thumb?bucket=download&name=<original>`
    自带 manifest resolve，前端走那个就够；此端点保留只为兼容旧 URL（仍按
    传入的 preprocess/{name} 直读，不绕 manifest）。
    """
    if "/" in name or "\\" in name or ".." in name or not name:
        raise HTTPException(400, "invalid name")
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"项目不存在: id={pid}")
    _, pre = preprocess_svc.project_paths(p)
    f = pre / name
    if not f.exists() or f.suffix.lower() not in datasets.IMAGE_EXTS:
        logger.info("preprocess thumb 404: pid=%s name=%s -> %s", pid, name, f)
        raise HTTPException(404)
    return _thumb_response(f, size)
