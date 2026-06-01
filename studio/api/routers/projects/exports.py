"""Train / Bundle 导出 + 跨项目导入（PR-6.5 commit 2 从 server.py 抽出）。

6 routes：
    GET  /api/projects/{pid}/versions/{vid}/train.zip           train/ + manifest.json 临时打包
    GET  /api/projects/{pid}/versions/{vid}/bundle.zip          按选项临时打包 bundle（schema v2）
    POST /api/projects/{pid}/versions/{vid}/export-bundle       打包到 data_exports/
    POST /api/projects/import-bundle                            从 PathPicker 路径 / data_exports 导入
    POST /api/projects/import-bundle/upload                     上传 zip 导入
    POST /api/projects/import-train                             上传训练集 zip → 新建 project + v1

train.zip / bundle.zip / export-bundle 用 ZIP_STORED（PNG/jpg 已压缩再压浪费 CPU），
打包完成 / 失败 publish version_train_zip_ready / _failed + version_bundle_zip_ready /
_failed —— 前端 <a> 直链触发下载 + SSE 清 app-side "打包中..." 状态。
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ...errors import _data_export_path, _export_result, _unique_data_export_path
from ...schemas.exports import BundleImportBody, BundleOptionsBody
from ._shared import (
    _project_payload,
    _publish_project_state,
    _publish_version_state,
)
from .... import db
from ....services.projects import projects, versions
from ....infrastructure.event_bus import bus
from ....paths import DATA_EXPORTS, REPO_ROOT, USER_PRESETS_DIR
from ....services.data_io import train_io

router = APIRouter()


@router.get("/api/projects/{pid}/versions/{vid}/train.zip")
def export_version_train_zip(
    pid: int, vid: int, background: BackgroundTasks,
) -> FileResponse:
    """打包 version 的 train/ + manifest.json 为 zip 一次性下载。

    实现：写到临时文件再 FileResponse；响应发完后 BackgroundTasks 清理。
    与 outputs.zip 一致用 ZIP_STORED（PNG/jpg 已压缩，再压只是浪费 CPU）。

    打包完成 / 失败 publish version_train_zip_ready / _failed —— 前端用 <a>
    直链触发下载（浏览器原生进度条），SSE 事件用于清 app-side "打包中..." 状态
    + 失败时弹 toast。和 outputs.zip 一套范式。
    """
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        p = projects.get_project(conn, pid)
        assert p is not None

        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            train_io.export_train(conn, vid, tmp_path)
        except train_io.TrainIOError as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({
                "type": "version_train_zip_failed",
                "project_id": pid,
                "version_id": vid,
                "error": str(exc),
            })
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({
                "type": "version_train_zip_failed",
                "project_id": pid,
                "version_id": vid,
                "error": str(exc),
            })
            raise

    bus.publish({
        "type": "version_train_zip_ready",
        "project_id": pid,
        "version_id": vid,
    })
    background.add_task(lambda: tmp_path.unlink(missing_ok=True))
    archive_name = f"{p['slug']}-{v['label']}.train.zip"
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=archive_name,
        background=background,
    )


@router.get("/api/projects/{pid}/versions/{vid}/bundle.zip")
def export_version_bundle(
    pid: int,
    vid: int,
    background: BackgroundTasks,
    train: bool = True,
    train_captions: bool = True,
    reg: bool = False,
    reg_captions: bool = False,
    include_config: bool = False,
) -> FileResponse:
    """按选项临时打包 bundle.zip（schema_version 2）并交给浏览器下载。"""
    opts = train_io.BundleOptions(
        train=train,
        train_captions=train_captions,
        reg=reg,
        reg_captions=reg_captions,
        include_config=include_config,
    )

    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        p = projects.get_project(conn, pid)
        assert p is not None

        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            train_io.export_bundle(conn, vid, tmp_path, opts)
        except train_io.TrainIOError as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise

    bus.publish({"type": "version_bundle_zip_ready", "project_id": pid, "version_id": vid})
    background.add_task(lambda: tmp_path.unlink(missing_ok=True))
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=f"{p['slug']}-{v['label']}.bundle.zip",
        background=background,
    )


@router.post("/api/projects/{pid}/versions/{vid}/export-bundle")
def export_version_bundle_to_data_exports(
    pid: int,
    vid: int,
    body: BundleOptionsBody,
) -> dict[str, Any]:
    """按选项打包 bundle.zip 并保存到 data_exports/。"""
    opts = body.to_options()
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise HTTPException(404, f"版本不存在: id={vid}")
        p = projects.get_project(conn, pid)
        assert p is not None
        DATA_EXPORTS.mkdir(parents=True, exist_ok=True)
        dest = _unique_data_export_path(f"{p['slug']}-{v['label']}.bundle.zip")
        try:
            train_io.export_bundle(conn, vid, dest, opts)
        except train_io.TrainIOError as exc:
            dest.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            dest.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise

    bus.publish({"type": "version_bundle_zip_ready", "project_id": pid, "version_id": vid})
    return _export_result(dest)


def _bundle_import_payload(result: dict[str, Any]) -> dict[str, Any]:
    p = result["project"]
    _publish_project_state(p)
    _publish_version_state(result["version"])
    return {
        "project": _project_payload(p),
        "version": result["version"],
        "stats": result["stats"],
    }


def _import_bundle_from_path(dest: Path, original: str) -> dict[str, Any]:
    if not dest.exists():
        raise HTTPException(404, f"文件不存在: {original}")
    if not dest.is_file():
        raise HTTPException(400, "请选择 zip 文件")
    if dest.suffix.lower() != ".zip":
        raise HTTPException(400, "请选择 .zip 文件")
    with db.connection_for() as conn:
        try:
            result = train_io.import_bundle(conn, dest, USER_PRESETS_DIR)
        except train_io.TrainIOError as exc:
            raise HTTPException(400, str(exc)) from exc
    return _bundle_import_payload(result)


@router.post("/api/projects/import-bundle")
def import_bundle_zip(body: BundleImportBody) -> dict[str, Any]:
    """从 PathPicker 路径或 data_exports 文件名导入 bundle（v1/v2 均支持）。"""
    if body.filename:
        return _import_bundle_from_path(_data_export_path(body.filename), body.filename)
    assert body.path is not None
    dest = Path(body.path)
    if not dest.is_absolute():
        dest = (REPO_ROOT / dest).resolve()
    else:
        dest = dest.resolve()
    return _import_bundle_from_path(dest, body.path)


@router.post("/api/projects/import-bundle/upload")
async def import_bundle_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """上传 bundle zip → 新建 project + version。"""
    if not file.filename:
        raise HTTPException(400, "缺少上传文件")
    if Path(file.filename).suffix.lower() != ".zip":
        raise HTTPException(400, "请选择 .zip 文件")
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.close()
        return _import_bundle_from_path(Path(tmp.name), file.filename)
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/api/projects/import-train")
async def import_train_zip(file: UploadFile = File(...)) -> dict[str, Any]:
    """上传训练集 zip → 新建 project + v1（stage=tagging），返回新项目。"""
    if not file.filename:
        raise HTTPException(400, "缺少上传文件")
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        # UploadFile 内部本就是 SpooledTemporaryFile，大文件会落临时盘
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.close()
        tmp_path = Path(tmp.name)
        with db.connection_for() as conn:
            try:
                result = train_io.import_train(conn, tmp_path)
            except train_io.TrainIOError as exc:
                raise HTTPException(400, str(exc)) from exc
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass

    p = result["project"]
    _publish_project_state(p)
    _publish_version_state(result["version"])
    return {
        "project": _project_payload(p),
        "version": result["version"],
        "stats": result["stats"],
    }
