"""Queue 任务 output 文件 列表 / 下载 / 打包 / 系统级 open-folder（PR-6 commit 6 从 server.py 抽出）。

5 routes：
    GET  /api/queue/{task_id}/outputs                 列 output 目录所有文件（含 state/）
    GET  /api/queue/{task_id}/outputs.zip             打包 zip 一次性下载
    POST /api/queue/{task_id}/export-outputs          打包到 data_exports/
    GET  /api/queue/{task_id}/output/{filename:path}  下载指定文件
    POST /api/queue/{task_id}/open-folder             OS 文件管理器打开（仅 loopback）

关联 helpers 全留在本文件（只这 5 route 用）：_task_output_dir / _LOCALHOST_HOSTS /
_LORA_EXTS / _task_output_kind / _task_output_relpath / _iter_task_output_files /
_safe_output_relpath_or_400 / _select_task_output_files / _write_outputs_zip /
_task_archive_basename / _is_loopback。
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse

from ...errors import _export_result, _safe_join_or_400, _unique_data_export_path
from ...schemas.queue import DeleteOutputsBody, ExportOutputsBody
from .... import db
from ....services.projects import projects, versions
from ....infrastructure.event_bus import bus
from ....paths import DATA_EXPORTS

router = APIRouter()

_LOCALHOST_HOSTS = {"127.0.0.1", "::1", "localhost"}
_LORA_EXTS = {".safetensors", ".ckpt", ".pt", ".bin"}


def _task_output_dir(task: dict[str, Any]) -> Optional[Path]:
    """根据 task 推断 output 目录：versions/{label}/output。

    没 project_id / version_id 的老任务（PP1 之前）→ 返回 None；调用方应该
    处理为「无 output 目录」。
    """
    pid = task.get("project_id")
    vid = task.get("version_id")
    if not (pid and vid):
        return None
    with db.connection_for() as conn:
        v = versions.get_version(conn, int(vid))
        p = projects.get_project(conn, int(pid))
    if not v or not p or v["project_id"] != pid:
        return None
    return versions.version_dir(int(pid), p["slug"], v["label"]) / "output"


def _select_task_output_files(task_id: int, files: Optional[list[str]] = None) -> tuple[dict[str, Any], list[Path], bool]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    out_dir = _task_output_dir(task)
    if not out_dir or not out_dir.exists():
        raise HTTPException(404, "no output dir")
    all_files = _iter_task_output_files(out_dir, task_id)
    if not all_files:
        raise HTTPException(404, "empty output dir")
    if not files:
        return task, all_files, False
    by_path = {_task_output_relpath(out_dir, f): f for f in all_files}
    selected: list[Path] = []
    missing: list[str] = []
    for name in files:
        _safe_output_relpath_or_400(out_dir, name)
        f = by_path.get(name)
        if f:
            selected.append(f)
        else:
            missing.append(name)
    if missing:
        raise HTTPException(404, f"file(s) not found: {', '.join(missing)}")
    if not selected:
        raise HTTPException(400, "empty files list")
    return task, selected, True


def _write_outputs_zip(dest: Path, out_dir: Path, selected: list[Path]) -> None:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for f in selected:
            zf.write(f, arcname=_task_output_relpath(out_dir, f))


def _task_archive_basename(task: dict[str, Any]) -> Optional[str]:
    """task 关联 project / version → "{slug}-{label}"，用作 outputs zip 文件名
    前缀。和 train.zip 命名风格一致（PP7：{slug}-{label}.train.zip）。

    没 project / version → None，调用方 fallback 到 task_{id}。
    """
    pid = task.get("project_id")
    vid = task.get("version_id")
    if not (pid and vid):
        return None
    with db.connection_for() as conn:
        v = versions.get_version(conn, int(vid))
        p = projects.get_project(conn, int(pid))
    if not v or not p or v["project_id"] != pid:
        return None
    return f"{p['slug']}-{v['label']}"


def _is_loopback(request: Request) -> bool:
    client = request.client
    return bool(client and client.host in _LOCALHOST_HOSTS)


def _task_output_kind(path: Path) -> str:
    name = path.name
    suffix = path.suffix.lower()
    if name.startswith("training_state_") and suffix == ".pt":
        return "training_state"
    if name.startswith("pause_step_") and suffix == ".pt":
        return "pause_state"
    if name == "auto_epoch_state.pt":
        return "auto_epoch_state"
    if suffix in _LORA_EXTS and not name.startswith("training_state_"):
        return "lora"
    return "other"


def _task_output_relpath(out_dir: Path, path: Path) -> str:
    return path.relative_to(out_dir).as_posix()


def _iter_task_output_files(out_dir: Path, task_id: int) -> list[Path]:
    files = [p for p in out_dir.iterdir() if p.is_file()]
    state_dir = out_dir / "state" / f"task_{task_id}"
    if state_dir.exists():
        files.extend(
            p for p in state_dir.rglob("*")
            if p.is_file() and _task_output_kind(p) in {"training_state", "pause_state", "auto_epoch_state"}
        )
    return sorted(files, key=lambda p: _task_output_relpath(out_dir, p))


def _safe_output_relpath_or_400(base: Path, relpath: str) -> Path:
    if not relpath or relpath.startswith(("/", "\\")):
        raise HTTPException(400, "invalid path")
    parts = Path(relpath.replace("\\", "/")).parts
    if any(part in ("", ".", "..") for part in parts):
        raise HTTPException(400, "invalid path")
    return _safe_join_or_400(base, *parts)


@router.get("/api/queue/{task_id}/outputs")
def list_task_outputs(task_id: int, request: Request) -> dict[str, Any]:
    """列出 task 关联 version 的 output 目录里所有文件。"""
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    out_dir = _task_output_dir(task)
    files: list[dict[str, Any]] = []
    if out_dir and out_dir.exists():
        for f in _iter_task_output_files(out_dir, task_id):
            relpath = _task_output_relpath(out_dir, f)
            try:
                st = f.stat()
            except OSError:
                continue
            kind = _task_output_kind(f)
            files.append({
                "name": f.name,
                "path": relpath,
                "size": st.st_size,
                "mtime": st.st_mtime,
                "kind": kind,
                "is_lora": kind == "lora",
            })
    return {
        "task_id": task_id,
        "output_dir": str(out_dir) if out_dir else None,
        "exists": bool(out_dir and out_dir.exists()),
        "supports_open_folder": _is_loopback(request),
        "files": files,
        "archive_basename": _task_archive_basename(task),
    }


@router.get("/api/queue/{task_id}/outputs.zip")
def download_task_outputs_zip(
    task_id: int,
    background: BackgroundTasks,
    files: Optional[str] = None,
) -> FileResponse:
    """把 output 目录里的文件打包成 zip 一次性下载。"""
    import tempfile
    wanted = [n for n in files.split(",") if n] if files else None
    if files is not None and not wanted:
        raise HTTPException(400, "empty files list")
    task, selected, partial = _select_task_output_files(task_id, wanted)
    out_dir = _task_output_dir(task)
    assert out_dir is not None  # _select_task_output_files 已经校验

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        _write_outputs_zip(tmp_path, out_dir, selected)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        bus.publish({
            "type": "task_outputs_zip_failed",
            "task_id": task_id,
            "error": str(e),
        })
        raise

    bus.publish({"type": "task_outputs_zip_ready", "task_id": task_id})
    background.add_task(lambda: tmp_path.unlink(missing_ok=True))

    basename = _task_archive_basename(task) or f"task_{task_id}"
    archive_name = (
        f"{basename}_outputs_selected.zip" if partial
        else f"{basename}_outputs.zip"
    )
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=archive_name,
        background=background,
    )


@router.post("/api/queue/{task_id}/export-outputs")
def export_task_outputs_to_data_exports(
    task_id: int,
    body: ExportOutputsBody,
) -> dict[str, Any]:
    """把 output 文件打包保存到 data_exports/。"""
    task, selected, partial = _select_task_output_files(task_id, body.files)
    DATA_EXPORTS.mkdir(parents=True, exist_ok=True)
    basename = _task_archive_basename(task) or f"task_{task_id}"
    archive_name = f"{basename}_outputs_selected.zip" if partial else f"{basename}_outputs.zip"
    dest = _unique_data_export_path(archive_name)
    out_dir = _task_output_dir(task)
    assert out_dir is not None
    try:
        _write_outputs_zip(dest, out_dir, selected)
    except Exception as e:
        dest.unlink(missing_ok=True)
        bus.publish({"type": "task_outputs_zip_failed", "task_id": task_id, "error": str(e)})
        raise
    bus.publish({"type": "task_outputs_zip_ready", "task_id": task_id})
    return _export_result(dest)


@router.get("/api/queue/{task_id}/output/{filename:path}")
def download_task_output(task_id: int, filename: str) -> FileResponse:
    """下载 output 目录下的指定文件。"""
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    out_dir = _task_output_dir(task)
    if not out_dir or not out_dir.exists():
        raise HTTPException(404, "no output dir")
    f = _safe_output_relpath_or_400(out_dir, filename)
    if not f.exists() or not f.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(
        f,
        media_type="application/octet-stream",
        filename=f.name,
    )


@router.delete("/api/queue/{task_id}/outputs")
def delete_task_output_files(
    task_id: int,
    body: DeleteOutputsBody,
) -> dict[str, Any]:
    """删除 output 目录下的指定文件（批量）。

    body.files 是相对 output/ 的路径列表，禁绝对路径 / path traversal；
    任何一个不存在 → 404 拒绝整批，避免半删状态。
    """
    if not body.files:
        raise HTTPException(400, "empty files list")
    task, selected, _ = _select_task_output_files(task_id, body.files)
    deleted: list[str] = []
    out_dir = _task_output_dir(task)
    assert out_dir is not None
    for f in selected:
        try:
            f.unlink()
            deleted.append(_task_output_relpath(out_dir, f))
        except OSError as exc:
            raise HTTPException(500, f"failed to delete {f.name}: {exc}") from exc
    return {"deleted": deleted}


@router.post("/api/queue/{task_id}/open-folder")
def open_task_folder(task_id: int, request: Request) -> dict[str, Any]:
    """在 server 主机上用 OS 文件管理器打开 output 目录。

    **仅 loopback 请求允许**：云端部署时浏览器不在 server 那台机，开了用户也
    看不到，反而是远程命令执行入口；这里直接 403 拒绝。
    """
    if not _is_loopback(request):
        raise HTTPException(
            403, "open-folder is only available for local (loopback) requests"
        )
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    out_dir = _task_output_dir(task)
    if not out_dir or not out_dir.exists():
        raise HTTPException(404, "no output dir")
    try:
        import platform
        import subprocess as _sub
        sysname = platform.system()
        if sysname == "Windows":
            os.startfile(str(out_dir))  # type: ignore[attr-defined]
        elif sysname == "Darwin":
            _sub.Popen(["open", str(out_dir)])
        else:
            _sub.Popen(["xdg-open", str(out_dir)])
    except Exception as exc:
        raise HTTPException(500, f"failed to open folder: {exc}") from exc
    return {"opened": str(out_dir)}
