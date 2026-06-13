"""studio_data 目录迁移 —— 扫描体积 + 后台复制到自定义位置（ADR 无；小功能）。

流程（前端 Settings → 系统 → 存储位置）：
1. GET /api/studio-data/info       —— 当前/默认位置 + 全量扫描（文件数/字节数/顶层明细）
2. POST /api/studio-data/migrate   —— 校验后起后台线程复制；进度走 SSE
3. 复制完成 → 写仓库根指针文件 `studio_data_location.json` → 重启 server 生效

设计要点：
- **目标是父目录**：用户选任意目录，数据落 `目标/studio_data/`（整个
  studio_data「搬进去」），目标本身不要求为空 —— 只要求落地子目录不存在或
  为空（不 merge 进已有数据）。
- **只复制不删除**：旧数据原样保留（用户决策）；失败时清掉复制了一半的落地
  目录（开始前要求其为空 / 不存在，rmtree 安全），指针不写，等于什么都没发生。
- **sqlite 一致性**：server 进程随请求随时可能写 studio.db，直接 copy 可能
  截到写一半的页。`.db` 文件走 sqlite3 backup API（在线备份，拿到一致快照）；
  对应的 `-wal` / `-shm` 跳过（backup 产物自含）。
- **单飞**：同时只允许一个迁移（模块级 lock + 状态单例）。
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ..infrastructure.event_bus import bus
from ..infrastructure.paths import DEFAULT_STUDIO_DATA, STUDIO_DATA, STUDIO_DATA_POINTER

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL_SECONDS = 0.2

# 落地子目录名固定 —— 不跟随当前位置的目录名（老式迁移的自定义位置可能叫别的）
DATA_DIR_NAME = "studio_data"

Publish = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------

def scan_studio_data(root: Path | None = None) -> dict[str, Any]:
    """全量扫描 studio_data：总文件数 / 总字节数 + 顶层条目明细（确认 modal 显示用）。

    `-wal` / `-shm` 不计入（迁移时跳过，见模块 docstring）。目录不存在时返回全 0。
    """
    base = root if root is not None else STUDIO_DATA
    entries: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    if not base.is_dir():
        return {"total_files": 0, "total_bytes": 0, "entries": []}
    for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        files = 0
        size = 0
        if child.is_dir():
            for f in child.rglob("*"):
                if not f.is_file() or _skip_file(f):
                    continue
                files += 1
                try:
                    size += f.stat().st_size
                except OSError:
                    pass
        elif child.is_file():
            if _skip_file(child):
                continue
            files = 1
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
        entries.append({
            "name": child.name,
            "is_dir": child.is_dir(),
            "files": files,
            "bytes": size,
        })
        total_files += files
        total_bytes += size
    return {"total_files": total_files, "total_bytes": total_bytes, "entries": entries}


def _skip_file(p: Path) -> bool:
    """sqlite 伴生文件不复制：backup API 产物已是一致单文件。"""
    return p.name.endswith(".db-wal") or p.name.endswith(".db-shm")


# ---------------------------------------------------------------------------
# 迁移状态（单例）
# ---------------------------------------------------------------------------

@dataclass
class MigrationStatus:
    state: str = "idle"          # idle / running / done / error
    target: str = ""
    total_files: int = 0
    total_bytes: int = 0
    done_files: int = 0
    done_bytes: int = 0
    current_file: str = ""       # 相对路径，进度展示用
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_status = MigrationStatus()
_status_lock = threading.Lock()


def migration_status() -> dict[str, Any]:
    with _status_lock:
        return _status.as_dict()


def _set_status(**kw: Any) -> None:
    with _status_lock:
        for k, v in kw.items():
            setattr(_status, k, v)


# ---------------------------------------------------------------------------
# 校验 + 启动
# ---------------------------------------------------------------------------

def validate_target(target: Path, *, source: Path | None = None) -> Path:
    """迁移目标校验，不合法抛 ValueError（caller 转 422）；返回实际落地目录。

    target 是用户选的任意目录，数据落 `target/studio_data/`，所以 target 本身
    不要求为空。规则：绝对路径；target 已存在时必须是目录；落地目录不等于
    当前位置；落地目录与当前位置互不嵌套（copy 进自己子树会无限递归）；
    落地目录不存在或为空（不 merge 进已有数据）。
    """
    src = (source if source is not None else STUDIO_DATA).resolve()
    if not target.is_absolute():
        raise ValueError("目标必须是绝对路径")
    if target.exists() and not target.is_dir():
        raise ValueError("目标已存在且不是目录")
    dst = target.resolve() / DATA_DIR_NAME
    if dst == src:
        raise ValueError("目标与当前 studio_data 位置相同")
    for a, b in ((dst, src), (src, dst)):
        try:
            a.relative_to(b)
        except ValueError:
            continue
        raise ValueError("目标目录与当前 studio_data 互相嵌套")
    if dst.exists():
        if not dst.is_dir():
            raise ValueError(f"目标下已存在同名文件 {DATA_DIR_NAME}")
        if any(dst.iterdir()):
            raise ValueError(f"目标下已存在非空 {DATA_DIR_NAME} 目录")
    return dst


def start_migration(
    target: Path,
    *,
    source: Path | None = None,
    publish: Publish = bus.publish,
    pointer_file: Path | None = None,
) -> None:
    """校验 + 起后台复制线程。已有迁移在跑时抛 RuntimeError（caller 转 409）。

    target 是用户选的父目录，实际复制到 `target/studio_data/`（validate_target
    返回值）。source / pointer_file 参数仅测试注入用；生产走默认（当前
    STUDIO_DATA + 仓库根指针）。
    """
    src = (source if source is not None else STUDIO_DATA).resolve()
    ptr = pointer_file if pointer_file is not None else STUDIO_DATA_POINTER
    dst = validate_target(target, source=src)
    with _status_lock:
        if _status.state == "running":
            raise RuntimeError("已有迁移正在进行")
        _status.state = "running"
        _status.target = str(dst)
        _status.total_files = 0
        _status.total_bytes = 0
        _status.done_files = 0
        _status.done_bytes = 0
        _status.current_file = ""
        _status.error = ""
    t = threading.Thread(
        target=_run_migration,
        args=(src, dst, publish, ptr),
        name="studio-data-migration",
        daemon=True,
    )
    t.start()


# ---------------------------------------------------------------------------
# 复制线程
# ---------------------------------------------------------------------------

def _run_migration(src: Path, dst: Path, publish: Publish, pointer_file: Path) -> None:
    try:
        files = [
            f for f in sorted(src.rglob("*"))
            if f.is_file() and not _skip_file(f)
        ]
        total_bytes = 0
        for f in files:
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass
        _set_status(total_files=len(files), total_bytes=total_bytes)

        dst.mkdir(parents=True, exist_ok=True)
        last_pub = 0.0
        done_files = 0
        done_bytes = 0
        for f in files:
            rel = f.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                size = f.stat().st_size
                if f.suffix == ".db":
                    _backup_sqlite(f, out)
                else:
                    shutil.copy2(f, out)
            except FileNotFoundError:
                # 扫描后被删（如临时文件）—— 跳过，进度可能停在 <100%，无碍
                logger.info("迁移期间文件消失，跳过: %s", rel)
                continue
            done_files += 1
            done_bytes += size
            now = time.monotonic()
            if now - last_pub >= PROGRESS_INTERVAL_SECONDS:
                last_pub = now
                _set_status(done_files=done_files, done_bytes=done_bytes, current_file=str(rel))
                publish({
                    "type": "studio_data_migrate_progress",
                    "done_files": done_files,
                    "total_files": len(files),
                    "done_bytes": done_bytes,
                    "total_bytes": total_bytes,
                    "current_file": str(rel),
                })

        pointer_file.write_text(
            json.dumps({"path": str(dst)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _set_status(state="done", done_files=done_files, done_bytes=done_bytes, current_file="")
        publish({
            "type": "studio_data_migrate_done",
            "ok": True,
            "target": str(dst),
            "done_files": done_files,
            "done_bytes": done_bytes,
        })
        logger.info("studio_data 迁移完成: %s → %s（%d 文件），重启后生效", src, dst, done_files)
    except Exception as exc:
        logger.exception("studio_data 迁移失败: %s → %s", src, dst)
        # dst 是 target/studio_data 落地目录，开始前为空 / 不存在
        # （validate_target 保证），整树清掉等于回到迁移前；用户的 target 父目录不动
        shutil.rmtree(dst, ignore_errors=True)
        _set_status(state="error", error=str(exc))
        publish({"type": "studio_data_migrate_done", "ok": False, "error": str(exc)})


def _backup_sqlite(src_db: Path, out: Path) -> None:
    """sqlite 在线备份拿一致快照；非 sqlite 的 .db 文件回退普通复制。"""
    try:
        with sqlite3.connect(str(src_db)) as conn, sqlite3.connect(str(out)) as dst_conn:
            conn.backup(dst_conn)
    except sqlite3.Error:
        out.unlink(missing_ok=True)
        shutil.copy2(src_db, out)
