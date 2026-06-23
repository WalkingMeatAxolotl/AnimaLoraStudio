"""模型根目录迁移 —— 扫描体积 + 后台复制到自定义位置（镜像 studio_data 迁移）。

流程（前端 Settings → 系统 → 存储位置 → 模型根目录）：
1. GET  /api/models-root/info          —— 当前/默认位置 + 全量扫描（文件数/字节数/顶层明细）
2. POST /api/models-root/migrate       —— 校验后起后台线程复制；进度走 SSE
3. 复制完成 → 更新 `secrets.models.root` → **立即生效**（`models_root()` 每次现读
   secret，无需重启；区别于 studio_data 的指针文件 + 重启）

设计要点（与 `studio_data.py` 一致）：
- **目标是父目录**：用户选任意目录，数据落 `目标/models/`；目标本身不要求为空，只
  要求落地子目录不存在或为空（不 merge 进已有数据）。
- **只复制不删除**：旧目录原样保留（用户决策；已有 version 的 yaml 里烤死的旧绝对
  路径仍可解析）。失败时清掉复制了一半的落地目录，secret 不动，等于什么都没发生。
- **单飞**：模块级 lock + 状态单例。
- 模型权重无 sqlite/wal，直接 `shutil.copy2`，无 `.db` backup 分支。
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .. import secrets
from ..infrastructure.event_bus import bus
from ..infrastructure.paths import REPO_ROOT
from .models import models_root

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL_SECONDS = 0.2

# 落地子目录名固定 —— 不跟随当前位置的目录名（自定义位置可能叫别的）。
DATA_DIR_NAME = "models"

Publish = Callable[[dict[str, Any]], None]


def default_models_root() -> Path:
    """`secrets.models.root` 未设时的默认落点（同 `models_root()` 的回退）。"""
    return REPO_ROOT / "models"


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------

def scan_models_root(root: Path | None = None) -> dict[str, Any]:
    """全量扫描模型根目录：总文件数 / 总字节数 + 顶层条目明细（确认 modal 显示用）。

    目录不存在时返回全 0（还没下载过任何模型）。
    """
    base = root if root is not None else models_root()
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
                if not f.is_file():
                    continue
                files += 1
                try:
                    size += f.stat().st_size
                except OSError:
                    pass
        elif child.is_file():
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

    target 是用户选的任意目录，数据落 `target/models/`，所以 target 本身不要求
    为空。规则：绝对路径；target 已存在时必须是目录；落地目录不等于当前位置；
    落地目录与当前位置互不嵌套；落地目录不存在或为空。
    """
    src = (source if source is not None else models_root()).resolve()
    if not target.is_absolute():
        raise ValueError("目标必须是绝对路径")
    if target.exists() and not target.is_dir():
        raise ValueError("目标已存在且不是目录")
    dst = target.resolve() / DATA_DIR_NAME
    if dst == src:
        raise ValueError("目标与当前模型根目录位置相同")
    for a, b in ((dst, src), (src, dst)):
        try:
            a.relative_to(b)
        except ValueError:
            continue
        raise ValueError("目标目录与当前模型根目录互相嵌套")
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
) -> None:
    """校验 + 起后台复制线程。已有迁移在跑时抛 RuntimeError（caller 转 409）。

    source 参数仅测试注入用；生产走默认（当前 `models_root()`）。
    """
    src = (source if source is not None else models_root()).resolve()
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
        args=(src, dst, publish),
        name="models-root-migration",
        daemon=True,
    )
    t.start()


def _update_models_root_secret(new_root: Path) -> None:
    """复制成功后把 `secrets.models.root` 指到新落地目录（立即生效，无需重启）。"""
    cur = secrets.load()
    new_models = cur.models.model_copy(update={"root": str(new_root)})
    secrets.save(cur.model_copy(update={"models": new_models}))


# ---------------------------------------------------------------------------
# 复制线程
# ---------------------------------------------------------------------------

def _run_migration(src: Path, dst: Path, publish: Publish) -> None:
    try:
        files = [f for f in sorted(src.rglob("*")) if f.is_file()]
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
                    "type": "models_root_migrate_progress",
                    "done_files": done_files,
                    "total_files": len(files),
                    "done_bytes": done_bytes,
                    "total_bytes": total_bytes,
                    "current_file": str(rel),
                })

        # 复制完整 → 切 secret（立即生效）
        _update_models_root_secret(dst)
        _set_status(state="done", done_files=done_files, done_bytes=done_bytes, current_file="")
        publish({
            "type": "models_root_migrate_done",
            "ok": True,
            "target": str(dst),
            "done_files": done_files,
            "done_bytes": done_bytes,
        })
        logger.info(
            "模型根目录迁移完成: %s → %s（%d 文件），已更新 secrets.models.root（立即生效）",
            src, dst, done_files,
        )
    except Exception as exc:
        logger.exception("模型根目录迁移失败: %s → %s", src, dst)
        # dst 开始前为空 / 不存在（validate_target 保证），整树清掉等于回到迁移前；
        # secret 未动；用户的 target 父目录不受影响。
        shutil.rmtree(dst, ignore_errors=True)
        _set_status(state="error", error=str(exc))
        publish({"type": "models_root_migrate_done", "ok": False, "error": str(exc)})
