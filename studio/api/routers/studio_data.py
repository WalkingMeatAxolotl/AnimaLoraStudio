"""studio_data 存储位置 —— 查询 / 迁移到自定义目录。

3 routes：
    GET  /api/studio-data/info            当前/默认位置 + 全量扫描（文件数/字节）
    POST /api/studio-data/migrate         校验 + 起后台复制线程（进度走 SSE）
    GET  /api/studio-data/migrate_status  迁移状态快照（modal 重开 / SSE 漏事件兜底）

迁移协议：复制完成后写仓库根 `studio_data_location.json` 指针，**重启 server
生效**（paths.STUDIO_DATA 是 import 时求值；cli.py 重启循环拉新进程重新解析）。
旧位置数据保留不删。进度事件：`studio_data_migrate_progress` / `_done`。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..schemas.studio_data import StudioDataMigrateRequest
from ...infrastructure.paths import DEFAULT_STUDIO_DATA, STUDIO_DATA
from ...services import studio_data as svc
from .system import _check_no_running_tasks

router = APIRouter()


@router.get("/api/studio-data/info")
def studio_data_info(scan: bool = True) -> dict[str, Any]:
    """当前 / 默认位置；scan=true 时附全量扫描（大目录可能要数秒，前端确认
    modal 加载态等它；Settings 页仅显示路径用 scan=false 免扫盘）。"""
    return {
        "current": str(STUDIO_DATA),
        "default": str(DEFAULT_STUDIO_DATA),
        "is_custom": STUDIO_DATA.resolve() != DEFAULT_STUDIO_DATA.resolve(),
        "scan": svc.scan_studio_data() if scan else None,
    }


@router.post("/api/studio-data/migrate")
def studio_data_migrate(body: StudioDataMigrateRequest) -> dict[str, Any]:
    """起迁移。约束：无 running task（复制期间训练继续写文件会拷出半截数据）。"""
    _check_no_running_tasks()
    try:
        svc.start_migration(Path(body.target))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/api/studio-data/migrate_status")
def studio_data_migrate_status() -> dict[str, Any]:
    return svc.migration_status()
