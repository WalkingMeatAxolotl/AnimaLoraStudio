"""模型根目录存储位置 —— 查询 / 迁移到自定义目录（镜像 studio_data，无需重启）。

3 routes：
    GET  /api/models-root/info            当前/默认位置 + 全量扫描（文件数/字节）
    POST /api/models-root/migrate         校验 + 起后台复制线程（进度走 SSE）
    GET  /api/models-root/migrate_status  迁移状态快照（modal 重开 / SSE 漏事件兜底）

迁移协议：复制完成后更新 `secrets.models.root`，**立即生效**（`models_root()` 现读
secret，无指针文件、无需重启）。旧目录保留不删。进度事件：
`models_root_migrate_progress` / `_done`。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..schemas.models_storage import ModelsRootMigrateRequest
from ...domain.errors import ConflictError, ValidationError
from ...services import models_storage as svc
from ...services.models import models_root
from .system import _check_no_running_tasks

router = APIRouter()


@router.get("/api/models-root/info")
def models_root_info(scan: bool = True) -> dict[str, Any]:
    """当前 / 默认位置；scan=true 时附全量扫描（大目录可能要数秒，前端确认 modal
    加载态等它；Settings 页仅显示路径用 scan=false 免扫盘）。"""
    current = models_root()
    default = svc.default_models_root()
    return {
        "current": str(current),
        "default": str(default),
        "is_custom": current.resolve() != default.resolve(),
        "scan": svc.scan_models_root() if scan else None,
    }


@router.post("/api/models-root/migrate")
def models_root_migrate(body: ModelsRootMigrateRequest) -> dict[str, Any]:
    """起迁移。约束：无 running task（复制期间训练继续读权重 / move 语义不安全）。

    目标已有非空 models/ 且未传 on_conflict → 409 `target_conflict`（details 带
    目标现有文件统计 + 同名文件数），前端弹「跳过/覆盖/取消」后带 on_conflict 重发。
    """
    _check_no_running_tasks()
    try:
        svc.start_migration(Path(body.target), on_conflict=body.on_conflict)
    except svc.TargetConflictError as exc:
        # 注意放 except ValueError 前（它是 ValueError 子类）
        raise ConflictError(
            "Target already contains a non-empty models directory",
            code="models_root.target_conflict",
            details=exc.details,
        ) from exc
    except ValueError as exc:
        raise ValidationError(
            f"Invalid target location: {exc}",
            code="models_root.target_invalid",
            details={"reason": str(exc)}, http_status=422,
        ) from exc
    except RuntimeError as exc:
        raise ConflictError(
            "A models-root migration is already in progress",
            code="models_root.migration_busy",
        ) from exc
    return {"ok": True}


@router.get("/api/models-root/migrate_status")
def models_root_migrate_status() -> dict[str, Any]:
    return svc.migration_status()
