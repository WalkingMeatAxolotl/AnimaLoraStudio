"""共享 dependency helpers（PR-6 起从 server.py 抽出）。

跨 router 共用的 helper：拿 Supervisor 实例、版本 / 项目记录验证等。
未来会改 FastAPI `Depends(...)` 形式，本次先维持「直接调函数」风格保
零行为变更。
"""
from __future__ import annotations

from typing import Optional

from ..domain.errors import DomainError
from ..supervisor import Supervisor


def _supervisor() -> Supervisor:
    """从 app.state 取 Supervisor。lifespan startup 还没跑完时返 503。

    本 helper 内做 late import 避免 `api/app.py ↔ api/routers/* ↔ api/deps.py`
    三方循环——routers 在 app.py include 时还在初始化，此时 import api.app
    虽然拿得到 `app`（`app = FastAPI(...)` 已执行）但循环关系不健康。
    """
    from .app import app
    sup: Optional[Supervisor] = getattr(app.state, "supervisor", None)
    if sup is None:
        raise DomainError(
            "The service is still starting; try again in a moment",
            code="system.starting", http_status=503,
        )
    return sup


def _resolve_anima_model_paths() -> dict[str, str]:
    """解析 base 模型默认路径（先验生成 / 测试出图共用）。

    与新建训练 version 用的同一套解析（`default_paths_for_new_version`）：用户在
    Settings → 模型 切换 `selected_anima`（官方 variant 或注册的本地 custom
    `.safetensors`）即同时影响这里的主权重路径——所以能「在微调权重上测试出图」。
    """
    from ..services.models import default_paths_for_new_version
    return default_paths_for_new_version()
