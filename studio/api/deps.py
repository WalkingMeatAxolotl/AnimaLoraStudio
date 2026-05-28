"""共享 dependency helpers（PR-6 起从 server.py 抽出）。

跨 router 共用的 helper：拿 Supervisor 实例、版本 / 项目记录验证等。
未来会改 FastAPI `Depends(...)` 形式，本次先维持「直接调函数」风格保
零行为变更。
"""
from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

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
        raise HTTPException(503, "supervisor not running")
    return sup
