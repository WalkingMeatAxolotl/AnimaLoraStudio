"""全局凭证 / 服务配置（PR-6 commit 2 从 server.py 抽出）。

2 routes：
    GET /api/secrets    masked secrets snapshot（API key 等敏感字段 masked）
    PUT /api/secrets    更新 secrets，返回新 masked snapshot
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from ... import secrets

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/secrets")
def get_secrets() -> dict[str, Any]:
    return secrets.to_masked_dict(secrets.load())


@router.put("/api/secrets")
def put_secrets(body: dict[str, Any]) -> dict[str, Any]:
    new = secrets.update(body)
    # 用户在 Settings 里改了 generate.idle_timeout_minutes 后，立即同步给跑着的
    # daemon —— 不然要等下次出图 dispatch 才生效。daemon 还没起的话 set 也安全
    # （走 noop 分支，下次 dispatch 时一并应用）。
    try:
        from ...services.inference.daemon import get_daemon
        get_daemon().sync_idle_timeout_from_secrets()
    except Exception:
        logger.warning("failed to sync idle_timeout to daemon", exc_info=True)
    return secrets.to_masked_dict(new)
