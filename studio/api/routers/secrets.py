"""全局凭证 / 服务配置（PR-6 commit 2 从 server.py 抽出）。

2 routes：
    GET /api/secrets    masked secrets snapshot（API key 等敏感字段 masked）
    PUT /api/secrets    更新 secrets，返回新 masked snapshot
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ... import secrets

router = APIRouter()


@router.get("/api/secrets")
def get_secrets() -> dict[str, Any]:
    return secrets.to_masked_dict(secrets.load())


@router.put("/api/secrets")
def put_secrets(body: dict[str, Any]) -> dict[str, Any]:
    new = secrets.update(body)
    return secrets.to_masked_dict(new)
