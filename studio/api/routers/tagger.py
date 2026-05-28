"""Tagger 可用性检查（PR-6 commit 1 从 server.py 抽出）。

1 route：
    GET /api/tagger/{name}/check    检查指定 tagger 是否可用（wd14 / cltagger / llm）
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ...services.tagging.base import VALID_TAGGER_NAMES, get_tagger

router = APIRouter()


@router.get("/api/tagger/{name}/check")
def check_tagger(name: str) -> dict[str, Any]:
    if name not in VALID_TAGGER_NAMES:
        raise HTTPException(400, f"unknown tagger: {name}")
    try:
        t = get_tagger(name)
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "ok": False, "msg": str(exc)}
    ok, msg = t.is_available()
    return {
        "name": name,
        "ok": ok,
        "msg": msg,
        "requires_service": getattr(t, "requires_service", False),
    }
