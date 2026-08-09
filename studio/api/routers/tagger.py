"""Tagger 可用性检查（PR-6 commit 1 从 server.py 抽出）。

1 route：
    GET /api/tagger/{name}/check    检查指定 tagger 是否可用（wd14 / cltagger / llm）
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter

from ...domain.errors import ValidationError
from ...services.tagging.base import VALID_TAGGER_NAMES, get_tagger

router = APIRouter()


@router.get("/api/tagger/{name}/check")
def check_tagger(name: str, overrides: str | None = None) -> dict[str, Any]:
    """`overrides` 是 JSON dict（与 startTag 的 `<name>_overrides` 同构）。

    check 必须与本次打标同口径：页面上的模型版本 / 预设选择是本次覆盖，
    不落盘；不带 overrides 检查全局默认，会把选了已下载模型的用户误判为
    「需下载」并锁死开始按钮（issue #477）。
    """
    if name not in VALID_TAGGER_NAMES:
        raise ValidationError(
            f'Unknown tagger: "{name}"',
            code="tag.tagger_invalid", details={"name": name}, http_status=400,
        )
    parsed: dict[str, Any] | None = None
    if overrides:
        try:
            parsed = json.loads(overrides)
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            raise ValidationError(
                "overrides must be a JSON object",
                code="tag.tagger_overrides_invalid",
                details={"field": "overrides"}, http_status=400,
            )
    try:
        t = get_tagger(name, parsed)
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "ok": False, "msg": str(exc)}
    ok, msg = t.is_available()
    return {
        "name": name,
        "ok": ok,
        "msg": msg,
        "requires_service": getattr(t, "requires_service", False),
    }
