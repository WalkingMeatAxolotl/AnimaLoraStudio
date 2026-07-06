"""/api/system/update 请求 BaseModel（PR-6 commit 4 从 server.py 抽出）。"""
from __future__ import annotations

from pydantic import BaseModel


class UpdateRequest(BaseModel):
    target: str = "origin/master"  # ref / commit sha / origin/branch
    force: bool = False            # True = 覆盖 dirty 工作树（用户已确认强制覆盖）
