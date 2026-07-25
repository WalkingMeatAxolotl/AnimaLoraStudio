"""/api/projects + versions CRUD 请求 BaseModel（PR-6.5 commit 1 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ProjectCreate(BaseModel):
    title: str
    slug: Optional[str] = None
    note: Optional[str] = None
    initial_version_label: Optional[str] = "v1"


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    note: Optional[str] = None
    stage: Optional[str] = None
    active_version_id: Optional[int] = None


class VersionCreate(BaseModel):
    label: str
    fork_from_version_id: Optional[int] = None
    note: Optional[str] = None


class VersionUpdate(BaseModel):
    note: Optional[str] = None
    stage: Optional[str] = None
    config_name: Optional[str] = None
    trigger_word: Optional[str] = None


class EvalManifestPut(BaseModel):
    manifest: dict[str, Any]


class EvalRunRequest(BaseModel):
    """Manual eval trigger over an explicit checkpoint set.

    `task_id` 只是溯源：从训练页发起时带上（那批评估会挂在那次训练名下），从版本的
    评估页发起时省略 —— 评估的对象是 version 下的 checkpoint，不必存在对应的训练 task。
    """
    task_id: Optional[int] = None
    checkpoints: list[str]


