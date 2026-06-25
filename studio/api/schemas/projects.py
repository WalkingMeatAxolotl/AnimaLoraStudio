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


class EvalSamplesStart(BaseModel):
    checkpoint_path: Optional[str] = None
    max_items: Optional[int] = None


class EvalClipStart(BaseModel):
    model_name: Optional[str] = None


class EvalDinoStart(BaseModel):
    model_name: Optional[str] = None
