"""/api/presets/* 请求 BaseModel（PR-5 从 server.py inline 抽出）。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DuplicateRequest(BaseModel):
    new_name: str


class PresetExportBody(BaseModel):
    config: dict[str, Any]


class PresetImportBody(BaseModel):
    filename: str


class PresetImportFromPathBody(BaseModel):
    path: str
