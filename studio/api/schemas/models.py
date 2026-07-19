"""/api/models/* + /api/upscalers/* 请求 BaseModel（PR-6 commit 2 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ModelDownloadRequest(BaseModel):
    model_id: str           # "anima_main" | "anima_vae" | "qwen3" | "t5_tokenizer"
    variant: Optional[str] = None  # anima_main / krea2_main 使用，其他忽略


class FamilySwitchRequest(BaseModel):
    """训练配置切换模型族的预览计算（多模型 P4-3）。纯计算不落盘。"""
    target: str          # 目标族 id（"anima" / "krea2"）
    config: dict         # 当前 config dict（允许部分字段缺失）


class ModelSourceCandidateRequest(BaseModel):
    """统一模型来源候选的添加 / 移除请求（docs/design/model-source-unification.md §6）。

    POST 添加：kind=download 需 repo（单文件资产另需 filename）；kind=local 需
    path。DELETE 移除：只按身份键（download=(repo, filename)，local=path）匹配。
    """
    kind: str                # "download" | "local"
    repo: str = ""
    filename: str = ""
    path: str = ""
    extra: dict[str, str] = Field(default_factory=dict)


class UpscalerSelectRequest(BaseModel):
    label: str   # 预设 key、custom 文件名或本地绝对路径
