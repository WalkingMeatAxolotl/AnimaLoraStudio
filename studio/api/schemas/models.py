"""/api/models/* + /api/upscalers/* 请求 BaseModel（PR-6 commit 2 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ModelDownloadRequest(BaseModel):
    model_id: str           # "anima_main" | "anima_vae" | "qwen3" | "t5_tokenizer"
    variant: Optional[str] = None  # anima_main / krea2_main 使用，其他忽略


class AnimaCustomModelRequest(BaseModel):
    path: str   # 本地 .safetensors 主模型绝对路径（PathPicker 选盘上已有文件）


class FamilySwitchRequest(BaseModel):
    """训练配置切换模型族的预览计算（多模型 P4-3）。纯计算不落盘。"""
    target: str          # 目标族 id（"anima" / "krea2"）
    config: dict         # 当前 config dict（允许部分字段缺失）


class UpscalerSelectRequest(BaseModel):
    label: str   # 预设 key 或 custom 文件名


class UpscalerCustomDownloadRequest(BaseModel):
    source: str   # "hf" | "ms"
    repo_id: str  # 例 "Kim2091/UltraSharp" 或 ModelScope 同形式
    filename: str  # 例 "4x-UltraSharp.pth"
