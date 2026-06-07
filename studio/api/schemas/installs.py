"""安装 / runtime 类 endpoint 请求 BaseModel（PR-6 commit 3 从 server.py 抽出）。

涵盖 wd14 / torch / flash-attention / llm-tagger 域。xformers 无请求 body。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class WD14InstallRequest(BaseModel):
    target: str = "auto"  # "auto" | "gpu" | "cpu" | "directml"


class TorchReinstallRequest(BaseModel):
    target: str = "auto"  # "auto" | "cu128" | "cu126" | "cu124" | "cu118" | "cpu"


class FlashAttnInstallRequest(BaseModel):
    url: Optional[str] = None  # None = 自动从 GitHub Releases 选最优


class LLMModelsRefreshRequest(BaseModel):
    # preset_id 指定要更新的 preset；不传则用当前 current_preset
    preset_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = None


class LLMConnectionTestRequest(BaseModel):
    preset_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    timeout: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
