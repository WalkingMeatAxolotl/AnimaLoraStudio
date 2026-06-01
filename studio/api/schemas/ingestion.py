"""download / upload / preprocess BaseModel（PR-6.5 commit 3 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ...services.preprocess import core as preprocess_svc


class DownloadRequest(BaseModel):
    tag: str
    count: int = 20
    api_source: str = "gelbooru"


class EstimateRequest(BaseModel):
    tag: str
    api_source: str = "gelbooru"


class UploadFromPathBody(BaseModel):
    path: str


class PreprocessStartRequest(BaseModel):
    mode: str = "all"  # all | selected | all_force
    names: Optional[list[str]] = None
    model: str = preprocess_svc.DEFAULT_MODEL
    tile_size: int = preprocess_svc.DEFAULT_TILE_SIZE
    tile_pad: int = preprocess_svc.DEFAULT_TILE_PAD
    device: str = preprocess_svc.DEFAULT_DEVICE
    # target_area=None 走纯 4× 模型；非 None 走智能（够大跳模型 + LANCZOS 缩到目标）
    target_area: Optional[int] = preprocess_svc.DEFAULT_TARGET_AREA


class PreprocessRestoreRequest(BaseModel):
    """还原已处理图：删 manifest entry + 删 preprocess/{name} PNG。

    还原后该图回到「隐式 original」状态——下游 resolver 重新指向 download/。
    见 ADR 0004。
    """
    names: list[str]


class CropRect(BaseModel):
    """归一化裁剪矩形 [0..1]^4。x/y = 左上角，w/h = 宽高。"""
    x: float
    y: float
    w: float
    h: float
    label: Optional[str] = None


class PreprocessCropRequest(BaseModel):
    """裁剪 job 输入：源文件名 → 一个或多个归一化矩形。

    源文件名为 preprocess/ 下当前文件名（或 download/ 文件名兜底，若 preprocess/
    没对应）。每个矩形产出一张 PNG：N=1 覆盖 stem.png；N>1 输出 stem_c0.png /
    stem_c1.png / ... 并删除原 stem.png。
    """
    crops: dict[str, list[CropRect]]
