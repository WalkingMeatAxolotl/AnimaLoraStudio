"""bundle / train 导出导入 BaseModel（PR-6.5 commit 2 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, model_validator

from ...services.data_io import train_io


class BundleOptionsBody(BaseModel):
    train: bool = True
    train_captions: bool = True
    reg: bool = False
    reg_captions: bool = False
    include_config: bool = False

    def to_options(self) -> train_io.BundleOptions:
        return train_io.BundleOptions(
            train=self.train,
            train_captions=self.train_captions,
            reg=self.reg,
            reg_captions=self.reg_captions,
            include_config=self.include_config,
        )


class BundleImportBody(BaseModel):
    path: Optional[str] = None
    filename: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "BundleImportBody":
        if sum(bool(v) for v in (self.path, self.filename)) != 1:
            raise ValueError("exactly one of path or filename is required")
        return self
