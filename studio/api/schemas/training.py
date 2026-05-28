"""tag / captions / reg / version_config 请求 BaseModel（PR-6.5 commit 5 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class Wd14Overrides(BaseModel):
    """打标页对 wd14 设置的「本次任务覆盖」—— 仅在 worker 进程内生效，
    不写回 secrets.json。"""
    threshold_general: Optional[float] = None
    threshold_character: Optional[float] = None
    model_id: Optional[str] = None
    local_dir: Optional[str] = None
    blacklist_tags: Optional[list[str]] = None


class CLTaggerOverrides(BaseModel):
    """打标页对 CLTagger 设置的「本次任务覆盖」—— 仅在 worker 进程内生效。"""
    threshold_general: Optional[float] = None
    threshold_character: Optional[float] = None
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    tag_mapping_path: Optional[str] = None
    local_dir: Optional[str] = None
    add_rating_tag: Optional[bool] = None
    add_model_tag: Optional[bool] = None
    blacklist_tags: Optional[list[str]] = None


class LLMTaggerOverrides(BaseModel):
    """打标页对 LLM tagger 设置的「本次任务覆盖」—— 仅在 worker 进程内生效。

    - `current_preset`：切换 active preset id
    - 其余字段：覆盖 active preset 的同名字段
    - `api_key` 不允许 override（避免出现在 task params/日志）
    """
    current_preset: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    prompt: Optional[str] = None
    output_format: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None
    concurrency: Optional[int] = None
    requests_per_second: Optional[float] = None
    max_requests_per_minute: Optional[int] = None
    max_side: Optional[int] = None
    jpeg_quality: Optional[int] = None
    max_image_mb: Optional[float] = None


class TagJobRequest(BaseModel):
    tagger: str = "wd14"
    output_format: str = "txt"                # "txt" | "json"
    wd14_overrides: Optional[Wd14Overrides] = None
    cltagger_overrides: Optional[CLTaggerOverrides] = None
    llm_overrides: Optional[LLMTaggerOverrides] = None
    # 触发词；空串 / None = 不启用。打标时作为第一个 tag prepend 到 caption；
    # 同时持久化到 version.trigger_word，后续 train 阶段从私有 yaml 读出。
    trigger_word: Optional[str] = None


class CaptionEdit(BaseModel):
    tags: list[str]


class CommitItem(BaseModel):
    folder: str
    name: str
    tags: list[str]


class CommitRequest(BaseModel):
    items: list[CommitItem]


class BatchOp(BaseModel):
    op: str                                   # add|remove|replace|dedupe|stats
    scope: dict[str, Any]                     # {kind, folder?, names?}
    tags: Optional[list[str]] = None          # add/remove
    old: Optional[str] = None                 # replace
    new: Optional[str] = None                 # replace
    position: Optional[str] = "back"          # add: front|back
    top: int = 50                             # stats


class RegBuildRequest(BaseModel):
    # 目标数量永远 = train 总数（与源脚本一致），UI 不暴露
    excluded_tags: list[str] = []
    auto_tag: bool = True
    api_source: str = "gelbooru"
    incremental: bool = False  # PP5.1：补足 — 不清空已有图，只补缺口
    # PP5.5 进阶配置（默认值与源脚本一致）
    skip_similar: bool = True
    aspect_ratio_filter_enabled: bool = False
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    postprocess_method: str = "smart"  # smart | stretch | crop
    postprocess_max_crop_ratio: float = 0.1


class RegAiRequest(BaseModel):
    """先验生成请求 —— 不含 lora_configs，先验生成不带 LoRA。"""
    excluded_tags: list[str] = []
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 4.0
    sampler_name: str = "er_sde"
    scheduler: str = "simple"
    seed: int = 0
    incremental: bool = False
    mixed_precision: str = "bf16"


class FromPresetRequest(BaseModel):
    name: str  # 全局 preset 名


class SaveAsPresetRequest(BaseModel):
    name: str
    overwrite: bool = False
