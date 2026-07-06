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
    blacklist_tags: Optional[list[str]] = None


class CLTaggerOverrides(BaseModel):
    """打标页对 CLTagger 设置的「本次任务覆盖」—— 仅在 worker 进程内生效。"""
    threshold_general: Optional[float] = None
    threshold_character: Optional[float] = None
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    tag_mapping_path: Optional[str] = None
    add_copyright_tag: Optional[bool] = None
    add_meta_tag: Optional[bool] = None
    add_model_tag: Optional[bool] = None
    add_rating_tag: Optional[bool] = None
    add_quality_tag: Optional[bool] = None
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
    # 落盘格式跟着产物走，不再由请求指定（老客户端传 output_format 会被忽略）：
    # LLM json preset 产出结构化 caption_json → .json；其余（本地打标 tag list /
    # LLM text preset）→ .txt。已存在的 .json 仍按 .json 更新。
    # 已有 caption 文件时的策略："overwrite"（默认，覆盖）| "skip"（保留原文件）
    # | "append"（tag 级 merge + dedupe，写回原格式）。
    on_existing: str = "overwrite"
    wd14_overrides: Optional[Wd14Overrides] = None
    cltagger_overrides: Optional[CLTaggerOverrides] = None
    llm_overrides: Optional[LLMTaggerOverrides] = None
    # 触发词；空串 / None = 不启用。打标时作为第一个 tag prepend 到 caption；
    # 同时持久化到 version.trigger_word，后续 train 阶段从私有 yaml 读出。
    trigger_word: Optional[str] = None
    # 打标范围："all"（默认，train 全部文件夹 + validation）| "validation"（只打
    # held-out 验证集）|  某个 train 子文件夹名（如 "1_data"，只打那一个）。
    # 手动加入的验证图原本没 caption，靠这个把验证集纳入打标。
    scope: str = "all"


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
    excluded_tags: list[str] = []
    auto_tag: bool = True
    # A3：reg 集自动打标选 tagger。默认 wd14（保持向后兼容）；目前 UI 暴露 wd14/cltagger
    # 两个选项。LLM / JoyCaption 单独 PR 加，因 reg 图量大，慢/贵不适合默认路径。
    auto_tag_kind: str = "wd14"
    api_source: str = "gelbooru"
    # 默认增量（用户决策 2026-05-30）：reg 集很多时候希望沿用已有 + 只补缺，
    # 不希望开始生成时把昨天好不容易拉到的图清掉。full mode 走 worker 内
    # `clear_reg_dir` 把 reg/ 整个清零（含 .deleted_ids.json）。
    incremental: bool = True
    # A4：build 完后自动跑 dedup + 不够 → incremental 补足循环（最多 N 轮）。
    # 默认开（用户决策 2026-05-30）。手动按钮 (RegPreview "自动去重") 独立保留。
    auto_dedup: bool = True
    # B1（PR-2）：构建模式
    # - mirror：镜像 train 子文件夹结构（5_concept/、1_general/ ...），每个子文件夹
    #   按 train 图数独立拉（旧行为）。target_count 此模式下被忽略。
    # - flat：所有图进 `1_data/` 单桶；target_count 指定数量（None = train 总数）。
    # 默认 flat（用户决策 2026-05-30）；mode 切换前提是 reg 集已清空（UI 拦截）。
    build_mode: str = "flat"
    # B1：flat 模式下的目标图数；None = train 总图数。mirror 下被忽略。
    target_count: Optional[int] = None
    # 注：来源（booru / AI 先验）由前端 SourcePicker 决定调哪个 endpoint
    # （/reg/build vs /reg/generate-prior），不进本 schema。RegMeta.generation_method
    # 在 ai 路径里独立写入。
    # PP5.5 进阶配置（默认值与源脚本一致；仅 booru 模式生效）
    skip_similar: bool = True
    aspect_ratio_filter_enabled: bool = False
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    postprocess_method: str = "smart"  # smart | stretch | crop
    postprocess_max_crop_ratio: float = 0.1


class RegDeleteFilesRequest(BaseModel):
    """批量删除 reg 集中的指定图片（含同名 .txt caption）。

    `relative_paths` 是相对 reg/ 的路径列表，支持跨子文件夹。
    后端会把删除的 booru ID（= 文件名 stem）追加到 `reg/.deleted_ids.json`，
    增量补足（incremental build）时自动从搜索结果里排除，防止再下回来。
    """
    relative_paths: list[str]


class RegAiRequest(BaseModel):
    """先验生成请求 —— 不含 lora_configs，先验生成不带 LoRA。"""
    excluded_tags: list[str] = []
    # 本次先验生成临时选用的底模（官方 variant key 或注册的本地 custom 路径）；
    # None → 用 Settings 里 selected_anima。只换 transformer 权重。
    base_model: Optional[str] = None
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
