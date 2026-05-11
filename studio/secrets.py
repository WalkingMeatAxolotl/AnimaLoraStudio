"""全局服务凭证 + 配置 —— 集中存到 studio_data/secrets.json。

`studio_data/` 已被 .gitignore，本文件即可放真实 token / api key。
对外通过 `to_masked_dict()` 把敏感字段以 "***" 返回；前端 PUT
时若回传 "***" 表示「保持不变」，由 `update()` 的 deep-merge 处理。
"""
from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from .paths import STUDIO_DATA

SECRETS_FILE = STUDIO_DATA / "secrets.json"
MASK = "***"
SENSITIVE_FIELDS: tuple[str, ...] = (
    "gelbooru.api_key",
    "danbooru.api_key",
    "huggingface.token",
    "wandb.api_key",
    "llm_tagger.api_key",
    "modelscope.token",
)


class GelbooruConfig(BaseModel):
    user_id: str = ""
    api_key: str = ""
    save_tags: bool = False
    convert_to_png: bool = True
    # 新装默认 true：训练里 4-channel PNG 会让 VAE 把透明区域当噪声学进去，
    # 多数情况下用户都需要去掉 alpha。已存在 secrets.json 里显式 false 不受影响。
    remove_alpha_channel: bool = True


class DanbooruConfig(BaseModel):
    """Danbooru 用 HTTP Basic auth：username + api_key（可匿名跑，但有速率限制）。"""
    username: str = ""
    api_key: str = ""
    # 账户类型决定多 tag 搜索上限（free=2 / gold=6 / platinum=12）
    account_type: str = "free"


class HuggingFaceConfig(BaseModel):
    token: str = ""
    # PR-S3: HF 模型下载端点。`""` 走 huggingface_hub 默认（直连 huggingface.co）。
    # 默认 `hf-mirror.com` —— 项目主战场是国内用户，海外用户 Settings 切换。
    # 自定义 URL 也支持（tencent / sjtug / 自建反代等）。
    # huggingface_hub>=0.20 起 hf_hub_download / snapshot_download 都支持 `endpoint=` kwarg，
    # 我们 per-call 传，不依赖 HF_ENDPOINT env var（env var 只在模块 import 时读，
    # runtime 改设置无效）。
    endpoint: str = "https://hf-mirror.com"


class WandBConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    project: str = "AnimaLoraStudio"
    entity: str = ""
    base_url: str = ""
    mode: str = "online"
    log_samples: bool = True

    @model_validator(mode="after")
    def _normalize_values(self) -> "WandBConfig":
        if self.mode not in {"online", "offline", "disabled"}:
            self.mode = "online"
        return self


class ModelScopeConfig(BaseModel):
    token: str = ""
    # 魔搭社区（modelscope.cn）下载 token。公开模型不填也能下，私有 / 限速时需要。
    # 使用前需 pip install modelscope；下载时会优先找 MODELSCOPE_REPO_MAP 里的对应仓库，
    # 没有映射的模型自动回退 HuggingFace。


class DownloadConfig(BaseModel):
    """全局下载偏好（跨渠道共享）。"""
    # 全局排除 tag：搜索时自动追加 -tag1 -tag2（gelbooru / danbooru 语法一致）
    exclude_tags: list[str] = Field(default_factory=list)
    # PP9 — Booru API 池子调速（downloader + reg_builder 共用）
    parallel_workers: int = 4
    api_rate_per_sec: float = 2.0
    cdn_rate_per_sec: float = 5.0


class JoyCaptionConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    model: str = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    prompt_template: str = "Descriptive Caption"


class LLMPromptPresetConfig(BaseModel):
    id: str
    label: str
    prompt: str
    builtin: bool = False
    output_format: str = "json"  # json | text


def _default_llm_prompt_presets() -> list[LLMPromptPresetConfig]:
    from .llm_presets import builtin_llm_presets

    return [LLMPromptPresetConfig(**item) for item in builtin_llm_presets()]


class LLMTaggerConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = ""
    model_ids: list[str] = Field(default_factory=list)
    endpoint: str = "chat_completions"  # chat_completions | responses
    prompt_preset: str = "style_json"
    prompt_presets: list[LLMPromptPresetConfig] = Field(
        default_factory=_default_llm_prompt_presets
    )
    custom_prompt: str = ""
    temperature: float = 0.2
    max_tokens: int = 700
    timeout: int = 60
    max_retries: int = 3
    max_side: int = 1280
    jpeg_quality: int = 85
    max_image_mb: float = 5.0

    @model_validator(mode="after")
    def _normalize_values(self) -> "LLMTaggerConfig":
        if self.endpoint not in {"chat_completions", "responses"}:
            self.endpoint = "chat_completions"
        builtin_presets = _default_llm_prompt_presets()
        builtin_by_id = {preset.id: preset for preset in builtin_presets}
        presets: list[LLMPromptPresetConfig] = []
        seen: set[str] = set()
        for preset in builtin_presets:
            presets.append(preset)
            seen.add(preset.id)
        for preset in self.prompt_presets or []:
            preset.id = "".join(
                ch if ch.isalnum() or ch in ("_", "-") else "_"
                for ch in str(preset.id or "").strip()
            ).strip("_")
            preset.label = str(preset.label or preset.id).strip()
            preset.prompt = str(preset.prompt or "").strip()
            preset.output_format = (
                "text" if str(preset.output_format).strip() == "text" else "json"
            )
            if preset.id in builtin_by_id:
                continue
            if not preset.id or not preset.prompt or preset.id in seen:
                continue
            preset.builtin = False
            seen.add(preset.id)
            presets.append(preset)
        if not presets:
            presets = _default_llm_prompt_presets()
        self.prompt_presets = presets
        preset_ids = {p.id for p in self.prompt_presets}
        if self.prompt_preset not in preset_ids and self.prompt_preset != "custom":
            self.prompt_preset = self.prompt_presets[0].id
        if self.model and self.model not in self.model_ids:
            self.model_ids = [self.model, *self.model_ids]
        model_seen: set[str] = set()
        clean_model_ids: list[str] = []
        for model_id in self.model_ids:
            text = str(model_id or "").strip()
            key = text.lower()
            if not text or key in model_seen:
                continue
            model_seen.add(key)
            clean_model_ids.append(text)
        self.model_ids = clean_model_ids
        self.temperature = max(0.0, min(float(self.temperature), 2.0))
        self.max_tokens = max(64, int(self.max_tokens or 700))
        self.timeout = max(5, int(self.timeout or 60))
        self.max_retries = max(1, int(self.max_retries or 3))
        self.max_side = max(64, int(self.max_side or 1280))
        self.jpeg_quality = max(1, min(100, int(self.jpeg_quality or 85)))
        self.max_image_mb = max(0.1, float(self.max_image_mb or 5.0))
        return self


# 默认 WD14 候选模型；用户可在「设置 → WD14 → 候选模型」里增删，
# 当前选中的 `model_id` 永远会被规范化进 `model_ids`（见 WD14Config validator）。
DEFAULT_WD14_MODELS: tuple[str, ...] = (
    "SmilingWolf/wd-eva02-large-tagger-v3",
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3",
    "SmilingWolf/wd-v1-4-convnext-tagger-v2",
)


class WD14Config(BaseModel):
    model_id: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    model_ids: list[str] = Field(
        default_factory=lambda: list(DEFAULT_WD14_MODELS)
    )
    local_dir: Optional[str] = None
    threshold_general: float = 0.35
    threshold_character: float = 0.85
    blacklist_tags: list[str] = Field(default_factory=list)
    # PP8 — batch 推理大小；GPU EP 时按这个走，CPU 兜底自动降到 1
    batch_size: int = 8

    @model_validator(mode="after")
    def _ensure_model_ids_invariant(self) -> "WD14Config":
        """保证 `model_id ∈ model_ids` 且候选列表不为空。

        - 列表为空（含旧 secrets.json 没这个字段然后被显式置空）→ 回填默认 4 项。
        - 当前选中的 model_id 不在列表里 → 加到列表头（用户既能跑临时模型，
          dropdown 也始终能显示当前值）。
        副作用：用户若想从候选中「删除当前选中」，需先在打标 / 设置页切到另一个
        model_id 再删；前端会强制这种顺序。
        """
        if not self.model_ids:
            self.model_ids = list(DEFAULT_WD14_MODELS)
        if self.model_id and self.model_id not in self.model_ids:
            self.model_ids = [self.model_id, *self.model_ids]
        return self


class CLTaggerConfig(BaseModel):
    model_id: str = "cella110n/cl_tagger"
    model_path: str = "cl_tagger_1_02/model.onnx"
    tag_mapping_path: str = "cl_tagger_1_02/tag_mapping.json"
    local_dir: Optional[str] = None
    threshold_general: float = 0.35
    threshold_character: float = 0.6
    add_rating_tag: bool = False
    add_model_tag: bool = False
    blacklist_tags: list[str] = Field(default_factory=list)
    # 与 WD14 一致：只有 CUDA EP 时才真正 batch，CPU 自动降到 1。
    batch_size: int = 8


class QueueConfig(BaseModel):
    """队列调度策略（PP10.2）。

    Studio supervisor 使用双槽位调度：TRAIN 槽跑训练 task，DATA 槽跑
    数据准备 job（download / tag / reg_build）。download 永远与训练并行
    （IO-only，不抢 GPU）；tag / reg_build 走 GPU，默认在训练时**推迟执行**
    避免 OOM。把 `allow_gpu_during_train` 打开后才允许并行（用户自己确认
    显存够）。
    """
    allow_gpu_during_train: bool = False


class ModelsConfig(BaseModel):
    """全局模型配置（PP7）。

    - `root`：模型存放根目录。`None/""` → 回退到 `REPO_ROOT/models/`（默认）。
      云端 / 大容量数据盘可改成绝对路径，比如 `D:/anima-models` 或 `/data/anima`。
      所有训练模型（Anima / VAE / Qwen3 / T5 tokenizer / WD14）共享这一根目录。
    - `selected_anima`：当前默认主模型 variant。Studio 创建新 version 时根据
      此字段把 `transformer_path` 写成绝对路径到 yaml；已存在 version 不动
      （保证训练重现性）。
    """
    root: Optional[str] = None
    selected_anima: str = "preview3-base"


class GenerateConfig(BaseModel):
    """测试出图 daemon 行为（PR Phase 2）。

    - `preview_every_n_steps`：中间步预览节流。0=关；>0 → daemon 用 TAEFlux
      decode 每 N 步推一张 256px JPEG 给前端。需要 TAEFlux 模型已下载
      （settings 入口或 POST /api/generate/taeflux/install）。
    - `attention_backend`：注意力后端选择。`'auto'`（默认）→ 装了什么用什么
      （优先级 flash_attn > xformers > none/SDPA）；显式值（flash_attn/
      xformers/none）则强制 —— 想 debug 或对比时手动指定。
    """
    preview_every_n_steps: int = 3
    attention_backend: str = "auto"


class Secrets(BaseModel):
    gelbooru: GelbooruConfig = Field(default_factory=GelbooruConfig)
    danbooru: DanbooruConfig = Field(default_factory=DanbooruConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    modelscope: ModelScopeConfig = Field(default_factory=ModelScopeConfig)
    # 模型下载源。"huggingface"（默认）走 HF + endpoint 配置；
    # "modelscope" 走魔搭社区，没有对应映射的模型自动回退 HF。
    download_source: str = "huggingface"
    joycaption: JoyCaptionConfig = Field(default_factory=JoyCaptionConfig)
    llm_tagger: LLMTaggerConfig = Field(default_factory=LLMTaggerConfig)
    wd14: WD14Config = Field(default_factory=WD14Config)
    cltagger: CLTaggerConfig = Field(default_factory=CLTaggerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load() -> Secrets:
    """读 secrets.json；缺失或损坏时返回默认实例（不抛错）。"""
    if not SECRETS_FILE.exists():
        return Secrets()
    try:
        return Secrets.model_validate_json(SECRETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        # 文件损坏不应阻断 Studio 启动；用默认值覆盖
        return Secrets()


def save(s: Secrets) -> None:
    SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_FILE.write_text(s.model_dump_json(indent=2), encoding="utf-8")


def get(path: str) -> Any:
    """点路径取值，例：`get('wd14.threshold_general')`。"""
    cur: Any = load()
    for seg in path.split("."):
        cur = getattr(cur, seg)
    return cur


def update(partial: dict[str, Any]) -> Secrets:
    """deep-merge `partial` 进当前持久化值；返回新 Secrets 并落盘。

    - `partial` 里 leaf 值为 MASK ("***") 时，表示「保持原值不变」。
    - 未提及的字段沿用旧值。
    """
    current_dict = load().model_dump()
    merged = _deep_merge(current_dict, partial)
    new = Secrets.model_validate(merged)
    save(new)
    return new


def to_masked_dict(s: Secrets) -> dict[str, Any]:
    """GET /api/secrets 返回此结构；敏感字段非空时替换为 MASK。"""
    d = s.model_dump()
    for path in SENSITIVE_FIELDS:
        segs = path.split(".")
        cur: Any = d
        for seg in segs[:-1]:
            cur = cur[seg]
        if cur.get(segs[-1]):
            cur[segs[-1]] = MASK
    return d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """把 patch 合并到 base：嵌套 dict 递归合并；leaf 值为 MASK 则丢弃。"""
    out = dict(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        elif val == MASK:
            # 保持旧值
            continue
        else:
            out[key] = val
    return out


def has_gelbooru_credentials() -> bool:
    """便捷：用于前端 / 端点判断是否已经配好 Gelbooru。"""
    g = load().gelbooru
    return bool(g.user_id and g.api_key)


def has_credentials_for(api_source: str) -> bool:
    """各下载渠道的「能不能跑」判定：
    - gelbooru: 必须有 user_id + api_key（API 强制要求）
    - danbooru: 匿名也能跑（仅速率受限），所以始终 True
    """
    if api_source == "gelbooru":
        return has_gelbooru_credentials()
    if api_source == "danbooru":
        return True
    return False
