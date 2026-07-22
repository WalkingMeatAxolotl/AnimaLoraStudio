"""全局服务凭证 + 配置 —— 集中存到 studio_data/secrets.json。

`studio_data/` 已被 .gitignore，本文件即可放真实 token / api key。
对外通过 `to_masked_dict()` 把敏感字段以 "***" 返回；前端 PUT
时若回传 "***" 表示「保持不变」，由 `update()` 的 deep-merge 处理。
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from .paths import STUDIO_DATA

SECRETS_FILE = STUDIO_DATA / "secrets.json"
MASK = "***"
# 点路径 + `*` 通配支持：`llm_tagger.presets.*.api_key` 会遍历 list 内每个 dict。
SENSITIVE_FIELDS: tuple[str, ...] = (
    "gelbooru.api_key",
    "danbooru.api_key",
    "huggingface.token",
    "wandb.presets.*.api_key",
    "llm_tagger.presets.*.api_key",
    "modelscope.token",
)


class GelbooruConfig(BaseModel):
    user_id: str = ""
    api_key: str = ""


class DanbooruConfig(BaseModel):
    """Danbooru HTTP Basic auth：username + api_key。

    PR #38 起强制绑定（不再允许匿名）：
    - Danbooru 挂了 Cloudflare 后，匿名 UA 已不可靠（CF 可能随时收紧）
    - 强制账户让我们 UA 带 (by username)，CF 拦匿名时不会一锅端
    - danbooru 端按账户配速率上限（标准 2 req/s，高于匿名）
    """
    username: str = ""
    api_key: str = ""
    # 账户类型决定多 tag 搜索上限（free=2 / gold=6 / platinum=12）
    account_type: str = "free"


class HuggingFaceConfig(BaseModel):
    token: str = ""
    # PR-S3: HF 模型下载端点。`""` 走 huggingface_hub 默认（直连 huggingface.co）。
    # 0.8.2 hotfix：默认从 `hf-mirror.com` 切回 `""`（HF 官方）。hf-mirror 当前
    # 在所有 huggingface_hub 版本下均触发 `FileMetadataError`（commit_hash None），
    # 国内用户走 ModelScope 或自建反代；hf-mirror preset 暂从 UI 隐藏，但 endpoint
    # 字段仍接受任意 URL（用户可手动粘贴）。复查清单见 docs/todo/hf-mirror-recheck.md。
    # 自定义 URL 也支持（tencent / sjtug / 自建反代等）。
    # huggingface_hub>=0.20 起 hf_hub_download / snapshot_download 都支持 `endpoint=` kwarg，
    # 我们 per-call 传，不依赖 HF_ENDPOINT env var（env var 只在模块 import 时读，
    # runtime 改设置无效）。
    endpoint: str = ""


class WandBPresetConfig(BaseModel):
    """一套 WandB 账号 + 上传策略预设（对齐 LLMPresetConfig 的预设模式）。

    0.18 起 WandB 配置预设化：顶层 WandBConfig 只留 enabled + 预设切换，
    账号（api_key/entity/base_url）和上传策略全部下沉到 preset，可整套切换。
    """
    id: str = "default"
    label: str = "Default"
    api_key: str = ""
    project: str = "AnimaLoraStudio"
    entity: str = ""
    base_url: str = ""
    mode: str = "online"
    # 默认开 — wandb 启用时一并上传采样图，省得用户每次额外勾一次。私有 IP / NSFW
    # 数据集请在 Settings 里关掉这个开关；关了之后只上传指标，图片不出本机。
    log_samples: bool = True
    # 上传前缩到最长边像素；原图常 2K+，512 已足够 wandb 面板浏览，省流量。
    sample_max_side: int = 512
    # step 节流：>0 时只在 `global_step % N == 0` 上传，避免长训练上 GB 级图。
    # 0 = 不额外节流（按训练循环已有 sample 频率上传），baseline / epoch 边界始终上传。
    sample_every_n_steps: int = 0
    # Artifact 上传：模型 / 训练状态上传到 wandb Artifacts，方便云端管理和版本追踪。
    # policy = "all" 保留全部版本，"last" 只保留最新一份（上传新的后删除旧版本）。
    upload_model: bool = False
    upload_model_policy: str = "last"
    upload_state_manual: bool = False
    upload_state_manual_policy: str = "last"
    upload_state_auto: bool = False
    upload_state_auto_policy: str = "last"

    @model_validator(mode="after")
    def _normalize_values(self) -> "WandBPresetConfig":
        self.id = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_"
            for ch in str(self.id or "").strip()
        ).strip("_") or "default"
        self.label = str(self.label or self.id).strip()
        if self.mode not in {"online", "offline", "disabled"}:
            self.mode = "online"
        self.sample_max_side = max(64, int(self.sample_max_side or 512))
        self.sample_every_n_steps = max(0, int(self.sample_every_n_steps or 0))
        _valid_policies = {"all", "last"}
        if self.upload_model_policy not in _valid_policies:
            self.upload_model_policy = "last"
        if self.upload_state_manual_policy not in _valid_policies:
            self.upload_state_manual_policy = "last"
        if self.upload_state_auto_policy not in _valid_policies:
            self.upload_state_auto_policy = "last"
        return self


class WandBConfig(BaseModel):
    """全局 WandB：顶层只留总开关 + 当前预设指针，字段全在 preset 里。

    老扁平 schema（enabled + 平铺字段）由 _migrate_legacy_schema 包成
    id="default" 的单 preset。训练进程经 supervisor 注入 WANDB_* env 读
    `active` 预设 —— secrets 不落任何 yaml。
    """
    enabled: bool = False
    current_preset: str = "default"
    presets: list[WandBPresetConfig] = Field(
        default_factory=lambda: [WandBPresetConfig()]
    )

    @model_validator(mode="after")
    def _normalize_values(self) -> "WandBConfig":
        # id 去重保序 + 保底至少一个 preset + current 指向存在的 id
        merged: list[WandBPresetConfig] = []
        seen: set[str] = set()
        for preset in self.presets:
            if preset.id and preset.id not in seen:
                merged.append(preset)
                seen.add(preset.id)
        if not merged:
            merged = [WandBPresetConfig()]
        self.presets = merged
        if self.current_preset not in {p.id for p in self.presets}:
            self.current_preset = self.presets[0].id
        return self

    @property
    def active(self) -> WandBPresetConfig:
        """当前选中的 preset；validator 保证至少有一个。"""
        for preset in self.presets:
            if preset.id == self.current_preset:
                return preset
        return self.presets[0]


class ModelScopeConfig(BaseModel):
    token: str = ""
    # 魔搭社区（modelscope.cn）下载 token。公开模型不填也能下，私有 / 限速时需要。
    # 使用前需 pip install modelscope；下载时会优先找 MODELSCOPE_REPO_MAP 里的对应仓库，
    # 没有映射的模型自动回退 HuggingFace。


class EvalMetricModelsConfig(BaseModel):
    """LoRA eval metric model defaults.

    Metric API callers may still pass `model_name` explicitly. Empty request
    values fall back to these defaults so server-local ModelScope/HF cache paths
    do not need to be repeated for every metric run.
    """
    clip_model_name: str = "openai/clip-vit-base-patch32"
    dino_model_name: str = "facebook/dinov2-small"
    ccip_model_name: str = "ccip-caformer-24-randaug-pruned"
    # 启用哪些评估指标（Settings 复选框，见 eval_registry）。eval 只算勾选的；
    # 默认保留现有三指标，anime 域新指标（ccip_i / tag_recall）默认关、需用户开。
    enabled_metrics: list[str] = Field(
        default_factory=lambda: ["clip_t", "clip_i", "dino_i"]
    )
    # baseline 对照：训练后评估额外出一组纯底模(lora_scale=0)同 prompt/seed 图，
    # 各指标给出 Δ = checkpoint − baseline（解「绝对值难解读」）。每 task 一次。
    # 评估统一在训练后跑（inline / checkpoint-trigger 已移除）；是否评估由每个
    # version 训练配置的 eval_validation_enabled 决定。
    eval_baseline_enabled: bool = True


class DownloadConfig(BaseModel):
    """全局下载偏好（跨渠道共享）。"""
    # 全局排除 tag：搜索时自动追加 -tag1 -tag2（gelbooru / danbooru 语法一致）
    exclude_tags: list[str] = Field(default_factory=list)
    # PP9 — Booru API 池子调速（downloader + reg_builder 共用）
    parallel_workers: int = 4
    api_rate_per_sec: float = 2.0
    cdn_rate_per_sec: float = 5.0
    # 图片入库处理（曾挂在 gelbooru 下，实际被所有 booru 下载 / reg / 本地上传共用）：
    save_tags: bool = False
    convert_to_png: bool = True
    # 新装默认 true：训练里 4-channel PNG 会让 VAE 把透明区域当噪声学进去，
    # 多数情况下用户都需要去掉 alpha。已存在 secrets.json 里显式 false 不受影响。
    remove_alpha_channel: bool = True


class RegConfig(BaseModel):
    """正则集生成偏好（全局默认）。"""
    # 全局默认排除 tag：正则集生成页进入某个 build 且尚无本地选择时，用这份列表
    # 做初始排除（种子）。仅作初值，用户进页面后仍可逐 tag 增删，不影响已有 build 的
    # 本地记录。前端按本页约定归一到 booru 形态（下划线）后存进 excluded 集。
    default_excluded_tags: list[str] = Field(default_factory=list)


LLM_MESSAGE_ROLES: tuple[str, ...] = ("system", "user", "assistant")
LLM_MESSAGE_TYPES: tuple[str, ...] = ("text", "image")


class LLMMessage(BaseModel):
    """LLM payload 里的一条消息。

    type=text：普通文本消息，需指定 role (system/user/assistant)；content 为 prompt 文本
    type=image：图片占位 item，打标时后端把当前图片塞进这里
        - content 字段被忽略
        - role 固定为 "user"（OpenAI / Anthropic 都把 image 放在 user 侧）
        - 每个 preset 必须恰好有一个 type=image item（validator 兜底）
    """
    type: str = "text"
    role: str = "user"
    content: str = ""

    @model_validator(mode="after")
    def _normalize(self) -> "LLMMessage":
        if self.type not in LLM_MESSAGE_TYPES:
            self.type = "text"
        if self.type == "image":
            self.role = "user"
            self.content = ""
        else:
            if self.role not in LLM_MESSAGE_ROLES:
                self.role = "user"
        return self


def _default_messages_for(prompt: str) -> list["LLMMessage"]:
    """老 prompt 字段一行迁移 → [{system, prompt}, {image}]。"""
    msgs: list[LLMMessage] = []
    if prompt:
        msgs.append(LLMMessage(type="text", role="system", content=prompt))
    msgs.append(LLMMessage(type="image"))
    return msgs


class LLMPresetConfig(BaseModel):
    """完整 LLM tagger 预设：每条 preset 承载一整套 endpoint + messages + 生成参数。

    messages 是 OpenAI chat-completions 风格的消息序列，外加一个特殊 type=image item
    标记图片应当插入的位置。打标时后端按 messages 顺序铺开成 API payload。

    builtin: bool 仅标识 id 是否来自 builtin 列表（用于 UI 显示「重置为默认」）
    —— 不锁字段，用户改 builtin preset 的任何字段都会持久化。
    """
    id: str
    label: str = ""
    builtin: bool = False
    # endpoint 身份
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    model_ids: list[str] = Field(default_factory=list)
    endpoint: str = "chat_completions"  # chat_completions | responses
    # prompt 消息序列（含图片位置）
    messages: list[LLMMessage] = Field(default_factory=lambda: _default_messages_for(""))
    output_format: str = "json"  # json | text
    # Assist tagging: pre-tag images with a local ONNX tagger before LLM calls,
    # then inject tags into {{tags}} placeholders in text messages.
    assist_tagger: str = ""  # "" | wd14 | cltagger
    # 生成参数
    temperature: float = 0.2
    max_tokens: int = 700
    # 图片处理
    max_side: int = 1280
    jpeg_quality: int = 85
    max_image_mb: float = 5.0
    # 重试 / 超时
    timeout: int = 60
    max_retries: int = 3
    # 请求池 / 节流
    concurrency: int = 1
    requests_per_second: float = 0.0
    max_requests_per_minute: int = 0

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_prompt(cls, data: Any) -> Any:
        """兼容旧 schema 的 prompt: str → messages list。"""
        if not isinstance(data, dict):
            return data
        if "messages" in data and data["messages"]:
            return data
        legacy_prompt = str(data.pop("prompt", "") or "").strip()
        data["messages"] = [m.model_dump() if isinstance(m, LLMMessage) else m
                            for m in _default_messages_for(legacy_prompt)]
        return data

    @model_validator(mode="after")
    def _normalize_values(self) -> "LLMPresetConfig":
        self.id = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_"
            for ch in str(self.id or "").strip()
        ).strip("_")
        self.label = str(self.label or self.id).strip()
        if self.endpoint not in {"chat_completions", "responses"}:
            self.endpoint = "chat_completions"
        if self.output_format not in {"json", "text"}:
            self.output_format = "json"
        if self.assist_tagger not in {"", "wd14", "cltagger"}:
            self.assist_tagger = ""
        self.temperature = max(0.0, min(float(self.temperature), 2.0))
        self.max_tokens = max(64, int(self.max_tokens or 700))
        self.timeout = max(5, int(self.timeout or 60))
        self.max_retries = max(1, int(self.max_retries or 3))
        self.concurrency = max(1, min(8, int(self.concurrency or 1)))
        self.requests_per_second = max(
            0.0,
            min(60.0, float(self.requests_per_second or 0.0)),
        )
        self.max_requests_per_minute = max(
            0,
            min(3600, int(self.max_requests_per_minute or 0)),
        )
        self.max_side = max(64, int(self.max_side or 1280))
        self.jpeg_quality = max(1, min(100, int(self.jpeg_quality or 85)))
        self.max_image_mb = max(0.1, float(self.max_image_mb or 5.0))
        # 当前选中的 model 始终出现在候选列表头部（与 WD14Config 一致）
        if self.model and self.model not in self.model_ids:
            self.model_ids = [self.model, *self.model_ids]
        seen: set[str] = set()
        clean: list[str] = []
        for mid in self.model_ids:
            text = str(mid or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            clean.append(text)
        self.model_ids = clean
        # messages 兜底：必须恰好一个 type=image item；缺则补到末尾
        if not self.messages:
            self.messages = _default_messages_for("")
        else:
            has_image = any(m.type == "image" for m in self.messages)
            if not has_image:
                self.messages = [*self.messages, LLMMessage(type="image")]
            else:
                # 多个 image → 只保留第一个
                kept: list[LLMMessage] = []
                seen_image = False
                for m in self.messages:
                    if m.type == "image":
                        if seen_image:
                            continue
                        seen_image = True
                    kept.append(m)
                self.messages = kept
        return self


def _default_llm_presets() -> list[LLMPresetConfig]:
    from .llm_presets import builtin_llm_presets

    return [LLMPresetConfig(**item) for item in builtin_llm_presets()]


class LLMTaggerConfig(BaseModel):
    """LLM tagger 顶层配置：只保留 \"当前选中 preset id\" + \"preset 列表\"。

    所有 endpoint / prompt / 生成参数都下沉到 LLMPresetConfig。
    """
    current_preset: str = "style_json"
    presets: list[LLMPresetConfig] = Field(default_factory=_default_llm_presets)

    @model_validator(mode="after")
    def _normalize_values(self) -> "LLMTaggerConfig":
        from .llm_presets import BUILTIN_PRESET_ORDER, builtin_llm_presets

        builtin_defaults = {item["id"]: item for item in builtin_llm_presets()}
        user_by_id = {p.id: p for p in self.presets if p.id}

        merged: list[LLMPresetConfig] = []
        seen_ids: set[str] = set()
        # 1) 按 builtin 顺序排列：用户改过的覆盖 builtin default；缺失则补回 default
        for bid in BUILTIN_PRESET_ORDER:
            if bid in user_by_id:
                preset = user_by_id[bid]
                preset.builtin = True
                merged.append(preset)
                seen_ids.add(bid)
            elif bid in builtin_defaults:
                preset = LLMPresetConfig(**builtin_defaults[bid])
                preset.builtin = True
                merged.append(preset)
                seen_ids.add(bid)
        # 2) 追加用户自定义 preset（id 不在 builtin 列表）
        for preset in self.presets:
            if preset.id and preset.id not in seen_ids:
                preset.builtin = False
                merged.append(preset)
                seen_ids.add(preset.id)
        if not merged:
            merged = _default_llm_presets()
        self.presets = merged
        preset_ids = {p.id for p in self.presets}
        if self.current_preset not in preset_ids:
            self.current_preset = self.presets[0].id
        return self

    @property
    def active(self) -> LLMPresetConfig:
        """当前选中的 preset；validator 保证至少有一个。"""
        for preset in self.presets:
            if preset.id == self.current_preset:
                return preset
        return self.presets[0]


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
    threshold_general: float = 0.35
    threshold_character: float = 0.6
    # CLTagger 模型输出 8 个 category：General / Character 走阈值过滤，其余 6 个
    # 按 bool 开关 gate。默认勾上 General / Character / Copyright 三类——LoRA
    # 训练标准 caption 形态；Artist / Meta / Model / Rating / Quality 默认关，
    # 避免污染 caption（画师名以及 "highres", "best quality", "explicit" 这类元信息）。
    add_copyright_tag: bool = True
    add_artist_tag: bool = False
    add_meta_tag: bool = False
    add_model_tag: bool = False
    add_rating_tag: bool = False
    add_quality_tag: bool = False
    blacklist_tags: list[str] = Field(default_factory=list)
    # 与 WD14 一致：只有 CUDA EP 时才真正 batch，CPU 自动降到 1。
    batch_size: int = 8


class QueueConfig(BaseModel):
    """队列调度策略（R-1 资源档位模型，docs/design/queue-resource-model-0.17.md）。

    工作项分三档：exclusive（训练/正则 AI/出图/评估出图，底模级显存，全系统
    同时只跑 1 个，永不并行）、light（打标/超分/正则构建/评估指标，数百 MB
    小模型）、io（下载，恒放行）。

    - `light_tasks_during_train`：exclusive 任务运行时是否允许 light 档并行。
      默认开启——轻量任务只加载小模型。独占档不受此开关影响（老开关
      `allow_gpu_during_train` 会连评估出图一起放行，是 OOM 隐患，已废弃；
      语义变化故不迁移旧值）。
    """
    light_tasks_during_train: bool = True


class ModelsConfig(BaseModel):
    """全局模型配置（PP7）。

    - `root`：模型存放根目录。`None/""` → 回退到 `REPO_ROOT/models/`（默认）。
      云端 / 大容量数据盘可改成绝对路径，比如 `D:/anima-models` 或 `/data/anima`。
      所有训练模型（Anima / VAE / Qwen3 / T5 tokenizer / WD14）共享这一根目录。
    - `selected_anima`：当前默认主模型。可为官方 variant key（`1.0` 等）**或**
      `custom_anima_paths` 里某个本地 `.safetensors` 绝对路径。Studio 创建新
      version 时根据此字段把 `transformer_path` 写成绝对路径到 yaml；已存在
      version 不动（保证训练重现性）。
    - `custom`：按模型族保存用户通过 PathPicker 注册的本地主模型权重。
      `custom_anima_paths` 保留为旧客户端兼容读写面。仅注册路径，不下载、
      不复制；条目失效时解析自动回退到官方 variant。
    - `selected_upscaler`：预处理默认放大器。可为预设 label（如 "4x-AnimeSharp"）
      或自定义/上传的文件名（如 "my-anime-model.pth"）。空串/None → 用
      DEFAULT_UPSCALER 兜底。
    - `auto_sync_paths`：fork 预设到 version 时，是否自动用全局模型路径覆盖
      预设里的 4 个模型字段（transformer / vae / text_encoder / t5_tokenizer）。
      ON（默认）→ 多数用户：永不碰 4 字段，fork 始终用 Settings 全局值；
      4 字段在项目页 / 预设页 UI 上 disabled。
      OFF → 独立模型用户：fork 时尊重预设值，4 字段可编辑 + picker。
    """
    root: Optional[str] = None
    # per-family 选中主模型（多模型 PR-4）：family_id → variant key 或 custom 路径。
    # 老键 selected_anima 由 before-validator 迁移（settings PUT 的 merged dict
    # 会同时带两键——入站 selected_anima 优先，覆盖 merge 进来的旧 selected）。
    selected: dict[str, str] = Field(default_factory=lambda: {"anima": "1.0"})
    # per-family 选中文本编码器 variant（krea2："bf16"|"fp8"，缺失=bf16）。
    # 决定训练新建 version 的 text_encoder_path 默认 + 测试出图 TE 默认；
    # 已存在 version 的 config 不动（训练重现性，与 selected 同口径）。
    selected_te: dict[str, str] = Field(default_factory=dict)
    # per-family 本地主模型路径。老键 custom_anima_paths 由 validator 迁移，
    # computed_field 保留旧客户端读面。
    custom: dict[str, list[str]] = Field(default_factory=dict)
    selected_upscaler: str = "4x-AnimeSharp"
    auto_sync_paths: bool = True

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_model_fields(cls, data):
        """迁移 Anima 老键，同时保留按 family 的新结构。"""
        if isinstance(data, dict) and (
            "selected_anima" in data or "custom_anima_paths" in data
        ):
            data = dict(data)
            if "selected_anima" in data:
                legacy_selected = data.pop("selected_anima")
                selected = dict(data.get("selected") or {})
                if legacy_selected:
                    selected["anima"] = str(legacy_selected)
                data["selected"] = selected
            if "custom_anima_paths" in data:
                legacy_custom = data.pop("custom_anima_paths")
                custom = dict(data.get("custom") or {})
                custom["anima"] = list(legacy_custom or [])
                data["custom"] = custom
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def selected_anima(self) -> str:
        """兼容读面（前端 settings 读 + dump 落盘回显）；写请走 selected。"""
        return self.selected.get("anima") or "1.0"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def custom_anima_paths(self) -> list[str]:
        """兼容旧客户端的 Anima 本地模型列表；写请走 custom。"""
        return list(self.custom.get("anima") or [])


class GenerateConfig(BaseModel):
    """测试出图 daemon 行为（PR Phase 2）。

    - `preview_every_n_steps`：中间步预览节流。0=关；>0 → daemon 用 TAEFlux
      decode 每 N 步推一张 256px JPEG 给前端。需要 TAEFlux 模型已下载
      （settings 入口或 POST /api/generate/taeflux/install）。
    - `attention_backend`：注意力后端选择。`'auto'`（默认）→ 装了什么用什么
      （优先级 flash_attn > xformers > none/SDPA）；显式值（flash_attn/
      xformers/none）则强制 —— 想 debug 或对比时手动指定。
    - `idle_timeout_minutes`：daemon 闲置 N 分钟自动卸载模型释放 VRAM。
      0 = 关闭，模型常驻直到用户手动清。计时只在 daemon idle + 模型已 load
      时跑；进 busy / 已 unload 时取消。
    - `vae_precision`：测试出图 VAE decode 精度。`'bf16'`（默认）对齐 ComfyUI
      在现代 GPU 上的 auto VAE dtype；`'fp32'` 全精度 decode（显存高峰更大，
      daemon 会在 decode 前临时 offload DiT/Qwen 腾显存）。
    - `lora_merge_precision`：FP8 底模 LoRA merge 的临时 delta 计算精度。
      `'fp32'`（默认）对齐 ComfyUI；`'bf16'` 降低 delta 计算量、通常更快，
      但总体峰值未必下降，结果也可能与 ComfyUI 有轻微数值差异。只影响
      加载/切换 LoRA，不影响采样精度。
    - `save_test_images`：开关测试出图自动落盘。默认关；开后每次出完图前端
      会调 /api/generate/save 把成图存到 studio_data/test/<date>/{single,xy}/
      image_N.png（N 按当前文件夹已有最大编号+1）。compare 模式不落盘。
    - `vram_policy`：测试出图显存策略（krea2 生效）。`'auto'`（默认）按空闲
      显存决定文本编码器与 DiT 是否让位；`'save_vram'` 强制顺序化（峰值最
      低，每图多几秒搬运）；`'performance'` 全部常驻显存（峰值最高、零搬运）。
    - `ram_guard`：内存/显存水位保护。加载大模型前按权重文件实际大小
      预算系统内存与 GPU 空闲显存，任一不足时中止并报可操作错误
      （默认开；显存检查可拦多进程叠加）；关闭后资源不足时继续加载，
      可能触发整机换页卡顿。
    - `task_timeout_minutes`：出图任务超时兜底。任务开始后超 N 分钟未
      完成 → 强制终止 daemon 进程（卡死场景协议级取消无效，只能进程级
      kill；下次任务自动重启）。0（默认）= 关闭。
    """
    preview_every_n_steps: int = 3
    attention_backend: str = "auto"
    vae_precision: str = "bf16"
    lora_merge_precision: Literal["fp32", "bf16"] = "fp32"
    idle_timeout_minutes: int = 10
    save_test_images: bool = False
    vram_policy: str = "auto"
    ram_guard: bool = True
    #: 换出到内存的 DiT 层数（0=关闭，krea2 生效）。与 vram_policy 分工不同：
    #: 前者管模型之间谁让位，本项管单个 DiT 内部——单个模型自己就装不下显存时
    #: 唯一的办法。见 docs/design/block-swap.md。
    blocks_to_swap: int = 0
    task_timeout_minutes: int = 0


class SystemConfig(BaseModel):
    """系统级偏好（ADR 0002 / 0005）。

    - `update_channel`：用户订阅哪条更新轨道。"stable"（默认）= 只看稳定版
      更新提示；"dev" = 看 dev 通道（最近 commit 时间线、可切到 dev HEAD）。
      这是**用户视图偏好**，与 git 工作树状态解耦 —— 切 toggle 不触发任何
      git 操作；真正"切到 dev HEAD" / "更新到 vX.Y.Z" 是单独按钮。
    - `show_dev_channel`：deprecated，由 `_migrate_legacy_schema` 一次性迁移成
      `update_channel`（true → "dev"，false → "stable"），保留字段以便旧
      secrets.json 读取时 pydantic 不报错；新代码不要再用。
    - `enable_automagic_v2`：实验性 feature flag。Automagic v2（fused backward）
      未正式发布，UI 默认隐藏 automagic_variant 字段（/api/schema 动态打 hidden）。
      Settings 页**故意不渲染**这个开关 —— 只能手改 secrets.json 启用；CLI/yaml
      路径不受影响（validate 仍拦 grad_accum/fp16 等不兼容组合）。
    """
    update_channel: str = "stable"  # "stable" / "dev"
    show_dev_channel: bool = False  # deprecated, 仅作迁移源
    enable_automagic_v2: bool = False  # 实验性：文件级开关，UI 不暴露


class ProxyConfig(BaseModel):
    """全局 HTTP/HTTPS 代理配置。"""
    enabled: bool = False
    http_proxy: str = ""  # 例如: http://127.0.0.1:7890
    https_proxy: str = ""
    no_proxy: str = ""    # 例外地址，如 localhost,127.0.0.1


# 按类型分别选下载源的 key（双源类型）。固定 HF 的（cltagger / t5 / taeflux）
# 不在此列，路由强制 HF。training = anima 主+VAE + qwen3 + t5 这一整组训练前置。
DOWNLOAD_SOURCE_TYPES: tuple[str, ...] = ("training", "wd14", "upscaler")
DOWNLOAD_SOURCE_VALUES: tuple[str, ...] = ("huggingface", "modelscope")


# ---------------------------------------------------------------------------
# 统一模型来源候选（docs/design/model-source-unification.md）
# ---------------------------------------------------------------------------


class SourceCandidate(BaseModel):
    """用户添加的模型来源候选。

    - kind="download"：`repo`（HF/MS repo id）+ 单文件资产另需 `filename`
      （upscaler / 主模型）；目录型资产（wd14 / eval / cltagger）只有 repo。
    - kind="local"：`path` 本地绝对路径（文件或目录）。永不被删除文件，
      移除只是移出候选列表。
    - `extra`：域特有键（cltagger：model_path / tag_mapping_path 相对 repo
      根的双文件路径）。
    """
    kind: Literal["download", "local"]
    repo: str = ""
    filename: str = ""
    path: str = ""
    extra: dict[str, str] = Field(default_factory=dict)

    def identity(self) -> tuple[str, str, str]:
        """去重身份键：download=(repo, filename)，local=(path,)。"""
        if self.kind == "download":
            return ("download", self.repo, self.filename)
        return ("local", self.path, "")


# repo 型 domain（选中值 = repo id 或本地绝对路径）。这些 domain 参与
# 「选中值不在内置也不在候选 → 自动补候选」的统一不变量；upscaler
# （文件名语义 + 扫盘兜底）与主模型族（families 注册表在 services 层，
# 解析回退逻辑健全）不在此列，其候选完全由端点维护。
MODEL_SOURCE_REPO_DOMAINS: tuple[str, ...] = (
    "wd14", "cltagger", "eval_clip", "eval_dino", "eval_ccip",
)


def is_abs_path(value: str) -> bool:
    """跨平台绝对路径判断（win 盘符 / UNC / posix 根）；repo id 形如
    `owner/name` 均为相对 → False。"""
    return (
        PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    )


class Secrets(BaseModel):
    gelbooru: GelbooruConfig = Field(default_factory=GelbooruConfig)
    danbooru: DanbooruConfig = Field(default_factory=DanbooruConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    reg: RegConfig = Field(default_factory=RegConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    modelscope: ModelScopeConfig = Field(default_factory=ModelScopeConfig)
    eval_metrics: EvalMetricModelsConfig = Field(
        default_factory=EvalMetricModelsConfig
    )
    # 旧的全局下载源（已退役为「迁移种子」）。不再有 UI 开关；新模型按类型在
    # download_sources 里各自选源。保留此字段仅为兼容旧 secrets.json：load 时把它
    # 的值种子填充到尚未设过的 download_sources 类型，避免老（尤其国内设了
    # modelscope 的）用户静默回退 HF。
    download_source: str = "huggingface"
    # 按类型分别选下载源：{"training"|"wd14"|"upscaler": "huggingface"|"modelscope"}。
    # 选中源缺某个 variant/文件时由 downloader 自动回退另一源。
    download_sources: dict[str, str] = Field(default_factory=dict)
    # JoyCaptionConfig 已并入 llm_tagger 的 joycaption builtin preset；
    # secrets.json 里若残留 joycaption 字段，由 _migrate_legacy_schema 迁移后丢弃。
    llm_tagger: LLMTaggerConfig = Field(default_factory=LLMTaggerConfig)
    wd14: WD14Config = Field(default_factory=WD14Config)
    cltagger: CLTaggerConfig = Field(default_factory=CLTaggerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    # 统一模型来源候选：domain → 用户添加的候选列表。domain 白名单校验在
    # API 层（families 注册表在 services 层）。内置 preset 不在此存储——
    # 候选全集 = 代码内置 + 本字段。当前选中值仍写各 domain 原字段
    # （wd14.model_id / eval_metrics.*_model_name / models.selected 等，两条
    # 兼容纪律见 docs/design/model-source-unification.md §3）。
    model_sources: dict[str, list[SourceCandidate]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _sync_legacy_source_fields(cls, data: Any) -> Any:
        """旧候选字段（wd14.model_ids / models.custom）→ model_sources 全量同步。

        入站 dict **带这些键**时（旧盘文件 / 老客户端 PUT）以其为准重建对应
        kind 的候选集——保留老客户端增删语义；`update()` 已把 merge base 里的
        重建键剥掉，故新 UI 只写 model_sources 时不会被过期旧值覆盖。
        另一半（model_sources → 旧字段重建写盘）见 after-validator。
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        raw_sources = data.get("model_sources")
        sources: dict[str, list[dict[str, Any]]] = {}
        if isinstance(raw_sources, dict):
            for domain, cands in raw_sources.items():
                if isinstance(cands, list):
                    sources[str(domain)] = [
                        dict(c) for c in cands if isinstance(c, dict)
                    ]

        def _sync(domain: str, kind: str, wanted: list[dict[str, Any]]) -> None:
            cur = sources.get(domain, [])
            kept = [c for c in cur if c.get("kind") != kind]
            sources[domain] = wanted + kept

        wd14_raw = data.get("wd14")
        if isinstance(wd14_raw, dict) and isinstance(wd14_raw.get("model_ids"), list):
            wanted = [
                {"kind": "download", "repo": str(m)}
                for m in wd14_raw["model_ids"]
                if str(m).strip() and str(m) not in DEFAULT_WD14_MODELS
            ]
            _sync("wd14", "download", wanted)

        models_raw = data.get("models")
        if isinstance(models_raw, dict):
            custom = models_raw.get("custom")
            if not isinstance(custom, dict) and isinstance(
                models_raw.get("custom_anima_paths"), list
            ):
                custom = {"anima": models_raw["custom_anima_paths"]}
            if isinstance(custom, dict):
                for family, paths in custom.items():
                    if not isinstance(paths, list):
                        continue
                    wanted = [
                        {"kind": "local", "path": str(p)}
                        for p in paths if str(p).strip()
                    ]
                    _sync(str(family), "local", wanted)

        data["model_sources"] = sources
        return data

    @model_validator(mode="after")
    def _model_sources_invariants(self) -> "Secrets":
        """统一不变量 + 兼容写盘面重建。

        1. repo 型 domain 的当前选中值若既非内置 preset 也不在候选里 → 自动
           补一条候选（WD14 现有「model_id 永远可见」不变量的推广；用户先切
           走再移除，前端强制该顺序）。
        2. 重建兼容面：wd14.model_ids = 内置 + download 候选（回滚可读）；
           models.custom = 各族 local 候选路径。二者在 `update()` 的 merge
           base 中被剥掉，唯一真源是 model_sources。
        """
        cltagger_official = str(
            CLTaggerConfig.model_fields["model_id"].default
        )
        selected_by_domain: dict[str, tuple[str, tuple[str, ...]]] = {
            "wd14": (self.wd14.model_id, DEFAULT_WD14_MODELS),
            "cltagger": (self.cltagger.model_id, (cltagger_official,)),
            "eval_clip": (
                self.eval_metrics.clip_model_name,
                (str(EvalMetricModelsConfig.model_fields["clip_model_name"].default),),
            ),
            "eval_dino": (
                self.eval_metrics.dino_model_name,
                (str(EvalMetricModelsConfig.model_fields["dino_model_name"].default),),
            ),
            "eval_ccip": (
                self.eval_metrics.ccip_model_name,
                (str(EvalMetricModelsConfig.model_fields["ccip_model_name"].default),),
            ),
        }
        for domain in MODEL_SOURCE_REPO_DOMAINS:
            sel, builtins = selected_by_domain[domain]
            sel = (sel or "").strip()
            if not sel or sel in builtins:
                continue
            cands = self.model_sources.setdefault(domain, [])
            if any(
                (c.kind == "download" and c.repo == sel)
                or (c.kind == "local" and c.path == sel)
                for c in cands
            ):
                continue
            if is_abs_path(sel):
                cands.append(SourceCandidate(kind="local", path=sel))
            elif domain == "cltagger":
                # fork repo 迁移自带当前双文件相对路径（镜像覆盖退役，D4）
                cands.append(SourceCandidate(
                    kind="download", repo=sel,
                    extra={
                        "model_path": self.cltagger.model_path,
                        "tag_mapping_path": self.cltagger.tag_mapping_path,
                    },
                ))
            else:
                cands.append(SourceCandidate(kind="download", repo=sel))

        # 兼容面重建（写盘给旧版本读；运行时读的选中值字段不在此列）
        wd14_downloads = [
            c.repo for c in self.model_sources.get("wd14", [])
            if c.kind == "download" and c.repo
        ]
        self.wd14.model_ids = list(DEFAULT_WD14_MODELS) + [
            m for m in wd14_downloads if m not in DEFAULT_WD14_MODELS
        ]
        custom: dict[str, list[str]] = {}
        for domain, cands in self.model_sources.items():
            if domain in MODEL_SOURCE_REPO_DOMAINS or domain == "upscaler":
                continue
            paths = [c.path for c in cands if c.kind == "local" and c.path]
            if paths or domain in self.models.custom:
                custom[domain] = paths
        self.models.custom = custom
        return self

    @model_validator(mode="after")
    def _seed_and_normalize_download_sources(self) -> "Secrets":
        """旧全局 download_source → 按类型 download_sources 的迁移种子 + 归一化。

        尚未设过的类型从旧全局值继承（老用户不丢源偏好）；非法值回落 huggingface。
        每次 load 都 setdefault（幂等）：用户一旦在某类型上显式选过就不会被覆盖。
        """
        legacy = str(self.download_source or "").strip().lower()
        if legacy not in DOWNLOAD_SOURCE_VALUES:
            legacy = "huggingface"
        for key in DOWNLOAD_SOURCE_TYPES:
            self.download_sources.setdefault(key, legacy)
        for key, val in list(self.download_sources.items()):
            if str(val).strip().lower() not in DOWNLOAD_SOURCE_VALUES:
                self.download_sources[key] = "huggingface"
            else:
                self.download_sources[key] = str(val).strip().lower()
        return self


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load() -> Secrets:
    """读 secrets.json；缺失或损坏时返回默认实例（不抛错）。"""
    if not SECRETS_FILE.exists():
        return Secrets()
    try:
        raw = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
        raw = _migrate_legacy_schema(raw) if isinstance(raw, dict) else raw
        return Secrets.model_validate(raw)
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
    - llm_tagger.presets 是 list[dict]，按 preset.id 匹配做按 id deep-merge，
      让前端 PUT 整个 list 时单个 preset 的 api_key=MASK 也能保持原值。
    - 未提及的字段沿用旧值。
    """
    current_dict = load().model_dump()
    # 剥离 models 的 read-compat computed 键（selected_anima / custom_anima_paths）：
    # 它们不是存储字段，留在 merge base 里会以「入站 legacy 键」的身份经
    # _migrate_legacy_model_fields 覆盖 partial 新写入的 selected/custom。
    # 真正入站的 legacy 键（老客户端）在 partial 里，照旧获胜。
    models_base = current_dict.get("models")
    if isinstance(models_base, dict):
        models_base.pop("selected_anima", None)
        models_base.pop("custom_anima_paths", None)
        # model_sources 兼容重建键（同上语义）：留在 merge base 会经
        # _sync_legacy_source_fields 用过期值覆盖 partial 新写入的 model_sources。
        models_base.pop("custom", None)
    wd14_base = current_dict.get("wd14")
    if isinstance(wd14_base, dict):
        wd14_base.pop("model_ids", None)
    merged = _deep_merge(current_dict, partial)
    new = Secrets.model_validate(merged)
    save(new)
    return new


def to_masked_dict(s: Secrets) -> dict[str, Any]:
    """GET /api/secrets 返回此结构；敏感字段非空时替换为 MASK。

    SENSITIVE_FIELDS 支持 `*` 通配（用于 llm_tagger.presets.*.api_key 这种
    list-of-dict 场景）。
    """
    d = s.model_dump()
    for path in SENSITIVE_FIELDS:
        _apply_mask(d, path.split("."))
    return d


def _apply_mask(node: Any, segs: list[str]) -> None:
    if not segs:
        return
    head, *rest = segs
    if head == "*":
        if isinstance(node, list):
            for item in node:
                _apply_mask(item, rest)
        return
    if not isinstance(node, dict):
        return
    if not rest:
        if node.get(head):
            node[head] = MASK
        return
    _apply_mask(node.get(head), rest)


# ---------------------------------------------------------------------------
# WandB preset 导入导出（0.18 预设化）
# ---------------------------------------------------------------------------


def get_wandb_preset(preset_id: str) -> Optional["WandBPresetConfig"]:
    """按 id 取 preset（**含真实 api_key**，绕过 mask）——只给显式导出端点用。"""
    for preset in load().wandb.presets:
        if preset.id == preset_id:
            return preset
    return None


def import_wandb_preset(
    data: Any, fallback_label: str = ""
) -> tuple[Secrets, "WandBPresetConfig"]:
    """导入一条 wandb preset：id 撞名自动加后缀，导入后设为当前选中。

    - 兼容旧前端 JSON 导出格式 ``{kind, version, preset: {...}}``（自动解包）
    - ``api_key == MASK`` 哨兵（旧客户端导出）按空处理；带真实 key 的备份文件
      原样恢复
    - 值非法时抛 pydantic ValidationError，由 caller 翻 400
    """
    if not isinstance(data, dict):
        raise ValueError("preset data must be a mapping")
    payload = dict(data)
    inner = payload.get("preset")
    if isinstance(inner, dict):
        payload = dict(inner)
    if str(payload.get("api_key") or "") == MASK:
        payload["api_key"] = ""

    label = str(payload.get("label") or fallback_label or "imported").strip() or "imported"
    slug = "".join(
        ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label
    ).strip("_") or "imported"

    s = load()
    used = {p.id for p in s.wandb.presets}
    pid, idx = slug, 1
    while pid in used:
        idx += 1
        pid = f"{slug}_{idx}"

    preset = WandBPresetConfig(**{**payload, "id": pid, "label": label})
    s.wandb.presets.append(preset)
    s.wandb.current_preset = preset.id
    new = Secrets.model_validate(s.model_dump())  # 重跑 validator（去重/回退保底）
    save(new)
    return new, preset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _migrate_legacy_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """老 schema → 新 schema 一次性迁移。

    迁移目标 (PR #18 schema → preset-unified schema)：
    1. 顶层 LLMTaggerConfig.base_url / api_key / model / endpoint / temperature /
       max_tokens / max_side / jpeg_quality / max_image_mb / timeout / max_retries
       下沉到每个 preset
    2. prompt_presets[{id,label,prompt,builtin,output_format}] 升级为完整 preset
       （继承顶层 endpoint + 生成参数字段）
    3. prompt_preset = "custom" + custom_prompt 非空 → 建一个 `user_custom` preset
    4. JoyCaptionConfig.base_url / model / prompt_template → 写入 joycaption preset
       （base_url/model 直接覆盖；prompt_template 非默认时建 `user_joycaption`）
    5. 删 raw["joycaption"] 字段
    6. system.show_dev_channel=true → system.update_channel="dev"（ADR 0005）

    幂等：新 schema（llm_tagger 含 current_preset / presets）直接返回。
    """
    # 6. system 通道偏好一次性迁移（无论后面 llm_tagger path 怎么走都先做）
    sys_raw = raw.get("system")
    if isinstance(sys_raw, dict):
        # 新字段已显式设过 → 不覆盖（幂等）
        if "update_channel" not in sys_raw and sys_raw.get("show_dev_channel") is True:
            sys_raw["update_channel"] = "dev"

    # 8. R-1 资源档位（0.17）：queue.allow_gpu_during_train 废弃。语义变化
    #    （老开关连 eval_samples 等底模级任务一起放行，是 OOM 隐患；新开关
    #    light_tasks_during_train 只辖轻量档且默认开），且 save() 全量落盘使
    #    「显式 false」与「默认 false」不可分辨 —— 故不迁移旧值，直接丢弃。
    q_raw = raw.get("queue")
    if isinstance(q_raw, dict):
        q_raw.pop("allow_gpu_during_train", None)

    # 9. WandB 预设化（0.18）：老扁平 wandb {enabled, api_key, project, ...} →
    #    {enabled, current_preset, presets: [{id: "default", ...}]}。enabled 留
    #    顶层（总开关不随预设切换），其余字段整体下沉成 id="default" 的 preset。
    #    幂等：已有 presets 键直接跳过。
    wb_raw = raw.get("wandb")
    if isinstance(wb_raw, dict) and "presets" not in wb_raw:
        enabled = bool(wb_raw.pop("enabled", False))
        preset = {**wb_raw, "id": "default", "label": "Default"}
        raw["wandb"] = {
            "enabled": enabled,
            "current_preset": "default",
            "presets": [preset],
        }

    # 7. gelbooru 的图片入库设置搬到全局 download.*（这三个本被所有 booru 下载 /
    #    reg / 本地上传共用，不该挂在 gelbooru 下）。download 侧未显式设过才搬，幂等。
    gel_raw = raw.get("gelbooru")
    if isinstance(gel_raw, dict):
        dl_raw = raw.setdefault("download", {})
        if isinstance(dl_raw, dict):
            for k in ("save_tags", "convert_to_png", "remove_alpha_channel"):
                if k in gel_raw and k not in dl_raw:
                    dl_raw[k] = gel_raw[k]
                gel_raw.pop(k, None)

    llm_old = raw.get("llm_tagger")
    if not isinstance(llm_old, dict):
        # 不存在 llm_tagger 字段：可能是更老的 secrets.json；交给 pydantic 用默认值
        raw.pop("joycaption", None)
        return raw

    # 已经是新 schema：仅清理可能残留的 joycaption 字段后直接返回
    if "presets" in llm_old or "current_preset" in llm_old:
        raw.pop("joycaption", None)
        return raw

    # 老顶层字段（PR #18 schema）
    def _get(key: str, default: Any) -> Any:
        val = llm_old.get(key)
        return default if val is None else val

    old_base_url = _get("base_url", "")
    old_api_key = _get("api_key", "")
    old_model = _get("model", "")
    old_model_ids = list(_get("model_ids", []) or [])
    old_endpoint = _get("endpoint", "chat_completions")
    old_temperature = _get("temperature", 0.2)
    old_max_tokens = _get("max_tokens", 700)
    old_timeout = _get("timeout", 60)
    old_max_retries = _get("max_retries", 3)
    old_concurrency = _get("concurrency", 1)
    old_requests_per_second = _get("requests_per_second", 0.0)
    old_max_requests_per_minute = _get("max_requests_per_minute", 0)
    old_max_side = _get("max_side", 1280)
    old_jpeg_quality = _get("jpeg_quality", 85)
    old_max_image_mb = _get("max_image_mb", 5.0)
    old_custom_prompt = str(_get("custom_prompt", "")).strip()
    old_prompt_preset = _get("prompt_preset", "style_json")
    old_prompt_presets = list(_get("prompt_presets", []) or [])

    from .llm_presets import builtin_llm_presets  # 局部 import 避免循环

    builtin_defaults = {item["id"]: item for item in builtin_llm_presets()}

    def _endpoint_fields() -> dict[str, Any]:
        return {
            "base_url": old_base_url,
            "api_key": old_api_key,
            "model": old_model,
            "model_ids": list(old_model_ids),
            "endpoint": old_endpoint,
            "temperature": old_temperature,
            "max_tokens": old_max_tokens,
            "max_side": old_max_side,
            "jpeg_quality": old_jpeg_quality,
            "max_image_mb": old_max_image_mb,
            "timeout": old_timeout,
            "max_retries": old_max_retries,
            "concurrency": old_concurrency,
            "requests_per_second": old_requests_per_second,
            "max_requests_per_minute": old_max_requests_per_minute,
        }

    new_presets: list[dict[str, Any]] = []
    for p in old_prompt_presets:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        if not pid:
            continue
        base_default = builtin_defaults.get(pid, {})
        merged = {
            **_endpoint_fields(),
            "id": pid,
            "label": p.get("label") or base_default.get("label") or pid,
            "builtin": pid in builtin_defaults,
            "prompt": p.get("prompt") or base_default.get("prompt", ""),
            "output_format": p.get("output_format") or base_default.get("output_format", "json"),
        }
        # joycaption builtin 用其自己的推荐 temperature/max_tokens（如果用户没改过老顶层）
        if pid in builtin_defaults and old_temperature == 0.2 and old_max_tokens == 700:
            merged["temperature"] = base_default.get("temperature", old_temperature)
            merged["max_tokens"] = base_default.get("max_tokens", old_max_tokens)
        new_presets.append(merged)

    current = str(old_prompt_preset or "").strip() or "style_json"
    if current == "custom" and old_custom_prompt:
        new_presets.append({
            **_endpoint_fields(),
            "id": "user_custom",
            "label": "自定义",
            "builtin": False,
            "prompt": old_custom_prompt,
            "output_format": "json",
        })
        current = "user_custom"

    # JoyCaption 卡片合并 ----
    joycap = raw.get("joycaption") if isinstance(raw.get("joycaption"), dict) else {}
    joy_base_url = str(joycap.get("base_url", "") or "").strip()
    joy_model = str(joycap.get("model", "") or "").strip()
    joy_prompt = str(joycap.get("prompt_template", "") or "").strip()

    joycap_default_base = "http://localhost:8000/v1"
    joycap_default_model = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    joycap_default_prompt = "Descriptive Caption"

    if joy_base_url or joy_model:
        # 写入 joycaption preset（如果 old prompt_presets 没含 joycaption，建一个）
        joy_preset = next((p for p in new_presets if p["id"] == "joycaption"), None)
        if joy_preset is None:
            joy_default = builtin_defaults.get("joycaption", {})
            joy_preset = {**_endpoint_fields(), **joy_default, "id": "joycaption", "builtin": True}
            new_presets.append(joy_preset)
        if joy_base_url and joy_base_url != joycap_default_base:
            joy_preset["base_url"] = joy_base_url
        if joy_model and joy_model != joycap_default_model:
            joy_preset["model"] = joy_model
            if joy_model not in joy_preset.get("model_ids", []):
                joy_preset["model_ids"] = [joy_model, *joy_preset.get("model_ids", [])]
    if joy_prompt and joy_prompt != joycap_default_prompt:
        # 用户改过 joycaption prompt_template → 建 user 自定义 preset，保留这份 prompt
        new_presets.append({
            "base_url": joy_base_url or joycap_default_base,
            "api_key": "",
            "model": joy_model or joycap_default_model,
            "model_ids": [joy_model] if joy_model else [],
            "endpoint": "chat_completions",
            "temperature": 0.6,
            "max_tokens": 300,
            "max_side": 1280,
            "jpeg_quality": 85,
            "max_image_mb": 5.0,
            "timeout": 60,
            "max_retries": 3,
            "concurrency": 1,
            "requests_per_second": 0.0,
            "max_requests_per_minute": 0,
            "id": "user_joycaption",
            "label": "JoyCaption（自定义 prompt）",
            "builtin": False,
            "prompt": joy_prompt,
            "output_format": "text",
        })

    raw["llm_tagger"] = {
        "current_preset": current,
        "presets": new_presets,
    }
    raw.pop("joycaption", None)
    return raw


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """把 patch 合并到 base：嵌套 dict 递归合并；leaf 值为 MASK 则丢弃。

    list[dict] 含 id 字段时（如 llm_tagger.presets）按 id deep-merge：保留 base
    里 patch 没动到的 preset；patch 中存在的 preset 与 base 同 id 项 deep-merge。
    """
    out = dict(base)
    for key, val in patch.items():
        if (
            isinstance(val, list)
            and isinstance(out.get(key), list)
            and val
            and all(isinstance(x, dict) and "id" in x for x in val)
            and all(isinstance(x, dict) and "id" in x for x in out[key])
        ):
            base_by_id = {x["id"]: x for x in out[key]}
            merged_list: list[Any] = []
            seen: set[str] = set()
            for px in val:
                bx = base_by_id.get(px["id"], {})
                merged_list.append(_deep_merge(bx, px))
                seen.add(px["id"])
            out[key] = merged_list
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        elif val == MASK:
            # 保持旧值
            continue
        else:
            out[key] = val
    return out


def has_danbooru_credentials() -> bool:
    """前端 / 端点判断是否已经配好 Danbooru auth。"""
    d = load().danbooru
    return bool(d.username and d.api_key)


def has_gelbooru_credentials() -> bool:
    """便捷：用于前端 / 端点判断是否已经配好 Gelbooru。"""
    g = load().gelbooru
    return bool(g.user_id and g.api_key)


def has_credentials_for(api_source: str) -> bool:
    """各下载渠道的「能不能跑」判定（两个 source 都强制绑定，no anon）：
    - gelbooru: 必须有 user_id + api_key（API 强制要求）
    - danbooru: 必须有 username + api_key（PR #38 起，CF 收紧后强制）
    """
    if api_source == "gelbooru":
        return has_gelbooru_credentials()
    if api_source == "danbooru":
        return has_danbooru_credentials()
    return False
