"""全局服务凭证 + 配置 —— 集中存到 studio_data/secrets.json。

`studio_data/` 已被 .gitignore，本文件即可放真实 token / api key。
对外通过 `to_masked_dict()` 把敏感字段以 "***" 返回；前端 PUT
时若回传 "***" 表示「保持不变」，由 `update()` 的 deep-merge 处理。
"""
from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .paths import STUDIO_DATA

SECRETS_FILE = STUDIO_DATA / "secrets.json"
MASK = "***"
# 点路径 + `*` 通配支持：`llm_tagger.presets.*.api_key` 会遍历 list 内每个 dict。
SENSITIVE_FIELDS: tuple[str, ...] = (
    "gelbooru.api_key",
    "danbooru.api_key",
    "huggingface.token",
    "wandb.api_key",
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


class WandBConfig(BaseModel):
    enabled: bool = False
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
    def _normalize_values(self) -> "WandBConfig":
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
    eval_baseline_enabled: bool = True
    # 评估时机（全局）。是否评估由每个 version 训练配置的 eval_validation_enabled
    # 决定；启用后，这里选何时评：训练全部结束后批量评 / 训练中每存一个 LoRA 就评。
    auto_eval_trigger: Literal["after_training", "checkpoint"] = "after_training"


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
    # CLTagger 模型输出 7 个 category：General / Character 走阈值过滤，其余 5 个
    # 按 bool 开关 gate。默认勾上 General / Character / Copyright 三类——LoRA
    # 训练标准 caption 形态；Meta / Model / Rating / Quality 默认关，避免污染
    # caption（例如 "highres", "best quality", "explicit" 这类元信息）。
    add_copyright_tag: bool = True
    add_meta_tag: bool = False
    add_model_tag: bool = False
    add_rating_tag: bool = False
    add_quality_tag: bool = False
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
    - `selected_anima`：当前默认主模型。可为官方 variant key（`1.0` 等）**或**
      `custom_anima_paths` 里某个本地 `.safetensors` 绝对路径。Studio 创建新
      version 时根据此字段把 `transformer_path` 写成绝对路径到 yaml；已存在
      version 不动（保证训练重现性）。
    - `custom_anima_paths`：用户通过设置页 PathPicker 注册的本地主模型权重
      （`.safetensors` 绝对路径列表）。用来微调训练 / 在微调权重上测试出图。
      仅注册路径，不下载、不复制；条目失效（文件被删/移走）时解析自动回退到
      官方 variant。
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
    selected_anima: str = "1.0"
    custom_anima_paths: list[str] = Field(default_factory=list)
    selected_upscaler: str = "4x-AnimeSharp"
    auto_sync_paths: bool = True


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
    - `save_test_images`：开关测试出图自动落盘。默认关；开后每次出完图前端
      会调 /api/generate/save 把成图存到 studio_data/test/<date>/{single,xy}/
      image_N.png（N 按当前文件夹已有最大编号+1）。compare 模式不落盘。
    """
    preview_every_n_steps: int = 3
    attention_backend: str = "auto"
    vae_precision: str = "bf16"
    idle_timeout_minutes: int = 10
    save_test_images: bool = False


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
