"""Krea2 Qwen3-VL text conditioning and variable-length cache lifecycle.

The prompt template, selected hidden layers, interior-padding gather, and output
layout are adapted from kohya-ss/musubi-tuner's Krea2 text encoder and cache
implementation at commit 8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6.
Copyright 2026 Kohya S. and musubi-tuner contributors. Apache-2.0.
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2/krea2_encoder.py
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2_cache_text_encoder_outputs.py

This repository loads the official sharded Hugging Face directory; the shared
``TextCacheStore`` sidecar protocol and the lazy load/release lifecycle that
keeps the 4B text encoder out of VRAM after pre-caching live in
``training.text_stack.VarlenTextStack``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from studio.infrastructure.log_messages import msg
from studio.services.inference.checkpoint_arch import ensure_supported_quantization_file
from ...text_stack import VarlenTextStack


logger = logging.getLogger(__name__)

KREA2_TEXT_FINGERPRINT = "qwen3-vl-4b-instruct-krea2-12x2560-v1"
#: 训练 caption 的 padding 定长下限——不是长度上限（两条路径都不截断）。
#: 保留这个定长是文本缓存兼容性依据，见 Krea2TextStack._pad_to_floor。
KREA2_MAX_LENGTH = 512
KREA2_SELECTED_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_TEXT_WIDTH = 2560

_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_PREFIX_TOKENS = 34
_SUFFIX_TOKENS = 5


@dataclass(frozen=True)
class Krea2TextCondition:
    """Padded Krea2 text condition consumed by the DiT."""

    context: Tensor
    attention_mask: Tensor


def gather_valid_text(hidden_states: Tensor, attention_mask: Tensor) -> list[Tensor]:
    """Gather valid tokens in order, including suffix tokens after interior padding."""

    if hidden_states.ndim < 3:
        raise ValueError("Krea2 hidden_states 必须至少为 (B, seq, features)")
    if attention_mask.ndim != 2:
        raise ValueError("Krea2 attention_mask 必须为 (B, seq)")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError(
            "Krea2 hidden_states 与 attention_mask 的 batch/seq 维不一致："
            f"{tuple(hidden_states.shape[:2])} != {tuple(attention_mask.shape)}"
        )

    mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
    gathered = [hidden_states[index][mask[index]] for index in range(mask.shape[0])]
    if any(item.shape[0] == 0 for item in gathered):
        raise ValueError("Krea2 文本条件不能没有有效 token")
    return gathered


def pad_text_conditions(
    contexts: Sequence[Tensor],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Krea2TextCondition:
    """Right-pad variable-length ``(seq, layers, width)`` tensors for one batch."""

    if not contexts:
        raise ValueError("Krea2 文本 batch 不能为空")
    shape = tuple(contexts[0].shape[1:])
    if contexts[0].ndim != 3 or any(
        item.ndim != 3 or tuple(item.shape[1:]) != shape or item.shape[0] == 0
        for item in contexts
    ):
        raise ValueError("Krea2 context 必须是非空且层数/宽度一致的 (seq, layers, width)")

    target = torch.device(device)
    max_length = max(item.shape[0] for item in contexts)
    padded = torch.zeros(
        len(contexts), max_length, *shape, device=target, dtype=dtype,
    )
    mask = torch.zeros(len(contexts), max_length, device=target, dtype=torch.bool)
    for index, item in enumerate(contexts):
        length = item.shape[0]
        padded[index, :length].copy_(item.to(device=target, dtype=dtype))
        mask[index, :length] = True
    return Krea2TextCondition(context=padded, attention_mask=mask)


def _cast_linear_forward(self, input: Tensor) -> Tensor:
    # ComfyUI manual_cast 语义（ops.py cast_bias_weight）：权重低精度常驻，
    # 前向 cast 到 input.dtype 计算。fp16→fp32 cast 精确，逐层 cast 与
    # 整模 upcast 数值逐位一致——显存差别（8.9GB vs 17.8GB）才是取舍点。
    weight = self.weight.to(input.dtype)
    bias = self.bias.to(input.dtype) if self.bias is not None else None
    return torch.nn.functional.linear(input, weight, bias)


def _cast_embedding_forward(self, input: Tensor) -> Tensor:
    # comfy ops Embedding：weight cast 到 compute dtype 再 lookup（row-select
    # 与 cast 可交换，数值等价）——由此激活流从源头进入 compute dtype 域。
    return torch.nn.functional.embedding(
        input,
        self.weight.to(self._krea2_compute_dtype),
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )


def patch_manual_cast(model: torch.nn.Module, compute_dtype: torch.dtype) -> int:
    """ComfyUI manual_cast 等价 patch（sd.py:258 ``set_model_compute_dtype``）。

    Embedding 输出进入 compute dtype 域后，全部 Linear 逐层把低精度权重
    cast 到 input.dtype（=compute dtype）计算；RMSNorm / rotary 无需 patch——
    transformers 实现里 fp32 激活流叠 torch type promotion 与 Comfy 的
    weight-cast 语义数值一致。返回 patch 的模块数。
    """
    from training.families.krea2.quant_fp8 import _FP8_TORCH_DTYPES  # noqa: PLC0415

    patched = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.dtype in _FP8_TORCH_DTYPES:
                # fp8_scaled 层已挂 dequant 前向（cast 到 input.dtype + 乘
                # scale = manual_cast 的 fp8 版）；覆盖会丢 scale
                continue
            module.forward = MethodType(_cast_linear_forward, module)
            patched += 1
        elif isinstance(module, torch.nn.Embedding):
            module._krea2_compute_dtype = compute_dtype
            module.forward = MethodType(_cast_embedding_forward, module)
            patched += 1
    if patched:
        logger.debug(
            "krea2_te: manual cast modules=%d compute_dtype=%s "
            "(weights kept in storage dtype)",
            patched, compute_dtype,
        )
    return patched


#: comfy 单文件 TE 的 text 侧键前缀 → HF Qwen3VLForConditionalGeneration 键。
#: comfy 打包（Comfy-Org qwen3vl_4b_fp8_scaled）把 language_model 直挂
#: ``model.``；visual 侧（model.visual.*）两边一致零映射。
_COMFY_TE_TEXT_PREFIXES = ("model.layers.", "model.embed_tokens.", "model.norm.")


def _comfy_te_single_file(model_path: Path) -> Path | None:
    """目录是 comfy 单文件 TE 布局时返回权重文件；HF 分片布局返回 None。"""
    if (model_path / "model.safetensors.index.json").exists():
        return None
    candidates = sorted(model_path.glob("*.safetensors"))
    return candidates[0] if len(candidates) == 1 else None


def _load_comfy_single_file_te(
    model_path: Path,
    weights_file: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    """加载 comfy 单文件布局的 Qwen3-VL（官方 fp8_scaled 形态）。

    config/tokenizer 小文件仍来自目录（下载中心 fp8 条目一并下载）；权重键
    做一条前缀映射（text 侧补 ``language_model.``），``comfy_quant`` 配置
    blob 丢弃，``weight_scale`` F32 标量收集后经 patch_fp8_linears 挂
    dequant 前向（与 DiT fp8_scaled 完全同款）。fp8 权重原样常驻；非量化
    键（embed/norm/visual）cast 到存储 dtype。lm_head 与 embed tied——
    文件不含该键，load 后 tie_weights() 重绑。
    """
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    from training.families.krea2.quant_fp8 import (  # noqa: PLC0415
        _FP8_TORCH_DTYPES,
        patch_fp8_linears,
    )

    # int8 版 TE（如 Comfy-Org 的 *_int8_convrot）：不拦会被 cast 成 bf16 后
    # 由 patch_fp8_linears 报「权重不是 fp8」
    ensure_supported_quantization_file(weights_file)
    logger.info(msg("train.loading_text_encoder_file", path=weights_file))
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "Qwen3-VL 单文件加载需要 accelerate（transformers 伴生依赖）"
        ) from exc

    # init_empty_weights 默认只把 parameters 放 meta；buffers（rotary
    # inv_freq 等 non-persistent，不在权重文件里）在 CPU 真实构造带正确
    # 值——纯 torch.device("meta") 上下文会把 buffer 也 meta 化，load 后
    # 无法恢复（曾致 offload 的 .to("cpu") 撞 meta tensor 崩溃）。
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    state_dict: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    with safe_open(str(weights_file), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.endswith(".comfy_quant"):
                continue
            mapped = key
            for prefix in _COMFY_TE_TEXT_PREFIXES:
                if key.startswith(prefix):
                    mapped = "model.language_model." + key[len("model."):]
                    break
            tensor = handle.get_tensor(key)
            if mapped.endswith(".weight_scale"):
                layer = mapped[: -len(".weight_scale")]
                scales[layer] = tensor.to(device=device)
                continue
            if tensor.dtype in _FP8_TORCH_DTYPES:
                state_dict[mapped] = tensor.to(device=device)
            else:
                state_dict[mapped] = tensor.to(device=device, dtype=dtype)

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    unexpected = list(result.unexpected_keys)
    missing = [k for k in result.missing_keys if k != "lm_head.weight"]
    if missing or unexpected:
        raise ValueError(
            f"Qwen3-VL comfy 单文件键不匹配：缺少 {missing[:5]}，"
            f"多出 {unexpected[:5]}"
        )
    # lm_head 与 embed tied（tie_word_embeddings）——文件不含该键；
    # transformers 的 tie_weights() 对 meta 构造 + assign 加载不重绑，
    # 手动指回 embed（零拷贝）。
    out_emb = model.get_output_embeddings()
    if out_emb is not None and out_emb.weight.device.type == "meta":
        out_emb.weight = model.get_input_embeddings().weight
    # 检查覆盖 parameters + buffers（漏 buffer 曾放过 rotary inv_freq 的
    # meta 残留：编码侥幸能跑，offload 全模型 .to("cpu") 遍历到即崩）
    leftover_meta = [
        name for name, tensor in [
            *model.named_parameters(), *model.named_buffers(),
        ]
        if tensor.device.type == "meta"
    ]
    if leftover_meta:
        raise ValueError(
            f"Qwen3-VL 单文件加载后仍有未物化参数：{leftover_meta[:5]}"
        )
    # buffers 构造在 CPU（init_empty_weights 只 meta 化参数）——搬到目标
    # device；参数已 assign 就位，.to() 对其 no-op
    model.to(device)
    if scales:
        patch_fp8_linears(model, scales)
        logger.debug(
            "fp8: dequant forward attached module=text_encoder layers=%d", len(scales),
        )
    return model.eval().requires_grad_(False)


def _default_model_loader(
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError(
            "当前 transformers 不含 Qwen3VLForConditionalGeneration；请安装支持 "
            "Qwen3-VL 的版本"
        ) from exc

    if not model_path.is_dir():
        raise ValueError(f"Krea2 文本编码器必须是 Hugging Face 模型目录：{model_path}")
    single = _comfy_te_single_file(model_path)
    if single is not None:
        return _load_comfy_single_file_te(model_path, single, device, dtype)
    logger.info(msg("train.loading_text_encoder_file", path=model_path))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
    )
    return model.eval().requires_grad_(False)


def _load_tokenizer(model_path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("Krea2 文本编码需要 transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    prefix_tokens = tokenizer(_PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
    suffix_tokens = tokenizer(_PROMPT_SUFFIX, add_special_tokens=False)["input_ids"]
    if len(prefix_tokens) != _PREFIX_TOKENS or len(suffix_tokens) != _SUFFIX_TOKENS:
        raise ValueError(
            "Krea2 tokenizer 与 Qwen3-VL-4B-Instruct prompt 模板不兼容："
            f"prefix={len(prefix_tokens)}, suffix={len(suffix_tokens)}"
        )
    return tokenizer


class Krea2TextStack(VarlenTextStack):
    """Lazy Qwen3-VL conditioner with cached and storage-free online modes.

    TE 生命周期、训练文本缓存与在线 LRU 在 ``training.text_stack.VarlenTextStack``；
    这里只有 Krea2 的 prompt 模板、12 层中间态堆叠与 fp8 单文件 TE 处理。
    """

    _label = "Krea2"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype | None = None,
        cache_enabled: bool = True,
        tokenizer=None,
        model_loader: Callable[[Path, torch.device, torch.dtype], Any] | None = None,
        text_fingerprint: str = KREA2_TEXT_FINGERPRINT,
        max_length: int = KREA2_MAX_LENGTH,
        selected_layers: Sequence[int] = KREA2_SELECTED_LAYERS,
        hidden_width: int = KREA2_TEXT_WIDTH,
        cache_batch_size: int = 1,
    ) -> None:
        model_path = Path(model_path)
        # 目录是官方 fp8_scaled 单文件形态（comfy 布局）——影响编排层的
        # TE 上卡显存判据（权重 ~5GB vs fp16 8.9GB）与训练文本缓存指纹
        self.is_fp8_storage = _comfy_te_single_file(model_path) is not None
        # fp8 TE 编码的嵌入与 bf16 有量化级差异——指纹区分防缓存混源
        # （换 TE 精度 → 指纹变 → sidecar 全量重编，一次性）
        if self.is_fp8_storage:
            text_fingerprint = f"{text_fingerprint}-tefp8"
        super().__init__(
            model_path,
            device=device,
            dtype=dtype,
            compute_dtype=compute_dtype,
            cache_enabled=cache_enabled,
            tokenizer=tokenizer if tokenizer is not None else _load_tokenizer(model_path),
            model_loader=model_loader or _default_model_loader,
            text_fingerprint=text_fingerprint,
            cache_batch_size=cache_batch_size,
        )
        self.max_length = int(max_length)
        self.selected_layers = tuple(int(index) for index in selected_layers)
        self.hidden_width = int(hidden_width)

        if self.max_length <= 0 or not self.selected_layers or self.hidden_width <= 0:
            raise ValueError("Krea2 文本编码配置必须为正数且 selected_layers 不能为空")

    def _after_model_load(self, model) -> None:
        # 生成场景「fp16 存储 + fp32 compute」（ComfyUI sd.py:258 口径）
        if self.compute_dtype is not None and self.compute_dtype != self.dtype:
            patch_manual_cast(model, self.compute_dtype)

    def _pad_conditions(self, contexts, *, device, dtype) -> Krea2TextCondition:
        return pad_text_conditions(contexts, device=device, dtype=dtype)

    def _pad_to_floor(self, encoded: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """把 caption 段右 pad 到 max_length 定长下限（不足才补，超出原样）。

        等价于旧的 ``padding="max_length"``——定长以内逐位一致，是文本缓存
        跨版本兼容的依据。批内出现超长样本时 ``padding="longest"`` 已把同批
        短样本 pad 到那个更长的宽度，短样本的 suffix 位置随之后移：同一
        caption 换一种分批组合会编出略有差异的嵌入。缓存只写一次，训练中不
        会自相矛盾；cache_batch_size=1（现状）下该分支根本不触发。
        """
        input_ids = encoded["input_ids"]
        floor = self.max_length + _PREFIX_TOKENS - _SUFFIX_TOKENS
        deficit = floor - input_ids.shape[1]
        if deficit <= 0:
            return {"input_ids": input_ids, "attention_mask": encoded["attention_mask"]}
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        return {
            "input_ids": torch.nn.functional.pad(
                input_ids, (0, deficit), value=int(pad_id) if pad_id is not None else 0,
            ),
            "attention_mask": torch.nn.functional.pad(
                encoded["attention_mask"], (0, deficit), value=0,
            ),
        }

    def _tokenize(self, captions: Sequence[str]) -> tuple[Tensor, Tensor]:
        text = [_PROMPT_PREFIX + str(caption) for caption in captions]
        suffix = [_PROMPT_SUFFIX] * len(text)
        if not self.cache_enabled:
            # 在线模式（generate）不截断：超过训练口径 512 token 的 prompt
            # 完整进模型——质量后果由用户掌握（不拦截、不静默丢尾巴）。
            # varlen 链路（gather_valid_text / DiT text_len）对任意长度无
            # 结构约束；pad 到 batch 内最长即可。
            encoded = self.tokenizer(
                text,
                truncation=False,
                return_length=False,
                return_overflowing_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
        else:
            # 训练 cached 模式同样不截断（与在线模式口径一致）：DiT 侧文本
            # 位置编码恒为零（krea2_modeling text_pos），文本长度对 DiT 无
            # 结构约束；TE 是 Qwen3-VL，512 远在其上下文之内。代价是显存与
            # 缓存体积随长度线性涨（一条 512 token ≈31MB），由用户掌握。
            #
            # 短 caption 仍 pad 到 max_length 定长是缓存兼容性要求：这是
            # interior padding（pad 在 caption 与 suffix 之间），suffix 的
            # RoPE 绝对位置因此恒定在定长之后。改成「只 pad 到批内最长」会
            # 让 ≤max_length 的 caption 嵌入数值改变，已有 sidecar 全量失效。
            # 保留定长下限 → 定长以内逐位不变 → 指纹无需变更。
            encoded = self.tokenizer(
                text,
                truncation=False,
                return_length=False,
                return_overflowing_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
            encoded = self._pad_to_floor(encoded)
        suffix_encoded = self.tokenizer(suffix, return_tensors="pt")
        input_ids = torch.cat(
            [encoded["input_ids"], suffix_encoded["input_ids"]], dim=1,
        ).to(self.device, non_blocking=True)
        mask = torch.cat(
            [encoded["attention_mask"].bool(), suffix_encoded["attention_mask"].bool()],
            dim=1,
        ).to(self.device, non_blocking=True)
        return input_ids, mask

    def _encode_many(self, captions: Sequence[str]) -> list[Tensor]:
        if not captions:
            return []
        model = self.ensure_model()
        input_ids, mask = self._tokenize(captions)
        backbone = getattr(model, "model", model)
        with torch.inference_mode():
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None or max(self.selected_layers) >= len(hidden_states):
            count = 0 if hidden_states is None else len(hidden_states)
            raise RuntimeError(
                f"Qwen3-VL 返回 hidden_states 层数不足：{count}，"
                f"需要索引 {max(self.selected_layers)}"
            )
        stacked = torch.stack(
            [hidden_states[index] for index in self.selected_layers], dim=2,
        )[:, _PREFIX_TOKENS:]
        cropped_mask = mask[:, _PREFIX_TOKENS:]
        contexts = gather_valid_text(stacked, cropped_mask)
        for context in contexts:
            self._validate_context(context, source="Qwen3-VL 输出")
        return contexts

    def _validate_context(self, context: object, *, source: str) -> Tensor:
        if not isinstance(context, Tensor):
            raise ValueError(f"{source} 缺少 tensor context")
        expected = (len(self.selected_layers), self.hidden_width)
        if (
            context.ndim != 3
            or context.shape[0] == 0
            or tuple(context.shape[1:]) != expected
            or not context.is_floating_point()
        ):
            raise ValueError(
                f"{source} context 必须为非空浮点 (seq, {expected[0]}, {expected[1]})，"
                f"实际 {tuple(context.shape)} / {context.dtype}"
            )
        return context


def load_krea2_text_stack(
    model_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    compute_dtype: torch.dtype | None = None,
    cache_enabled: bool = True,
) -> Krea2TextStack:
    """Create a lazy Krea2 text stack; only the small tokenizer loads immediately."""

    return Krea2TextStack(
        model_path,
        device=device,
        dtype=dtype,
        compute_dtype=compute_dtype,
        cache_enabled=cache_enabled,
    )
