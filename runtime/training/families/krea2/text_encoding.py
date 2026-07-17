"""Krea2 Qwen3-VL text conditioning and variable-length cache lifecycle.

The prompt template, selected hidden layers, interior-padding gather, and output
layout are adapted from kohya-ss/musubi-tuner's Krea2 text encoder and cache
implementation at commit 8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6.
Copyright 2026 Kohya S. and musubi-tuner contributors. Apache-2.0.
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2/krea2_encoder.py
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2_cache_text_encoder_outputs.py

This repository loads the official sharded Hugging Face directory, uses the
shared ``TextCacheStore`` sidecar protocol, and owns the lazy load/release
lifecycle needed to keep the 4B text encoder out of VRAM after pre-caching.
"""

from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from ...text_cache import TextCacheEntry, TextCacheStore


logger = logging.getLogger(__name__)

KREA2_TEXT_FINGERPRINT = "qwen3-vl-4b-instruct-krea2-12x2560-v1"
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
_CACHE_TENSOR_KEY = "context"


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
    patched = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = MethodType(_cast_linear_forward, module)
            patched += 1
        elif isinstance(module, torch.nn.Embedding):
            module._krea2_compute_dtype = compute_dtype
            module.forward = MethodType(_cast_embedding_forward, module)
            patched += 1
    if patched:
        logger.info(
            "Krea2 TE manual_cast：%d 个模块以 %s 计算（权重常驻存储 dtype）",
            patched, compute_dtype,
        )
    return patched


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
    logger.info("加载 Krea2 Qwen3-VL 文本编码器：%s", model_path)
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


class Krea2TextStack:
    """Lazy Qwen3-VL conditioner with cached and storage-free online modes."""

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
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        # None = compute 跟随存储 dtype（训练路径现状）；生成场景传 fp32 +
        # dtype=fp16 复刻 ComfyUI「fp16 存储 + fp32 compute」口径（sd.py:258）
        self.compute_dtype = compute_dtype
        self.cache_enabled = bool(cache_enabled)
        self.tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(self.model_path)
        self._model_loader = model_loader or _default_model_loader
        self._model = None
        self._offloaded = False
        self.store = TextCacheStore(text_fingerprint)
        self.max_length = int(max_length)
        self.selected_layers = tuple(int(index) for index in selected_layers)
        self.hidden_width = int(hidden_width)
        self.cache_batch_size = int(cache_batch_size)
        self._caption_entries: dict[str, list[TextCacheEntry]] = {}
        self._prompt_captions: list[str] = []
        self._cache_root: Path | None = None
        # 在线模式（generate）的 prompt→context 内存 LRU（Comfy conditioning
        # 节点缓存同款语义）：命中时 TE 完全不动。存 CPU、保原 dtype（fp32
        # 一条 ≈63MB，容量 16 ≈1GB RAM）——cast 会破坏「首图与缓存命中图
        # 逐位一致」。cached 模式（训练）不经此路径。
        self._online_lru: OrderedDict[str, Tensor] = OrderedDict()
        self._online_lru_capacity = 16

        if self.max_length <= 0 or not self.selected_layers or self.hidden_width <= 0:
            raise ValueError("Krea2 文本编码配置必须为正数且 selected_layers 不能为空")
        if self.cache_batch_size <= 0:
            raise ValueError("Krea2 cache_batch_size 必须为正数")

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_model_on_device(self) -> bool:
        """TE 当前是否驻留目标设备（编排层判断是否需要腾显存搬它上来）。"""
        return self._model is not None and not self._offloaded

    def _online_lru_get(self, caption: str) -> Tensor | None:
        context = self._online_lru.get(caption)
        if context is not None:
            self._online_lru.move_to_end(caption)
        return context

    def _online_lru_put(self, caption: str, context: Tensor) -> None:
        self._online_lru[caption] = context
        self._online_lru.move_to_end(caption)
        while len(self._online_lru) > self._online_lru_capacity:
            self._online_lru.popitem(last=False)

    def online_conditions_cached(self, captions: Sequence[str]) -> bool:
        """这批 caption 是否全部命中在线 LRU（编排层 peek：全命中 → 编码
        阶段 TE 不需要上 GPU，按需让位判断可整体跳过）。"""
        return (
            not self.cache_enabled
            and all(str(caption) in self._online_lru for caption in captions)
        )

    def ensure_model(self):
        if self._model is None:
            self._model = self._model_loader(self.model_path, self.device, self.dtype)
            if self.compute_dtype is not None and self.compute_dtype != self.dtype:
                patch_manual_cast(self._model, self.compute_dtype)
        elif self._offloaded:
            # 上次采样前被 offload 到 CPU（见 offload_model）——搬回目标设备
            self._model.to(self.device)
            self._offloaded = False
        return self._model

    def offload_model(self) -> None:
        """把 TE 挪到 CPU 给 DiT 腾显存（Comfy parity：free_memory 的
        「编码后卸载 CLIP 到 offload_device」语义）。

        Generate 场景 DiT(26.3GB bf16) + Qwen3-VL(8.9GB) 同驻 ≈ 35GB，超出
        32GB 支持下限——采样前必须让 DiT 独占。下个 prompt 由 ensure_model
        搬回（GPU↔CPU 秒级，远快于 release 后从盘重载）。
        """
        if self._model is None or self._offloaded:
            return
        self._model.to("cpu")
        self._offloaded = True
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_model(self) -> None:
        """Release the cached-mode TE before the 12.9B DiT occupies the device."""

        if self._model is None:
            return
        self._model = None
        self._offloaded = False
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _tokenize(self, captions: Sequence[str]) -> tuple[Tensor, Tensor]:
        text = [_PROMPT_PREFIX + str(caption) for caption in captions]
        suffix = [_PROMPT_SUFFIX] * len(text)
        encoded = self.tokenizer(
            text,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=self.max_length + _PREFIX_TOKENS - _SUFFIX_TOKENS,
            return_tensors="pt",
        )
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

    def _encode_in_chunks(
        self, captions: Sequence[str], *, log_progress: bool = False,
    ) -> dict[str, Tensor]:
        # log_progress 只在预缓存阶段开（对齐 VAE 缓存的"编码进度"口径）；
        # 训练中 miss repair 走的也是本函数，那边一两条不刷屏。
        encoded: dict[str, Tensor] = {}
        unique = list(dict.fromkeys(str(caption) for caption in captions))
        next_mark = 10
        for start in range(0, len(unique), self.cache_batch_size):
            chunk = unique[start:start + self.cache_batch_size]
            contexts = self._encode_many(chunk)
            for caption, context in zip(chunk, contexts):
                encoded[caption] = context.detach().cpu().contiguous()
            if log_progress:
                done = len(encoded)
                if done >= next_mark or done == len(unique):
                    logger.info("  文本编码进度: %d/%d", done, len(unique))
                    while next_mark <= done:
                        next_mark += 10
        return encoded

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

    def _context_from_payload(self, payload: Mapping[str, object] | None) -> Tensor | None:
        if payload is None:
            return None
        context = payload.get(_CACHE_TENSOR_KEY)
        try:
            return self._validate_context(context, source="Krea2 文本缓存")
        except ValueError:
            return None

    @staticmethod
    def _payload(context: Tensor) -> dict[str, Tensor]:
        return {_CACHE_TENSOR_KEY: context}

    def _prepare_caption_sidecars(self) -> None:
        contexts: dict[str, Tensor] = {}
        missing_entries: dict[str, list[TextCacheEntry]] = {}
        for caption, entries in self._caption_entries.items():
            context = None
            for entry in entries:
                cached = self._context_from_payload(self.store.read_caption(entry))
                if cached is None:
                    missing_entries.setdefault(caption, []).append(entry)
                elif context is None:
                    context = cached
            if context is not None:
                contexts[caption] = context

        to_encode = [
            caption for caption in self._caption_entries if caption not in contexts
        ]
        if self._caption_entries:
            if to_encode:
                logger.info(
                    "[text-cache] caption sidecar 命中 %d/%d，需编码 %d 条...",
                    len(contexts), len(self._caption_entries), len(to_encode),
                )
            else:
                logger.info(
                    "[text-cache] caption sidecar 全部命中（%d 条），跳过编码",
                    len(self._caption_entries),
                )
        contexts.update(self._encode_in_chunks(to_encode, log_progress=True))
        for caption, entries in missing_entries.items():
            for entry in entries:
                self.store.write_caption(entry, self._payload(contexts[caption]))

    def _prepare_prompt_bundle(self) -> None:
        if not self._prompt_captions:
            return
        if self._cache_root is None:
            raise ValueError("Krea2 prompt 缓存需要 cache_root")
        payloads: dict[str, dict[str, Tensor]] = {}
        missing = []
        for caption in self._prompt_captions:
            context = self._context_from_payload(
                self.store.read_prompt(self._cache_root, caption),
            )
            if context is None:
                missing.append(caption)
            else:
                payloads[caption] = self._payload(context)
        for caption, context in self._encode_in_chunks(missing).items():
            payloads[caption] = self._payload(context)
        if missing:
            self.store.write_prompt_bundle(self._cache_root, payloads)

    def prepare_text_cache(
        self,
        captions: Iterable[str],
        extra_prompts: Iterable[str],
        *,
        cache_entries: Iterable[TextCacheEntry] = (),
        cache_root: str | Path | None = None,
    ) -> None:
        """Populate all known text caches, or retain the TE for online mode."""

        entries = list(cache_entries)
        self._caption_entries = {}
        for entry in entries:
            self._caption_entries.setdefault(entry.caption, []).append(entry)
        caption_list = list(dict.fromkeys(str(caption) for caption in captions))
        prompt_list = [str(prompt) for prompt in extra_prompts]
        prompt_list.extend(
            caption for caption in caption_list if caption not in self._caption_entries
        )
        self._prompt_captions = list(dict.fromkeys(prompt_list))
        self._cache_root = Path(cache_root) if cache_root is not None else None

        if not self.cache_enabled:
            self.ensure_model()
            return
        try:
            self._prepare_caption_sidecars()
            self._prepare_prompt_bundle()
        finally:
            self.release_model()

    def _repair_prompt_bundle(self, required: Sequence[str]) -> dict[str, Tensor]:
        if self._cache_root is None:
            raise ValueError("Krea2 cached 模式编码未知 prompt 时需要 cache_root")
        self._prompt_captions = list(dict.fromkeys([*self._prompt_captions, *required]))
        contexts: dict[str, Tensor] = {}
        missing = []
        for caption in self._prompt_captions:
            context = self._context_from_payload(
                self.store.read_prompt(self._cache_root, caption),
            )
            if context is None:
                missing.append(caption)
            else:
                contexts[caption] = context
        contexts.update(self._encode_in_chunks(missing))
        if missing:
            self.store.write_prompt_bundle(
                self._cache_root,
                {caption: self._payload(context) for caption, context in contexts.items()},
            )
        return contexts

    def encode_text_for_batch(
        self,
        captions: Sequence[str],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Krea2TextCondition:
        """Return a padded condition; cached misses are repaired transparently."""

        caption_list = [str(caption) for caption in captions]
        if not caption_list:
            raise ValueError("Krea2 文本 batch 不能为空")
        if not self.cache_enabled:
            unique = list(dict.fromkeys(caption_list))
            contexts: dict[str, Tensor] = {}
            missing: list[str] = []
            for caption in unique:
                cached = self._online_lru_get(caption)
                if cached is None:
                    missing.append(caption)
                else:
                    contexts[caption] = cached
            for caption, context in zip(missing, self._encode_many(missing)):
                stored = context.detach().to("cpu")
                contexts[caption] = stored
                self._online_lru_put(caption, stored)
            ordered = [contexts[caption] for caption in caption_list]
            return pad_text_conditions(ordered, device=device, dtype=dtype)

        unique = list(dict.fromkeys(caption_list))
        contexts: dict[str, Tensor] = {}
        missing_sidecars: list[str] = []
        missing_prompts: list[str] = []
        for caption in unique:
            entries = self._caption_entries.get(caption)
            if entries:
                context = None
                for entry in entries:
                    context = self._context_from_payload(self.store.read_caption(entry))
                    if context is not None:
                        break
                if context is None:
                    missing_sidecars.append(caption)
                else:
                    contexts[caption] = context
            else:
                missing_prompts.append(caption)

        try:
            repaired = self._encode_in_chunks(missing_sidecars)
            for caption, context in repaired.items():
                contexts[caption] = context
                for entry in self._caption_entries[caption]:
                    self.store.write_caption(entry, self._payload(context))
            if missing_prompts:
                contexts.update(self._repair_prompt_bundle(missing_prompts))
        finally:
            self.release_model()

        ordered = [contexts[caption] for caption in caption_list]
        return pad_text_conditions(ordered, device=device, dtype=dtype)


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
