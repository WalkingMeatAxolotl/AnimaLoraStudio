"""族无关的变长文本条件栈（``TextSpec.strategy == "cached_varlen"`` 的族共用）。

自 ``families/krea2/text_encoding.py`` 的 ``Krea2TextStack`` 抽出（MOSS-Image 是
第二个消费者），行为不变。负责：

- TE 的惰性加载 / offload 到 CPU / 释放（训练「先缓存、释放 TE、再加载 DiT」）
- 训练文本缓存：图片 caption sidecar 与 sample prompt bundle 的读、补编、写
  （存储协议在 ``training.text_cache``）
- 在线模式（generate）的 prompt → context 内存 LRU

族子类只实现「怎么编码、产物长什么样、怎么拼 batch」：

- ``_encode_many(captions) -> list[Tensor]``：一组 caption → 各自的变长 context
- ``_validate_context(context, *, source) -> Tensor``：缓存产物形状 / dtype 校验
- ``_pad_conditions(contexts, *, device, dtype)``：右 pad 成 DiT 消费的 batch 条件
- 可选 ``_after_model_load(model)``：TE 刚加载完的族内处理（如 manual cast）
"""

from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from studio.infrastructure.log_messages import msg
from .text_cache import TextCacheEntry, TextCacheStore


logger = logging.getLogger(__name__)

_CACHE_TENSOR_KEY = "context"


class VarlenTextStack:
    """Lazy text conditioner with cached and storage-free online modes."""

    #: 报错文案里的族名（子类覆盖）
    _label = "Text"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        compute_dtype: torch.dtype | None,
        cache_enabled: bool,
        tokenizer,
        model_loader: Callable[[Path, torch.device, torch.dtype], Any],
        text_fingerprint: str,
        cache_batch_size: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        # None = compute 跟随存储 dtype（训练路径现状）；生成场景可传与存储不同的
        # compute dtype，由子类 _after_model_load 落实
        self.compute_dtype = compute_dtype
        self.cache_enabled = bool(cache_enabled)
        self.tokenizer = tokenizer
        self._model_loader = model_loader
        self._model = None
        self._offloaded = False
        self.store = TextCacheStore(text_fingerprint)
        self.cache_batch_size = int(cache_batch_size)
        self._caption_entries: dict[str, list[TextCacheEntry]] = {}
        self._prompt_captions: list[str] = []
        self._cache_root: Path | None = None
        # 在线模式（generate）的 prompt→context 内存 LRU（Comfy conditioning
        # 节点缓存同款语义）：命中时 TE 完全不动。存 CPU、保原 dtype——cast
        # 会破坏「首图与缓存命中图逐位一致」。cached 模式（训练）不经此路径。
        self._online_lru: OrderedDict[str, Tensor] = OrderedDict()
        self._online_lru_capacity = 16

        if self.cache_batch_size <= 0:
            raise ValueError(f"{self._label} cache_batch_size 必须为正数")

    # ── 子类钩子 ──────────────────────────────────────────────────────

    def _encode_many(self, captions: Sequence[str]) -> list[Tensor]:
        raise NotImplementedError

    def _validate_context(self, context: object, *, source: str) -> Tensor:
        raise NotImplementedError

    def _pad_conditions(self, contexts: Sequence[Tensor], *, device, dtype):
        raise NotImplementedError

    def _after_model_load(self, model) -> None:
        """TE 刚从磁盘加载完时调用（默认无操作）。"""

    # ── TE 生命周期 ───────────────────────────────────────────────────

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_model_on_device(self) -> bool:
        """TE 当前是否驻留目标设备（编排层判断是否需要腾显存搬它上来）。"""
        return self._model is not None and not self._offloaded

    def ensure_model(self):
        if self._model is None:
            self._model = self._model_loader(self.model_path, self.device, self.dtype)
            self._after_model_load(self._model)
            # TE 权重文件（数 GB 到十几 GB）的 mmap 缓存页归还系统（真机换页卡死案例）
            from training.sysmem import trim_working_set  # noqa: PLC0415

            trim_working_set()
        elif self._offloaded:
            # 上次采样前被 offload 到 CPU（见 offload_model）——搬回目标设备
            self._model.to(self.device)
            self._offloaded = False
        return self._model

    def offload_model(self) -> None:
        """把 TE 挪到 CPU 给 DiT 腾显存（Comfy parity：free_memory 的
        「编码后卸载 CLIP 到 offload_device」语义）。下个 prompt 由 ensure_model
        搬回（GPU↔CPU 秒级，远快于 release 后从盘重载）。"""
        if self._model is None or self._offloaded:
            return
        self._model.to("cpu")
        self._offloaded = True
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_model(self) -> None:
        """释放 cached 模式的 TE，再让 DiT 占用设备。"""

        if self._model is None:
            return
        self._model = None
        self._offloaded = False
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 在线 LRU（generate） ─────────────────────────────────────────

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

    def precache_online_prompts(self, captions: Sequence[str]) -> int:
        """任务级预编码：把这批 caption 编进在线 LRU（存 CPU）；返回新编码数。

        XY / 多 prompt generate 的 prompt 集合在任务开始前就封闭——先全部
        编码再 offload_model()，采样期 TE 归零显存占用（训练两段式加载的
        推理版）。cached 模式（训练）不适用，no-op。LRU 容量按本批需求
        抬升（上限 64），超出部分由逐格惰性路径兜底。
        """
        if self.cache_enabled:
            return 0
        unique = list(dict.fromkeys(str(caption) for caption in captions))
        self._online_lru_capacity = max(
            self._online_lru_capacity, min(len(unique), 64),
        )
        missing = [c for c in unique[:64] if c not in self._online_lru]
        step = max(1, self.cache_batch_size)
        for start in range(0, len(missing), step):
            chunk = missing[start:start + step]
            for caption, context in zip(chunk, self._encode_many(chunk)):
                self._online_lru_put(caption, context.detach().to("cpu"))
        return len(missing)

    # ── 训练文本缓存 ─────────────────────────────────────────────────

    def _encode_in_chunks(
        self, captions: Sequence[str], *, log_progress: bool = False,
    ) -> dict[str, Tensor]:
        # log_progress 只在预缓存阶段开（对齐 VAE 缓存的"编码进度"口径）；
        # 训练中 miss repair 走的也是本函数，那边一两条不刷屏。
        encoded: dict[str, Tensor] = {}
        unique = list(dict.fromkeys(str(caption) for caption in captions))
        # 进度步长自适应（Q3）：固定每 10 条在 1000 条数据集上是 100 行。
        stride = max(10, len(unique) // 10)
        next_mark = stride
        for start in range(0, len(unique), self.cache_batch_size):
            chunk = unique[start:start + self.cache_batch_size]
            contexts = self._encode_many(chunk)
            for caption, context in zip(chunk, contexts):
                encoded[caption] = context.detach().cpu().contiguous()
            if log_progress:
                done = len(encoded)
                if done >= next_mark or done == len(unique):
                    logger.info(msg(
                        "train.text_cache_progress", done=done, total=len(unique),
                    ))
                    while next_mark <= done:
                        next_mark += stride
        return encoded

    def _context_from_payload(self, payload: Mapping[str, object] | None) -> Tensor | None:
        if payload is None:
            return None
        context = payload.get(_CACHE_TENSOR_KEY)
        try:
            return self._validate_context(context, source=f"{self._label} 文本缓存")
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
                logger.info(msg(
                    "train.text_cache_hits",
                    hit=len(contexts), total=len(self._caption_entries),
                    todo=len(to_encode),
                ))
            else:
                logger.info(msg(
                    "train.text_cache_all_hit", n=len(self._caption_entries),
                ))
        contexts.update(self._encode_in_chunks(to_encode, log_progress=True))
        for caption, entries in missing_entries.items():
            for entry in entries:
                self.store.write_caption(entry, self._payload(contexts[caption]))

    def _prepare_prompt_bundle(self) -> None:
        if not self._prompt_captions:
            return
        if self._cache_root is None:
            raise ValueError(f"{self._label} prompt 缓存需要 cache_root")
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
            raise ValueError(f"{self._label} cached 模式编码未知 prompt 时需要 cache_root")
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
    ):
        """Return a padded condition; cached misses are repaired transparently."""

        caption_list = [str(caption) for caption in captions]
        if not caption_list:
            raise ValueError(f"{self._label} 文本 batch 不能为空")
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
            return self._pad_conditions(ordered, device=device, dtype=dtype)

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
        return self._pad_conditions(ordered, device=device, dtype=dtype)
