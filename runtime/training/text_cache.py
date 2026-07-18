"""模型族文本条件缓存的共享存储基建（多模型 Phase 2）。

本模块只负责缓存协议与磁盘 I/O，不知道 tokenizer / text encoder，也不 import
任何 family：编码行为仍由 ``ModelFamily.prepare_text_cache`` 自治。

协议（docs/design/multi-model/04-synthesis.md D19）：

- 图片 caption 缓存与图片同目录，使用 ``<完整图片名>.text.safetensors`` sidecar；
- key 由最终 caption、TE 指纹和格式版本共同决定；
- tensor 保留可变 token 长度，不 pad 到 512；
- sample / negative prompt 聚合到 task 档案根（``tasks/<id>/.text-cache/``）的
  一个 safetensors 文件——不落数据集 train/，避免被数据集扫描当 concept
  文件夹误触（D19 修订）；
- 写入使用 sibling tmp + ``os.replace``，训练中断不会留下半文件。
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional


TEXT_CACHE_FORMAT_VERSION = 1
_SIDECAR_SUFFIX = ".text.safetensors"
_PROMPT_CACHE_DIR = ".text-cache"
_PROMPT_CACHE_PREFIX = "prompts"
_TENSOR_SEPARATOR = "::"


@dataclass(frozen=True)
class TextCacheEntry:
    """一张图片的最终 caption 与其 sidecar 位置。"""

    image_path: Path
    caption: str
    cache_path: Path

    @classmethod
    def for_image(cls, image_path, caption: str) -> "TextCacheEntry":
        image = Path(image_path)
        return cls(
            image_path=image,
            caption=str(caption),
            cache_path=caption_sidecar_path(image),
        )


def caption_sha256(caption: str) -> str:
    """最终 caption 内容 hash（D3/D19 的独立失效分量）。"""

    return hashlib.sha256(str(caption).encode("utf-8")).hexdigest()


def text_cache_key(
    caption: str,
    text_fingerprint: str,
    *,
    format_version: int = TEXT_CACHE_FORMAT_VERSION,
) -> str:
    """返回 caption + TE 指纹 + 格式版本的稳定内容键。"""

    payload = (
        f"text-cache-v{int(format_version)}\0{text_fingerprint}\0{caption}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def caption_sidecar_path(image_path) -> Path:
    """图片 caption sidecar；保留原扩展名，避免 ``a.jpg``/``a.png`` 冲突。"""

    image = Path(image_path)
    return image.with_name(image.name + _SIDECAR_SUFFIX)


def prompt_cache_path(root, text_fingerprint: str) -> Path:
    """sample/negative prompt 聚合缓存路径（按 TE 指纹隔离）。

    ``root`` 由调用方决定——训练里传 task 档案根（``tasks/<id>/``），聚合
    文件落 ``<root>/.text-cache/``。
    """

    fp_short = hashlib.sha256(str(text_fingerprint).encode("utf-8")).hexdigest()[:12]
    return (
        Path(root)
        / _PROMPT_CACHE_DIR
        / f"{_PROMPT_CACHE_PREFIX}.{fp_short}.safetensors"
    )


def _normalise_tensors(tensors: Mapping[str, object]) -> dict:
    import torch

    if not tensors:
        raise ValueError("文本缓存至少需要一个 tensor")
    out = {}
    for name, value in tensors.items():
        key = str(name)
        if not key or _TENSOR_SEPARATOR in key:
            raise ValueError(f"非法文本缓存 tensor 名: {name!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"文本缓存值必须是 torch.Tensor: {key}")
        out[key] = value.detach().cpu().contiguous()
    return out


def _atomic_save(
    tensors: Mapping[str, object], metadata: Mapping[str, str], path: Path,
) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    # 同一数据集允许两个训练 task 并发预缓存：tmp 名带 pid/thread，最终文件仍由
    # os.replace 竞争为任一完整版本，不会互相截断临时文件。
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        save_file(dict(tensors), str(tmp_path), metadata=dict(metadata))
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _load(path: Path) -> tuple[dict, dict[str, str]]:
    from safetensors import safe_open

    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata


class TextCacheStore:
    """绑定一个 TE 指纹的 sidecar / prompt bundle 读写器。"""

    def __init__(
        self,
        text_fingerprint: str,
        *,
        format_version: int = TEXT_CACHE_FORMAT_VERSION,
    ) -> None:
        fingerprint = str(text_fingerprint).strip()
        if not fingerprint:
            raise ValueError("text_fingerprint 不能为空")
        self.text_fingerprint = fingerprint
        self.format_version = int(format_version)

    def key(self, caption: str) -> str:
        return text_cache_key(
            caption,
            self.text_fingerprint,
            format_version=self.format_version,
        )

    def write_caption(self, entry: TextCacheEntry, tensors: Mapping[str, object]) -> None:
        payload = _normalise_tensors(tensors)
        metadata = {
            "cache_kind": "caption_sidecar",
            "format_version": str(self.format_version),
            "text_fingerprint": self.text_fingerprint,
            "caption_sha256": caption_sha256(entry.caption),
            "cache_key": self.key(entry.caption),
        }
        _atomic_save(payload, metadata, entry.cache_path)

    def read_caption(self, entry: TextCacheEntry) -> Optional[dict]:
        """命中返回 CPU tensors；缺失、损坏或任一指纹失配返回 ``None``。"""

        if not entry.cache_path.is_file():
            return None
        try:
            tensors, metadata = _load(entry.cache_path)
        except Exception:
            return None
        expected = {
            "cache_kind": "caption_sidecar",
            "format_version": str(self.format_version),
            "text_fingerprint": self.text_fingerprint,
            "caption_sha256": caption_sha256(entry.caption),
            "cache_key": self.key(entry.caption),
        }
        if any(metadata.get(k) != v for k, v in expected.items()):
            return None
        return tensors or None

    def get_or_encode_caption(
        self,
        entry: TextCacheEntry,
        encoder: Callable[[str], Mapping[str, object]],
    ) -> tuple[dict, bool]:
        """读取 sidecar；miss 时编码并原子覆盖。返回 ``(tensors, was_hit)``。"""

        cached = self.read_caption(entry)
        if cached is not None:
            return cached, True
        encoded = _normalise_tensors(encoder(entry.caption))
        self.write_caption(entry, encoded)
        return encoded, False

    def write_prompt_bundle(
        self,
        root,
        encoded: Mapping[str, Mapping[str, object]],
    ) -> Path:
        """原子覆盖 sample/negative prompt 聚合缓存，tensor 可各自不同长度。"""

        payload = {}
        metadata = {
            "cache_kind": "prompt_bundle",
            "format_version": str(self.format_version),
            "text_fingerprint": self.text_fingerprint,
        }
        for caption, tensors in encoded.items():
            digest = self.key(str(caption))
            normalised = _normalise_tensors(tensors)
            metadata[f"entry.{digest}"] = caption_sha256(str(caption))
            for name, tensor in normalised.items():
                payload[f"{digest}{_TENSOR_SEPARATOR}{name}"] = tensor
        path = prompt_cache_path(root, self.text_fingerprint)
        if payload:
            _atomic_save(payload, metadata, path)
        else:
            path.unlink(missing_ok=True)
        return path

    def get_or_encode_prompts(
        self,
        root,
        captions,
        encoder: Callable[[str], Mapping[str, object]],
    ) -> tuple[dict[str, dict], int]:
        """批量读取聚合缓存并编码 misses；返回 ``(caption→tensors, hit 数)``。"""

        unique = list(dict.fromkeys(str(caption) for caption in captions))
        encoded: dict[str, dict] = {}
        hit_count = 0
        dirty = False
        for caption in unique:
            cached = self.read_prompt(root, caption)
            if cached is not None:
                encoded[caption] = cached
                hit_count += 1
                continue
            encoded[caption] = _normalise_tensors(encoder(caption))
            dirty = True
        if dirty:
            self.write_prompt_bundle(root, encoded)
        return encoded, hit_count

    def read_prompt(self, root, caption: str) -> Optional[dict]:
        path = prompt_cache_path(root, self.text_fingerprint)
        if not path.is_file():
            return None
        try:
            tensors, metadata = _load(path)
        except Exception:
            return None
        digest = self.key(caption)
        if (
            metadata.get("cache_kind") != "prompt_bundle"
            or metadata.get("format_version") != str(self.format_version)
            or metadata.get("text_fingerprint") != self.text_fingerprint
            or metadata.get(f"entry.{digest}") != caption_sha256(caption)
        ):
            return None
        prefix = f"{digest}{_TENSOR_SEPARATOR}"
        found = {
            key[len(prefix):]: value
            for key, value in tensors.items()
            if key.startswith(prefix)
        }
        return found or None
