"""text_cache_phase：构造随图 caption sidecar 计划并交给 ModelFamily 编码。

位置固定在 dataset 之后、optimizer 之前。dataset 仍只产出 ``captions``；缓存命中
与编码张量形状完全由 family 自治。Anima 的 online 策略走零开销 no-op。
"""

from __future__ import annotations

import logging
from pathlib import Path

from training.context import TrainingContext
from training.text_cache import TextCacheEntry


logger = logging.getLogger(__name__)


def _image_dataset(dataset):
    """从 latent/repeat wrapper 中找到拥有 samples + caption resolver 的底层集。"""

    seen = set()
    current = dataset
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "samples") and hasattr(current, "caption_for_sample"):
            return current
        next_dataset = getattr(current, "base_image_dataset", None)
        if next_dataset is None:
            next_dataset = getattr(current, "base_dataset", None)
        if next_dataset is None:
            next_dataset = getattr(current, "dataset", None)
        current = next_dataset
    return None


def _collect_entries(ctx: TrainingContext) -> list[TextCacheEntry]:
    """主集 + 正则集按图片去重；分辨率 fan-out/repeat 不重复编码文本。"""

    by_image: dict[str, TextCacheEntry] = {}
    for dataset in (ctx.base_dataset, ctx.reg_dataset):
        base = _image_dataset(dataset)
        if base is None:
            continue
        for sample in base.samples:
            image = Path(sample["image"])
            caption = base.caption_for_sample(sample)
            key = str(image)
            existing = by_image.get(key)
            if existing is not None:
                if existing.caption != caption:
                    raise ValueError(
                        f"同一图片解析出两个不同 caption，无法建立文本缓存: {image}"
                    )
                continue
            by_image[key] = TextCacheEntry.for_image(image, caption)
    return list(by_image.values())


def _extra_prompts(args) -> list[str]:
    prompts = getattr(args, "sample_prompts", None) or []
    if isinstance(prompts, str):
        prompts = [prompts]
    out = [str(p) for p in prompts if p is not None]
    single = getattr(args, "sample_prompt", "") or ""
    if single and single not in out:
        out.append(str(single))
    sampling_enabled = bool(
        int(getattr(args, "sample_steps", 0) or 0)
        or int(getattr(args, "sample_every", 0) or 0)
    )
    if sampling_enabled and not out:
        out.append("1girl, masterpiece")
    # CFG 无条件分支也需要编码；空串是合法且有意义的 prompt。
    out.append(str(getattr(args, "sample_negative_prompt", "") or ""))
    return list(dict.fromkeys(out))


def run(ctx: TrainingContext) -> None:
    strategy = ctx.family.spec.text.strategy
    if strategy == "online":
        ctx.family.prepare_text_cache([], [])
        return
    if strategy != "cached_varlen":  # registry 正常会先拦，这里保留可操作错误
        raise ValueError(f"未知文本策略: {strategy!r}")

    if not bool(getattr(ctx.args, "text_encoder_cache", True)):
        logger.info(
            "[text-cache] 缓存已关闭：不扫描/读写 sidecar，文本编码器常驻并逐 batch 编码"
        )
        ctx.family.prepare_text_cache(
            [],
            [],
            text=ctx.text_stack,
            device=ctx.device,
            dtype=ctx.dtype,
        )
        return

    entries = _collect_entries(ctx)
    captions = [entry.caption for entry in entries]
    extras = _extra_prompts(ctx.args)
    cache_root = Path(ctx.args.data_dir)
    logger.info(
        "[text-cache] 预缓存 %d 张图片 caption + %d 条采样/负面 prompt（varlen）",
        len(entries), len(extras),
    )
    ctx.family.prepare_text_cache(
        captions,
        extras,
        cache_entries=entries,
        cache_root=cache_root,
        text=ctx.text_stack,
        device=ctx.device,
        dtype=ctx.dtype,
    )
