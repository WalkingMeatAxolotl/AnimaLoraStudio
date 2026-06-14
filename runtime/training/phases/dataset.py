"""dataset_phase：build datasets + dataloader + VAE roundtrip 自检。

抽自 main() L257-342（ADR 0003 PR-B）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.context import TrainingContext
from training.dataset import (
    BucketBatchSampler,
    BucketManager,
    CachedLatentDataset,
    ImageDataset,
    MergedDataset,
    collate_fn,
    collate_fn_cached,
)


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """
    - 主数据集 / 正则数据集 + per-folder repeat
    - cache_latents 包 CachedLatentDataset
    - MergedDataset 串联主集 + 正则集
    - Windows num_workers > 0 兜底为 0（多进程 spawn 易崩）
    - BucketBatchSampler / DataLoader
    - VAE encode-decode 循环自检（vae_roundtrip.png）
    """
    args = ctx.args

    # 数据集
    ctx.bucket_mgr = BucketManager(
        args.resolution,
        constant_token_mode=getattr(args, "torch_compile", False),
    )
    ctx.base_dataset = ImageDataset(
        args.data_dir, args.resolution, ctx.bucket_mgr,
        shuffle_caption=args.shuffle_caption,
        keep_tokens=args.keep_tokens,
        flip_augment=args.flip_augment,
        tag_dropout=args.tag_dropout,
        prefer_json=args.prefer_json,
    )
    ctx.dataset = ctx.base_dataset

    # 正则数据集（Kohya 风格，防过拟合）
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    ctx.reg_dataset = None
    if reg_data_dir:
        if not Path(reg_data_dir).exists():
            logger.warning(f"正则数据集路径不存在，已跳过: {reg_data_dir}")
        elif len(ctx.base_dataset) == 0:
            logger.warning("主数据集为空，正则集已跳过")
        else:
            reg_caption = (getattr(args, "reg_caption", "") or "").strip()
            reg_base = ImageDataset(
                reg_data_dir, args.resolution, ctx.bucket_mgr,
                shuffle_caption=args.shuffle_caption,
                keep_tokens=args.keep_tokens,
                flip_augment=args.flip_augment,
                tag_dropout=0.0,  # 正则集通常不用 dropout
                prefer_json=args.prefer_json,
                caption_override=reg_caption if reg_caption else None,
            )
            ctx.reg_dataset = reg_base
            reg_weight = float(getattr(args, "reg_weight", 1.0) or 1.0)
            cap_preview = f", caption=\"{reg_caption[:50]}{'...' if len(reg_caption) > 50 else ''}\"" if reg_caption else ""
            weight_info = f", weight={reg_weight}" if reg_weight != 1.0 else ""
            logger.info(f"正则数据集: {reg_data_dir} ({len(reg_base)} 样本, per-folder repeat{weight_info}){cap_preview}")

    # 缓存 VAE latents（在 repeat 之前）
    ctx.use_cached = getattr(args, "cache_latents", False)
    if ctx.use_cached:
        # 0 = 跟随训练 batch size（对齐 kohya GUI 的 VAE batch size 语义）
        cache_batch_size = int(getattr(args, "vae_cache_batch_size", 0) or 0)
        if cache_batch_size <= 0:
            cache_batch_size = int(getattr(args, "batch_size", 1) or 1)
        ctx.dataset = CachedLatentDataset(
            ctx.dataset, ctx.vae, ctx.device, ctx.dtype,
            cache_batch_size=cache_batch_size,
        )
    if ctx.reg_dataset is not None and ctx.use_cached:
        ctx.reg_dataset = CachedLatentDataset(
            ctx.reg_dataset, ctx.vae, ctx.device, ctx.dtype,
            cache_batch_size=cache_batch_size,
        )

    # repeat: 主数据集和正则数据集均通过文件夹名 Kohya 风格 repeat（如 5_concept），无需全局 repeat
    if ctx.reg_dataset is not None:
        reg_weight = float(getattr(args, "reg_weight", 1.0) or 1.0)
        ctx.dataset = MergedDataset(ctx.dataset, ctx.reg_dataset, reg_weight=reg_weight)

    if args.num_workers > 0 and os.name == "nt":
        logger.warning("num_workers > 0 在 Windows 上容易崩溃：已强制设为 0（避免多进程 spawn 问题）")
        args.num_workers = 0

    dl_kwargs: dict = dict(
        num_workers=args.num_workers,
        pin_memory=getattr(args, "pin_memory", True) and torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = getattr(args, "prefetch_factor", 2)
        dl_kwargs["persistent_workers"] = True

    if ctx.use_cached:
        # drop_last=False：桶尾不足 batch_size 出短 batch 而非丢图。
        # 对齐 kohya sd-scripts / ostris ai-toolkit；diffusion 用 LayerNorm/GroupNorm，
        # 对动态 batch 不敏感，loop.py 也按 latents.shape[0] 动态读 bs。
        batch_sampler = BucketBatchSampler(
            ctx.dataset, batch_size=args.batch_size,
            drop_last=False, shuffle=True,
            seed=getattr(args, "seed", 42),
        )
        ctx.dataloader = DataLoader(
            ctx.dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn_cached,
            **dl_kwargs,
        )
    else:
        ctx.dataloader = DataLoader(
            ctx.dataset, batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **dl_kwargs,
        )

    # 文本编码缓存（预计算 Qwen hidden + T5 token ids）
    if ctx.use_cached and getattr(args, "cache_text_embeds", False):
        _build_text_embed_cache(ctx)

    # 训练前自检：VAE encode->decode 循环（快速排除 VAE/scale/shape 问题）
    try:
        if len(ctx.base_dataset) > 0:
            from PIL import Image
            item0 = ctx.base_dataset[0]
            pixels0 = item0["pixel_values"].unsqueeze(0).to(ctx.device, dtype=ctx.dtype)  # [1,3,H,W]
            with torch.no_grad():
                z0 = ctx.vae.model.encode(pixels0.unsqueeze(2), ctx.vae.scale)   # [1,16,1,h,w]
                recon0 = ctx.vae.model.decode(z0, ctx.vae.scale).squeeze(2)      # [1,3,H,W]
                recon0 = (recon0.clamp(-1, 1) + 1) / 2
            arr0 = (recon0[0].permute(1, 2, 0).detach().cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(arr0).save(ctx.sample_dir / "vae_roundtrip.png")
            logger.info("VAE roundtrip 自检已保存: samples/vae_roundtrip.png")
    except Exception as e:
        logger.warning(f"VAE roundtrip 自检失败（若 sample 仍是噪点，请优先修这个）: {e}")


def _build_text_embed_cache(ctx: TrainingContext) -> None:
    """预计算所有静态 caption 的文本编码并写入 npz（与 latent cache 共存）。"""
    import numpy as np

    from training.text_encoding import (
        _build_qwen_text_from_prompt,
        encode_qwen,
        tokenize_t5_comfy_literal,
        tokenize_t5_weighted,
    )

    args = ctx.args
    comfy_mode = bool(getattr(args, "caption_comfy_encoding", True))
    cache_mode_tag = "comfy" if comfy_mode else "legacy"

    cached_datasets: list = []
    if hasattr(ctx.dataset, "main_dataset"):
        cached_datasets.append(ctx.dataset.main_dataset)
        if ctx.dataset.reg_dataset:
            cached_datasets.append(ctx.dataset.reg_dataset)
    elif isinstance(ctx.dataset, CachedLatentDataset):
        cached_datasets.append(ctx.dataset)

    for cds in cached_datasets:
        if not isinstance(cds, CachedLatentDataset):
            continue
        _encode_text_for_dataset(cds, ctx, comfy_mode, cache_mode_tag)


def _encode_text_for_dataset(
    cds: "CachedLatentDataset",
    ctx: TrainingContext,
    comfy_mode: bool,
    cache_mode_tag: str,
) -> None:
    """为单个 CachedLatentDataset 编码全部 caption。"""
    import numpy as np

    from training.text_encoding import (
        _build_qwen_text_from_prompt,
        encode_qwen,
        tokenize_t5_comfy_literal,
        tokenize_t5_weighted,
    )

    to_encode: list[int] = []
    seen_npz: set = set()

    for i, sample in enumerate(cds.samples):
        npz_path = cds._get_npz_path(sample["image"])
        if npz_path in seen_npz:
            continue
        seen_npz.add(npz_path)
        if npz_path.exists():
            with np.load(npz_path) as data:
                if "qwen_emb" in data.files:
                    stored_mode = str(data["text_cache_mode"]) if "text_cache_mode" in data.files else "comfy"
                    if stored_mode == cache_mode_tag:
                        continue
        to_encode.append(i)

    if not to_encode:
        logger.info(f"文本编码缓存已就绪（{len(seen_npz)} 样本）")
        return

    logger.info(f"预计算文本编码: {len(to_encode)}/{len(seen_npz)} 样本...")

    base = cds.base_dataset
    while hasattr(base, "dataset"):
        base = base.dataset

    for count, i in enumerate(to_encode, 1):
        sample = cds.samples[i]
        npz_path = cds._get_npz_path(sample["image"])

        caption = ""
        if getattr(base, "caption_override", None) is not None:
            caption = base.caption_override
        elif sample.get("json_path") and hasattr(base, "_process_caption_json"):
            caption = base._process_caption_json(sample["json_path"]) or ""
        elif sample.get("txt_path"):
            raw = sample["txt_path"].read_text(encoding="utf-8").strip()
            if hasattr(base, "_process_caption_txt"):
                caption = base._process_caption_txt(raw)
            else:
                caption = raw

        if comfy_mode:
            qwen_text = str(caption)
        else:
            qwen_text = _build_qwen_text_from_prompt(caption)

        with torch.no_grad():
            qwen_emb, _qwen_attn = encode_qwen(
                ctx.qwen_model, ctx.qwen_tok, [qwen_text], ctx.device, max_length=512,
            )
            qwen_emb_np = qwen_emb[0].cpu().float().numpy()

            if comfy_mode:
                t5_ids, t5_attn, t5_w = tokenize_t5_comfy_literal(
                    ctx.t5_tok, [caption], max_length=512,
                )
            else:
                t5_ids, t5_attn, t5_w = tokenize_t5_weighted(
                    ctx.t5_tok, [caption], max_length=512,
                )
            t5_ids_np = t5_ids[0].numpy()
            t5_attn_np = t5_attn[0].numpy().astype(np.int8)
            t5_w_np = t5_w[0].numpy()

        existing = dict(np.load(npz_path))
        existing["qwen_emb"] = qwen_emb_np
        existing["t5_ids"] = t5_ids_np
        existing["t5_attn"] = t5_attn_np
        existing["t5_w"] = t5_w_np
        existing["text_cache_mode"] = np.array(cache_mode_tag)

        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=str(npz_path.parent))
        os.close(tmp_fd)
        try:
            np.savez(tmp_path, **existing)
            os.replace(tmp_path, npz_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        if count % 20 == 0 or count == len(to_encode):
            logger.info(f"  文本编码进度: {count}/{len(to_encode)}")
