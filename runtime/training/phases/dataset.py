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
    NavitPackBatchSampler,
    collate_fn,
    collate_fn_cached,
    collate_fn_navit_pack,
)


logger = logging.getLogger(__name__)


def _as_resolutions(value) -> list[int]:
    """把 config 的 resolution 归一成 list[int]。

    schema 标量、schema 列表、手写 YAML 标量都可能出现（merge_yaml_into_namespace
    是裸 setattr、不过 pydantic validator），这里统一兜底成非空 list。
    """
    if isinstance(value, (list, tuple)):
        out = [int(v) for v in value]
        return out or [1024]
    return [int(value)]


def _rope_max_side_tokens(ctx) -> int:
    """模型 RoPE 单边可寻址的 patch-token 上限（= max_img_h // patch_spatial）。

    NaViT 原生定尺寸据此在数据层封顶单边，避免超 ``_packed_rope_from_grid`` 的前向 fail-fast
    （白白浪费一次全量 VAE 缓存）。取不到（如无模型的单测）→ 0（不早封顶，仍由前向 fail-fast 兜底）。
    """
    pe = getattr(getattr(ctx, "model", None), "pos_embedder", None)
    try:
        return int(min(int(pe.max_h), int(pe.max_w)))
    except Exception:
        return 0


def _parse_token_ladder(value) -> list[int]:
    """把 navit_multiscale_token_ladder（逗号分隔字符串 / 列表 / 标量）解析成正整数 list。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace("，", ",").split(",")
    out = []
    for it in items:
        try:
            n = int(str(it).strip())
        except (TypeError, ValueError):
            continue
        if n > 0:
            out.append(n)
    return sorted(set(out))


def _native_dataset_kwargs(args, ctx) -> dict:
    """navit_native_resolution 开启时给 ImageDataset 的原生定尺寸参数；关闭 → 空 dict（行为中立）。"""
    navit_packing = bool(getattr(args, "navit_packing", False))
    if not (navit_packing and bool(getattr(args, "navit_native_resolution", False))):
        return {}
    kwargs = dict(
        native_resolution=True,
        native_token_budget=int(getattr(args, "navit_token_budget", 0) or 0),
        native_over_budget=str(getattr(args, "navit_native_over_budget", "downscale") or "downscale"),
        native_max_side_tokens=_rope_max_side_tokens(ctx),
    )
    logger.info(
        "[navit-native] 原生定尺寸已启用：单图 floor 对齐 16px、零 padding，绕过 ARB 桶量化；"
        "超预算策略=%s，token 预算=%d，RoPE 单边上限=%s tokens。",
        kwargs["native_over_budget"], kwargs["native_token_budget"],
        kwargs["native_max_side_tokens"] or "不限",
    )
    # navit 多尺度阶梯（navit_multiscale）：为原生大图追加低 token 档副本参与打包。
    if bool(getattr(args, "navit_multiscale", False)):
        ladder = _parse_token_ladder(getattr(args, "navit_multiscale_token_ladder", ""))
        kwargs["native_ms_token_ladder"] = ladder
        kwargs["native_ms_loss_weight"] = float(getattr(args, "navit_multiscale_loss_weight", 1.0) or 1.0)
        logger.info(
            "[navit-multiscale] 多尺度阶梯已启用：token 档=%s，副本 loss 权重=%.3g（1.0=等权）。",
            ladder or "（空，无副本）", kwargs["native_ms_loss_weight"],
        )
    return kwargs


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

    # 多分辨率：args.resolution 可能是标量或列表（手写 YAML 也可能是标量），统一归一成
    # list；base 档取第一项，其它档的 BucketManager 由 ImageDataset 按需建。
    res_list = _as_resolutions(args.resolution)
    ar_limit = float(getattr(args, "aspect_ratio_limit", 2.0))
    base_reso = res_list[0]

    # NaViT 原生定尺寸参数（navit_native_resolution）；关闭时为空 dict → 行为中立。
    native_kwargs = _native_dataset_kwargs(args, ctx)

    # 数据集
    ctx.bucket_mgr = BucketManager(base_reso, aspect_ratio_limit=ar_limit)
    ctx.base_dataset = ImageDataset(
        args.data_dir, base_reso, ctx.bucket_mgr,
        shuffle_caption=args.shuffle_caption,
        keep_tokens=args.keep_tokens,
        flip_augment=args.flip_augment,
        tag_dropout=args.tag_dropout,
        prefer_json=args.prefer_json,
        resolutions=res_list,
        aspect_ratio_limit=ar_limit,
        **native_kwargs,
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
                reg_data_dir, base_reso, ctx.bucket_mgr,
                shuffle_caption=args.shuffle_caption,
                keep_tokens=args.keep_tokens,
                flip_augment=args.flip_augment,
                tag_dropout=0.0,  # 正则集通常不用 dropout
                prefer_json=args.prefer_json,
                caption_override=reg_caption if reg_caption else None,
                resolutions=res_list,
                aspect_ratio_limit=ar_limit,
                **native_kwargs,
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
            ctx.dataset, ctx.vae, ctx.device, ctx.vae_dtype,
            cache_batch_size=cache_batch_size,
            encode_tiled=getattr(args, "cache_encode_tiled", False),
            encode_tile_px=getattr(args, "cache_encode_tile_px", 1024),
            encode_tile_overlap=getattr(args, "cache_encode_tile_overlap", 128),
            encode_max_pixels=getattr(args, "cache_encode_max_pixels", 0),
        )
    if ctx.reg_dataset is not None and ctx.use_cached:
        ctx.reg_dataset = CachedLatentDataset(
            ctx.reg_dataset, ctx.vae, ctx.device, ctx.vae_dtype,
            cache_batch_size=cache_batch_size,
            encode_tiled=getattr(args, "cache_encode_tiled", False),
            encode_tile_px=getattr(args, "cache_encode_tile_px", 1024),
            encode_tile_overlap=getattr(args, "cache_encode_tile_overlap", 128),
            encode_max_pixels=getattr(args, "cache_encode_max_pixels", 0),
        )

    # repeat: 主数据集和正则数据集均通过文件夹名 Kohya 风格 repeat（如 5_concept），无需全局 repeat
    if ctx.reg_dataset is not None:
        reg_weight = float(getattr(args, "reg_weight", 1.0) or 1.0)
        ctx.dataset = MergedDataset(ctx.dataset, ctx.reg_dataset, reg_weight=reg_weight)

    if args.num_workers > 0 and os.name == "nt":
        logger.warning("num_workers > 0 在 Windows 上容易崩溃：已强制设为 0（避免多进程 spawn 问题）")
        args.num_workers = 0

    if getattr(args, "navit_packing", False):
        # NaViT / Patch-n-Pack 块对角打包：按 token 预算把多张不同尺寸的图拼进
        # 一个训练序列（零 padding），替代 ARB 固定桶分批。需配合 cache_latents。
        batch_sampler = NavitPackBatchSampler(
            ctx.dataset,
            token_budget=int(getattr(args, "navit_token_budget", 16384) or 16384),
            max_images_per_pack=int(getattr(args, "navit_max_images_per_pack", 0) or 0),
            shuffle=True,
            seed=getattr(args, "seed", 42),
            drop_last=getattr(args, "navit_drop_last", False),
            strategy=getattr(args, "navit_pack_strategy", "next_fit"),
            # 不用 `or 256`：0 是合法值（全局 FFD，每 epoch 包固定），会被 falsy 吞掉。
            ffd_window=int(getattr(args, "navit_pack_ffd_window", 256)),
        )
        ctx.dataloader = DataLoader(
            ctx.dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn_navit_pack,
            num_workers=args.num_workers,
        )
    elif ctx.use_cached:
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
            num_workers=args.num_workers,
        )
    else:
        # 非缓存路径也必须按桶分批：collate_fn 用 torch.stack 拼 pixel_values，一个 batch
        # 混入不同桶尺寸（ARB 下不同长宽比 → 不同 H×W）会 RuntimeError。BucketBatchSampler
        # 靠 ImageDataset.bucket_for_index 把同尺寸样本分进同一 batch（缓存路径早已这么做，
        # 非缓存路径此前漏了 → bs>1 必崩）。drop_last=False 与缓存路径一致。
        batch_sampler = BucketBatchSampler(
            ctx.dataset, batch_size=args.batch_size,
            drop_last=False, shuffle=True,
            seed=getattr(args, "seed", 42),
        )
        ctx.dataloader = DataLoader(
            ctx.dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )

    # 训练前自检：VAE encode->decode 循环（快速排除 VAE/scale/shape 问题）
    try:
        if len(ctx.base_dataset) > 0:
            from PIL import Image
            item0 = ctx.base_dataset[0]
            pixels0 = item0["pixel_values"].unsqueeze(0).to(ctx.device, dtype=ctx.dtype)  # [1,3,H,W]
            with torch.no_grad():
                # encode/decode 均走 VAEWrapper（含 auto/on 分块），避免大图整图 op 触发系统内存回退卡死
                z0 = ctx.vae.encode(pixels0.unsqueeze(2))                        # [1,16,1,h,w]
                recon0 = ctx.vae.decode(z0).squeeze(2)                           # [1,3,H,W]
                recon0 = (recon0.clamp(-1, 1) + 1) / 2
            arr0 = (recon0[0].permute(1, 2, 0).detach().cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(arr0).save(ctx.sample_dir / "vae_roundtrip.png")
            logger.info("VAE roundtrip 自检已保存: samples/vae_roundtrip.png")
    except Exception as e:
        logger.warning(f"VAE roundtrip 自检失败（若 sample 仍是噪点，请优先修这个）: {e}")
