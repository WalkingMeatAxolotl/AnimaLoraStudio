"""SaturationPairDataset：POC image-pair dataset。

每个 sample 同时返回原图（高饱和 = pos）和 PIL 去色版（低饱和 = neg），
同分辨率同 crop。caption 不读 .txt/.json，统一短文本——POC 目的是让
LoRA **只学色差轴**，文本侧零信号最干净。

设计 trade-off：
- 固定单分辨率（args.resolution）：跳 ARB bucket。pair 必须 shape 一致，
  否则 4-forward step 里 noisy_pos / noisy_neg shape mismatch
- on-the-fly PIL desaturate：不 cache。POC 数据量小（30-80 张），每 step
  从磁盘 reload 不是瓶颈
- 不参与 reg / merged：reg 集语义跟 slider 任务冲突
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SaturationPairDataset(Dataset):
    """saturation tweaker POC 数据集。

    Args:
        data_dir: 图片目录（递归扫描；不解析 Kohya 文件夹 repeat 前缀，POC 简化）
        resolution: 强制单分辨率（pair shape 必须相同）
        caption: 所有图共用同一 caption；POC 推荐保持空或短词
        neg_strength: 负向 pair 的 ImageEnhance.Color factor（0.0=全灰度，1.0=原图）。
            默认 0.5 = 半饱和。
            **不要用 0.0**：那会让 LoRA 把"去色"整条向量学进去（亮度/对比度
            PIL 数学上保 L 没事，但色温 / 色相会被吸走），weight=-1 推理时
            退化成黑白 tweaker。0.5 让 axis 更短更纯，推理 weight 动态范围 +。
    """

    def __init__(
        self,
        data_dir: str | Path,
        resolution: int = 1024,
        caption: str = "a photo",
        neg_strength: float = 0.5,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = int(resolution)
        self.caption = caption
        self.neg_strength = float(neg_strength)
        self.image_paths: list[Path] = []
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir 不存在: {self.data_dir}")
        for p in sorted(self.data_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                self.image_paths.append(p)
        if not self.image_paths:
            raise RuntimeError(f"data_dir 下没扫到图片: {self.data_dir}")
        logger.info(
            f"SaturationPairDataset: {len(self.image_paths)} 张图 @ {self.resolution}x{self.resolution}, "
            f"caption=\"{self.caption}\", neg_strength={self.neg_strength}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")

        # 中心 crop 到正方形 + resize 到目标分辨率（POC 跳 ARB）
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)

        # pos = 原图，neg = 弱化饱和（默认 0.5；不走 0.0 避免变成"去色 tweaker"）
        img_pos = img
        img_neg = ImageEnhance.Color(img).enhance(self.neg_strength)

        return {
            "pixel_pos": self._to_tensor(img_pos),
            "pixel_neg": self._to_tensor(img_neg),
            "caption": self.caption,
        }


def collate_pair(batch: list[dict]) -> dict:
    """pair dataset collate：分别 stack pos / neg。"""
    return {
        "pixel_pos": torch.stack([b["pixel_pos"] for b in batch]),
        "pixel_neg": torch.stack([b["pixel_neg"] for b in batch]),
        "captions": [b["caption"] for b in batch],
    }


def build_dataloader(ctx) -> None:
    """concept slider 模式下替代 phases.dataset.run 的最小 builder。

    塞回 ctx.dataloader / ctx.dataset / ctx.base_dataset / ctx.use_cached
    保证后续 phases（optimizer / resume / loop）读字段不踩空。
    """
    from torch.utils.data import DataLoader

    args = ctx.args
    caption = str(getattr(args, "slider_caption", "") or "a photo")
    neg_strength = float(getattr(args, "slider_neg_strength", 0.5))
    ds = SaturationPairDataset(
        args.data_dir, resolution=args.resolution,
        caption=caption, neg_strength=neg_strength,
    )
    ctx.base_dataset = ds
    ctx.dataset = ds
    ctx.use_cached = False  # POC 不走 cached latent 路径

    # POC：dataset_phase 里 Windows num_workers 守卫沿用
    import os
    if args.num_workers > 0 and os.name == "nt":
        logger.warning("Windows 强制 num_workers=0（slider POC 简化）")
        args.num_workers = 0

    # ARB / bucket 全不用，shuffle=True 即可
    ctx.dataloader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pair, num_workers=args.num_workers, drop_last=False,
    )

    # VAE roundtrip 自检：把 pos 走一遍 encode-decode，验证 vae/dtype/scale 没问题
    try:
        item0 = ds[0]
        pixels0 = item0["pixel_pos"].unsqueeze(0).to(ctx.device, dtype=ctx.dtype)
        with torch.no_grad():
            z0 = ctx.vae.model.encode(pixels0.unsqueeze(2), ctx.vae.scale)
            recon0 = ctx.vae.model.decode(z0, ctx.vae.scale).squeeze(2)
            recon0 = (recon0.clamp(-1, 1) + 1) / 2
        arr0 = (recon0[0].permute(1, 2, 0).detach().cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
        Image.fromarray(arr0).save(ctx.sample_dir / "vae_roundtrip.png")
        logger.info("VAE roundtrip 自检已保存: samples/vae_roundtrip.png")
    except Exception as e:
        logger.warning(f"VAE roundtrip 自检失败: {e}")

    # 算 steps_per_epoch（optimizer_phase 要用；正常路径在 dataset_phase 后由
    # optimizer_phase 自己算，但 slider 这边走自定义路径，先填好 fallback）
    ctx.steps_per_epoch = max(1, len(ds) // max(1, args.batch_size))
