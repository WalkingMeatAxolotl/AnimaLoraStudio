"""训练噪声生成：基础高斯 + noise_offset + pyramid 多尺度低频。

抽自原 runtime/anima_train.py L1848-1896（ADR 0003 PR-A）。

参数化的纯数学函数，未来加新 noise scheme（如 fp_offset_v2）直接在本文件
增 if 分支即可，不需要 plugin subfolder。
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def make_noise(
    latents: torch.Tensor,
    noise_offset: float = 0.0,
    pyramid_iters: int = 0,
    pyramid_discount: float = 0.35,
) -> torch.Tensor:
    """生成训练噪声，可叠加低频扰动。

    noise_offset   — 给每样本/通道加低频偏移，缓解亮度均值偏差（SDXL 思路）
    pyramid_iters  — 叠加多尺度低频噪声，帮助模型快速学习全局光照/构图；
                     bilinear 插值避免 nearest 的块状结构干扰
    """
    noise = torch.randn_like(latents)

    if noise_offset > 0:
        shape = list(latents.shape)
        for ax in range(2, latents.ndim):
            shape[ax] = 1
        offset = torch.randn(*shape, device=latents.device, dtype=latents.dtype)
        noise = noise + noise_offset * offset

    if pyramid_iters > 0:
        try:
            spatial = list(latents.shape[-2:])
            cur = noise.clone()
            for i in range(pyramid_iters):
                r = 2 ** (i + 1)
                sh, sw = max(spatial[0] // r, 1), max(spatial[1] // r, 1)
                if latents.ndim == 5:
                    extra = torch.randn(
                        latents.shape[0], latents.shape[1], latents.shape[2], sh, sw,
                        device=latents.device, dtype=latents.dtype,
                    )
                    extra = F.interpolate(
                        extra.flatten(0, 1), size=spatial, mode="bilinear", align_corners=False,
                    ).view(latents.shape[0], latents.shape[1], latents.shape[2], *spatial)
                else:
                    extra = torch.randn(latents.shape[0], latents.shape[1], sh, sw,
                                        device=latents.device, dtype=latents.dtype)
                    extra = F.interpolate(extra, size=spatial, mode="bilinear", align_corners=False)
                cur = cur + extra * (pyramid_discount ** (i + 1))
                if min(sh, sw) <= 1:
                    break
            noise = cur / cur.std().clamp(min=1e-6)
        except Exception as exc:
            logger.warning(f"pyramid_noise 失败，回退标准噪声: {exc}")

    return noise
