"""latent2rgb 中间步预览 decode 的回归测试（CPU，无需 GPU / 模型 / 下载）。

背景：Anima 的 VAE 是 Qwen-Image VAE（models/vae/qwen_image_vae.safetensors），其
latent 空间就是 Wan2.1 的 16-ch 空间（ComfyUI `QwenImage.latent_format = Wan21`，
且本仓库 VAE 归一化 mean/std 与 ComfyUI `Wan21.latents_mean/std` 逐位一致）。之前
预览误用 TAEFlux（Flux VAE 的 tiny decoder）解码这个 WAN latent → 颜色反相/错乱。
改用 Wan2.1 `latent_rgb_factors` 线性投影（latent2rgb，即 ComfyUI 快速预览）。

本测试锁定：投影公式与 ComfyUI Latent2RGBPreviewer 逐像素一致 + 输出为放大后的
RGB 图，防止再次回归到错误 decoder / 错误范围映射。
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from runtime.anima_daemon import (
    _PREVIEW_TARGET_PX,
    _WAN21_LATENT_RGB_BIAS,
    _WAN21_LATENT_RGB_FACTORS,
    _decode_latent2rgb_preview,
)


def test_wan21_factors_shape() -> None:
    # 16 通道 → RGB：16 行 × 3 列 + 3 个 bias。
    assert len(_WAN21_LATENT_RGB_FACTORS) == 16
    assert all(len(row) == 3 for row in _WAN21_LATENT_RGB_FACTORS)
    assert len(_WAN21_LATENT_RGB_BIAS) == 3


def test_decode_returns_upscaled_rgb_image() -> None:
    # Anima latent shape [B, 16, F=1, H, W]；小 latent → 放大到 target 最长边、保持比例。
    latent = torch.randn(1, 16, 1, 16, 16)
    img = _decode_latent2rgb_preview(latent)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert max(img.size) == _PREVIEW_TARGET_PX  # 16px < target → 放大到 target
    assert img.size[0] == img.size[1]  # 1:1 latent → 1:1 图


def test_decode_matches_comfy_latent2rgb_math() -> None:
    # 用「最长边 ≥ target」的 latent 跳过放大（scale ≤ 1 不 resize），直接核对原始投影
    # 与 ComfyUI Latent2RGBPreviewer 公式逐像素一致。
    h, w = _PREVIEW_TARGET_PX, _PREVIEW_TARGET_PX // 3
    latent = torch.randn(1, 16, 1, h, w)
    img = _decode_latent2rgb_preview(latent)
    assert img.size == (w, h)  # PIL size 是 (W, H)，未放大

    x0 = latent[0, :, 0].float()  # [16, H, W]
    factors = torch.tensor(_WAN21_LATENT_RGB_FACTORS)
    bias = torch.tensor(_WAN21_LATENT_RGB_BIAS)
    rgb = torch.einsum("chw,cr->hwr", x0, factors) + bias  # [H, W, 3]
    rgb = ((rgb + 1.0) / 2.0).clamp(0.0, 1.0)  # [-1,1] → [0,1]，同 ComfyUI preview_to_image
    expected = (rgb.numpy() * 255).astype(np.uint8)
    assert np.array_equal(np.asarray(img), expected)


def test_decode_never_crashes_on_bad_input() -> None:
    # preview 不阻塞主流程：形状不对时返 None 而非抛。
    assert _decode_latent2rgb_preview(torch.randn(3, 3)) is None
