"""FLUX.2 VAE 加载 + 本项目 latent 排布适配（跨族共享资产，同 training.vae）。

排布（`latent_spaces.FLUX2_F8C32_BN`，MOSS 设计稿 §5.1 布局 B）：

    encode: pixels → encoder → mean(32ch, f8) → patchify(128ch, f16)
            → (z − bn_mean) / sqrt(bn_var + eps) → unpatchify(32ch, f8)
    decode: 逆过程

patchify 的通道顺序 ``c·4 + pi·2 + pj`` 与 BFL ``rearrange("c (i pi) (j pj) ->
(c pi pj) i j")``、diffusers ``_patchify_latents``、ComfyUI
``Flux2.latent_rgb_factors_reshape`` 一致，所以 ``patchify(本项目 latent)`` 就是
ComfyUI / BFL 的 128ch latent，逐元素相等。

适配器暴露与 WAN VAE 相同的 ``encode(x, scale)`` / ``decode(z, scale)`` /
``z_dim`` 接口，交给 ``VAEWrapper`` 复用整图 / 分块决策与 OOM 回退。共享循环与
缓存都经 ``vae.model.encode(pixels_5d, vae.scale)`` / ``vae._tiled_encode`` 取
latent，归一化因此必须在适配器这一层完成。
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from studio.infrastructure.log_messages import msg
from training.model_loading import _load_safetensors_state_dict
from training.vae import VAEWrapper

logger = logging.getLogger(__name__)

#: patchify 边长（FLUX.2 `ps = [2, 2]`）
_PATCH = 2
#: BFL `AutoEncoder.bn_eps`
_BN_EPS = 1e-4
#: 单文件 checkpoint 里可能出现的外层前缀（Comfy 整合包 / LDM 习惯）
_KNOWN_PREFIXES = ("vae.", "first_stage_model.")


def patchify(z: torch.Tensor) -> torch.Tensor:
    """``[b, c, h, w]`` → ``[b, c·4, h/2, w/2]``，通道序 ``c·4 + pi·2 + pj``。"""
    b, c, h, w = z.shape
    z = z.reshape(b, c, h // _PATCH, _PATCH, w // _PATCH, _PATCH)
    return z.permute(0, 1, 3, 5, 2, 4).reshape(b, c * _PATCH * _PATCH, h // _PATCH, w // _PATCH)


def unpatchify(z: torch.Tensor) -> torch.Tensor:
    """``patchify`` 的逆：``[b, c·4, h, w]`` → ``[b, c, 2h, 2w]``。"""
    b, c4, h, w = z.shape
    c = c4 // (_PATCH * _PATCH)
    z = z.reshape(b, c, _PATCH, _PATCH, h, w)
    return z.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * _PATCH, w * _PATCH)


class Flux2LatentAdapter(nn.Module):
    """包住 BFL ``AutoEncoder``，按布局 B 进出 latent（5D，t 恒为 1）。

    ``scale`` 是 ``VAEWrapper`` 传入的 ``[bn_mean, 1/bn_std]``（各 128 维，
    patchify 后的通道序），与 WAN VAE 的 ``scale`` 参数位置同构。
    """

    def __init__(self, autoencoder: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder
        self.z_dim = int(autoencoder.params.z_channels)

    @staticmethod
    def _weight_dtype(module: nn.Module) -> torch.dtype:
        # 输入按权重精度 cast：共享循环（vae.model.encode）与缓存分块
        # （vae._tiled_encode）都绕过 VAEWrapper.encode 的入口 cast 直接调到这里
        return next(module.parameters()).dtype

    @staticmethod
    def _check_latent_hw(h: int, w: int) -> None:
        if h % _PATCH or w % _PATCH:
            raise ValueError(
                f"FLUX.2 latent 尺寸 {h}×{w} 须为 {_PATCH} 的倍数（像素须为 16 的倍数）"
            )

    def _normalize(self, z: torch.Tensor, scale) -> torch.Tensor:
        mean, inv_std = scale
        zp = patchify(z.float())
        zp = (zp - mean.float().view(1, -1, 1, 1)) * inv_std.float().view(1, -1, 1, 1)
        return unpatchify(zp)

    def _denormalize(self, z: torch.Tensor, scale) -> torch.Tensor:
        mean, inv_std = scale
        zp = patchify(z.float())
        zp = zp / inv_std.float().view(1, -1, 1, 1) + mean.float().view(1, -1, 1, 1)
        return unpatchify(zp)

    def encode(self, x: torch.Tensor, scale) -> torch.Tensor:
        """``[b, 3, 1, H, W]`` → ``[b, 32, 1, H/8, W/8]``（布局 B，已归一化）。"""
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError(f"FLUX.2 VAE 只处理单帧图像，收到形状 {tuple(x.shape)}")
        moments = self.autoencoder.encoder(x[:, :, 0].to(self._weight_dtype(self.autoencoder.encoder)))
        mu = torch.chunk(moments, 2, dim=1)[0]
        self._check_latent_hw(mu.shape[-2], mu.shape[-1])
        return self._normalize(mu, scale).to(x.dtype).unsqueeze(2)

    def decode(self, z: torch.Tensor, scale) -> torch.Tensor:
        """``[b, 32, 1, h, w]`` → ``[b, 3, 1, 8h, 8w]``（未 clamp，同 WAN 口径）。"""
        if z.ndim != 5 or z.shape[2] != 1:
            raise ValueError(f"FLUX.2 VAE 只处理单帧 latent，收到形状 {tuple(z.shape)}")
        z2 = z[:, :, 0]
        self._check_latent_hw(z2.shape[-2], z2.shape[-1])
        raw = self._denormalize(z2, scale).to(self._weight_dtype(self.autoencoder.decoder))
        return self.autoencoder.decoder(raw).to(z.dtype).unsqueeze(2)


def _strip_known_prefix(sd: dict) -> dict:
    for prefix in _KNOWN_PREFIXES:
        if sd and all(k.startswith(prefix) for k in sd):
            return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def load_flux2_vae(vae_path, device, dtype, *, tiling: str = "auto") -> VAEWrapper:
    """加载 BFL / ComfyUI 格式的 FLUX.2 VAE 单文件（``flux2-vae.safetensors``）。

    键名必须与 BFL ``AutoEncoder`` 完全一致（strict）；diffusers 分片目录格式
    （``AutoencoderKLFlux2``）不在本函数支持范围。
    """
    from modeling.flux2.autoencoder import AutoEncoder, AutoEncoderParams

    autoencoder = AutoEncoder(AutoEncoderParams()).eval().requires_grad_(False)
    sd = _strip_known_prefix(_load_safetensors_state_dict(Path(vae_path)))
    result = autoencoder.load_state_dict(sd, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"{vae_path} 不是 BFL / ComfyUI 格式的 FLUX.2 VAE："
            f"缺少 {list(result.missing_keys)[:5]}，多出 {list(result.unexpected_keys)[:5]}"
        )
    autoencoder = autoencoder.to(device=device, dtype=dtype)

    bn = autoencoder.bn
    mean = bn.running_mean.detach().to(device=device, dtype=dtype)
    std = torch.sqrt(bn.running_var.detach().float() + _BN_EPS).to(device=device, dtype=dtype)
    logger.info(msg("train.vae_loaded"))
    return VAEWrapper(Flux2LatentAdapter(autoencoder), mean, std, tiling=tiling, latent_align=_PATCH)
