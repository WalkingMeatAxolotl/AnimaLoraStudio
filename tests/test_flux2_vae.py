"""FLUX.2 VAE 适配（MOSS 设计稿 §5.1 布局 B / §5.2）单测。

测：
- patchify 通道序与 BFL rearrange / ComfyUI ``Flux2.latent_rgb_factors_reshape`` 一致
- 适配器 encode / decode 与 BFL ``AutoEncoder.encode/decode`` 互为 reshape（逐元素相等）
- ``VAEWrapper`` 分块 encode 产出 32ch，且分块拼回与整图一致（归一化对齐全图 patch 网格）
- 分块尺寸不是 16px 整倍数时拒绝
- ``load_flux2_vae`` 按 BFL / Comfy 键名 strict 加载（容忍 ``vae.`` 前缀），错文件报错
- ``FLUX2_F8C32_BN`` 常量与共享几何假设一致
"""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

import modeling.flux2.autoencoder as ae_mod
from modeling.flux2.autoencoder import AutoEncoder, AutoEncoderParams
from training.families.latent_spaces import FLUX2_F8C32_BN, WAN21_F8C16
from training.vae import VAEWrapper
from training.vae_flux2 import Flux2LatentAdapter, load_flux2_vae, patchify, unpatchify


def _small_params() -> AutoEncoderParams:
    # GroupNorm(32) 要求通道为 32 的倍数；ch_mult 保持 4 级 → 仍是 f8
    return AutoEncoderParams(ch=32, ch_mult=[1, 2, 4, 4], num_res_blocks=1, z_channels=32)


def _small_autoencoder(seed: int = 0) -> AutoEncoder:
    torch.manual_seed(seed)
    ae = AutoEncoder(_small_params()).eval()
    with torch.no_grad():
        ae.bn.running_mean.copy_(torch.randn(128) * 0.3)
        ae.bn.running_var.copy_(torch.rand(128) * 2 + 0.5)
    return ae


def _wrapper(ae: AutoEncoder, **kwargs) -> VAEWrapper:
    mean = ae.bn.running_mean.clone()
    std = torch.sqrt(ae.bn.running_var + 1e-4)
    return VAEWrapper(Flux2LatentAdapter(ae), mean, std, latent_align=2, **kwargs)


def test_patchify_channel_order_matches_comfy_reshape() -> None:
    z = torch.randn(2, 32, 6, 10)
    zp = patchify(z)
    assert zp.shape == (2, 128, 3, 5)
    # ComfyUI latent_formats.Flux2.latent_rgb_factors_reshape（128ch/f16 → 32ch/f8）
    comfy_back = zp.reshape(2, 32, 2, 2, 3, 5).permute(0, 1, 4, 2, 5, 3).reshape(2, 32, 6, 10)
    assert torch.equal(comfy_back, z)
    assert torch.equal(unpatchify(zp), z)


def test_adapter_encode_is_bfl_latent_unpatchified() -> None:
    ae = _small_autoencoder()
    wrapper = _wrapper(ae)
    x = torch.randn(1, 3, 1, 32, 48)
    with torch.no_grad():
        ours = wrapper.model.encode(x, wrapper.scale)
        ref = ae.encode(x[:, :, 0])  # BFL：patchify + BN 归一化后的 128ch
    assert ours.shape == (1, 32, 1, 4, 6)
    torch.testing.assert_close(patchify(ours[:, :, 0]), ref, atol=1e-5, rtol=1e-5)


def test_adapter_decode_matches_bfl_decode() -> None:
    ae = _small_autoencoder()
    wrapper = _wrapper(ae)
    z_ref = torch.randn(1, 128, 2, 3)
    with torch.no_grad():
        ours = wrapper.model.decode(unpatchify(z_ref).unsqueeze(2), wrapper.scale)
        ref = ae.decode(z_ref)
    assert ours.shape == (1, 3, 1, 32, 48)
    torch.testing.assert_close(ours[:, :, 0], ref, atol=1e-5, rtol=1e-5)


def test_adapter_rejects_odd_latent_and_multi_frame() -> None:
    wrapper = _wrapper(_small_autoencoder())
    with pytest.raises(ValueError, match="16 的倍数"):
        wrapper.model.encode(torch.randn(1, 3, 1, 40, 48), wrapper.scale)  # latent 5×6
    with pytest.raises(ValueError, match="单帧"):
        wrapper.model.encode(torch.randn(1, 3, 2, 32, 32), wrapper.scale)


class _PointwiseEncoder(torch.nn.Module):
    """逐 8×8 块平均 + 1×1 投影：块与块独立，分块拼回应与整图逐元素相等。"""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.proj = torch.nn.Conv2d(3, 64, 1)

    def forward(self, x):
        return self.proj(torch.nn.functional.avg_pool2d(x, 8))


def test_wrapper_tiled_encode_matches_full_with_patch_aligned_tiles() -> None:
    ae = _small_autoencoder()
    ae.encoder = _PointwiseEncoder()
    wrapper = _wrapper(ae)
    assert wrapper.latent_channels == 32
    pixels = torch.randn(1, 3, 1, 400, 272)  # 50×34 latent，尾块起点非 tile 整倍数
    with torch.no_grad():
        full = wrapper.model.encode(pixels, wrapper.scale)
        tiled = wrapper._tiled_encode(pixels, tile_px=256, overlap_px=64)
    assert tiled.shape == (1, 32, 1, 50, 34)
    # 若块内 patch 分组与全图错位，子像素位置的归一化统计会用错，这里必然对不上
    torch.testing.assert_close(tiled, full, atol=1e-5, rtol=1e-5)


def test_adapter_casts_input_to_weight_dtype_on_direct_calls() -> None:
    """共享循环（vae.model.encode）与缓存分块（vae._tiled_encode）绕过 VAEWrapper.encode
    的入口 cast；bf16 权重 + fp32 像素曾在 conv_in 报 dtype 不一致。"""
    ae = _small_autoencoder().to(torch.bfloat16)
    wrapper = _wrapper(ae)
    pixels = torch.randn(1, 3, 1, 128, 128)  # fp32
    with torch.no_grad():
        z = wrapper.model.encode(pixels, wrapper.scale)
        tiled = wrapper._tiled_encode(pixels, tile_px=256, overlap_px=64)
        img = wrapper.model.decode(z, wrapper.scale)
    assert z.dtype == tiled.dtype == img.dtype == torch.float32


def test_wrapper_tiled_encode_rejects_non_16px_tiles() -> None:
    wrapper = _wrapper(_small_autoencoder())
    with pytest.raises(ValueError, match="16px"):
        wrapper._tiled_encode(torch.randn(1, 3, 1, 64, 64), tile_px=264, overlap_px=64)


def test_wrapper_rejects_decode_tiling_not_aligned() -> None:
    with pytest.raises(ValueError, match="latent_align"):
        VAEWrapper(Flux2LatentAdapter(_small_autoencoder()), torch.zeros(128), torch.ones(128),
                   latent_align=5)


def test_load_flux2_vae_strict_with_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ae_mod, "AutoEncoderParams", _small_params)
    ae = _small_autoencoder(seed=3)
    path = tmp_path / "flux2-vae.safetensors"
    save_file({f"vae.{k}": v.contiguous() for k, v in ae.state_dict().items()}, str(path))

    wrapper = load_flux2_vae(path, "cpu", torch.float32)
    assert wrapper.latent_channels == 32
    assert wrapper.latent_align == 2
    torch.testing.assert_close(wrapper.mean, ae.bn.running_mean)
    torch.testing.assert_close(wrapper.std, torch.sqrt(ae.bn.running_var + 1e-4))
    x = torch.randn(1, 3, 1, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(
            patchify(wrapper.encode(x)[:, :, 0]), ae.encode(x[:, :, 0]), atol=1e-5, rtol=1e-5,
        )


def test_load_flux2_vae_rejects_wrong_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ae_mod, "AutoEncoderParams", _small_params)
    sd = _small_autoencoder().state_dict()
    sd.pop("bn.running_mean")
    path = tmp_path / "not-flux2.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    with pytest.raises(ValueError, match="不是 BFL / ComfyUI 格式的 FLUX.2 VAE"):
        load_flux2_vae(path, "cpu", torch.float32)


def test_flux2_latent_spec_constants() -> None:
    lat = FLUX2_F8C32_BN
    assert lat.channels == 32 == len(lat.rgb_factors)
    assert lat.temporal is False
    # 与 Wan2.1 空间同几何（共享 dataset / masked loss 假设零改动），但指纹不同（不共享缓存）
    assert (lat.spatial_stride, lat.patch_spatial, lat.align_px) == (
        WAN21_F8C16.spatial_stride, WAN21_F8C16.patch_spatial, WAN21_F8C16.align_px,
    )
    assert lat.fingerprint != WAN21_F8C16.fingerprint
