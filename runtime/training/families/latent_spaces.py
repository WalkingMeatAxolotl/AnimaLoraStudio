"""族无关的 latent 空间常量（共享外部事实层）。

latent 空间是 VAE 的属性，不是模型族的属性——多个族可以共用同一空间
（D6：同指纹自动共享 npz 缓存）。空间的定义（归一化统计 / latent2rgb
投影系数）来自上游单一事实源，在此定义一次，族 spec 直接引用同一实例，
使「跨族共享」成为结构事实（同一性）而非各族副本 + 测试维持的相等性。

Wan2.1 空间：由 ComfyUI `comfy/latent_formats.py` 的 `Wan21` 定义；
Qwen-Image VAE 复用它（`comfy/supported_models.py`
`QwenImage.latent_format = Wan21`）。Anima 与 Krea 2 同用此空间（D6/D17）。

FLUX.2 空间：ComfyUI `latent_formats.Flux2` 定义为 128ch、f16（VAE 原生 32ch f8
latent 经 2×2 patchify 后再用 VAE 自带 BatchNorm 统计量归一化）。本项目缓存的是
它的等价排布——归一化后再 unpatchify 回 32ch、f8（MOSS 设计稿 §5.1 布局 B）：
归一化是逐 (通道, 子像素位置) 的仿射、patchify 是纯 reshape，两者可交换，信息
完全等价；reshape 保持 iid，共享循环里的加噪 / 目标代数不受影响；且
spatial_stride / patch_spatial 取值与 Wan2.1 空间相同，共享 dataset / 缓存 /
masked loss 的几何假设无需改动。
"""

from __future__ import annotations

# 相对导入：studio server 经 `runtime.training.*` 命名间接 import 本模块（bucket
# 分布预览），那边 sys.path 没有 runtime/，`training.*` 绝对导入会 ModuleNotFoundError。
from .spec import LatentSpec

# latent2rgb 快速预览的线性投影系数（"模糊但能看出图"）。取自 ComfyUI
# comfy/latent_formats.py 的 `Wan21.latent_rgb_factors[_bias]`（GPL-3.0，
# 见 THIRD_PARTY_NOTICES）。
_WAN21_RGB_FACTORS: tuple[tuple[float, float, float], ...] = (
    (-0.1299, -0.1692, 0.2932),
    (0.0671, 0.0406, 0.0442),
    (0.3568, 0.2548, 0.1747),
    (0.0372, 0.2344, 0.1420),
    (0.0313, 0.0189, -0.0328),
    (0.0296, -0.0956, -0.0665),
    (-0.3477, -0.4059, -0.2925),
    (0.0166, 0.1902, 0.1975),
    (-0.0412, 0.0267, -0.1364),
    (-0.1293, 0.0740, 0.1636),
    (0.0680, 0.3019, 0.1128),
    (0.0032, 0.0581, 0.0639),
    (-0.1251, 0.0927, 0.1699),
    (0.0060, -0.0633, 0.0005),
    (0.3477, 0.2275, 0.2950),
    (0.1984, 0.0913, 0.1861),
)
_WAN21_RGB_BIAS: tuple[float, float, float] = (-0.1835, -0.0868, -0.3360)

#: Qwen-Image VAE = Wan2.1 latent 空间，f8、16ch。
WAN21_F8C16 = LatentSpec(
    fingerprint="wan21-f8c16",
    channels=16,
    spatial_stride=8,
    patch_spatial=2,
    patch_temporal=1,
    temporal=False,
    rgb_factors=_WAN21_RGB_FACTORS,
    rgb_bias=_WAN21_RGB_BIAS,
)

# 取自 ComfyUI comfy/latent_formats.py 的 `Flux2.latent_rgb_factors[_bias]`
# （GPL-3.0，见 THIRD_PARTY_NOTICES）。Comfy 先用 `latent_rgb_factors_reshape`
# 把 128ch/f16 latent 还原成 32ch/f8 再乘这组系数——正是本项目缓存的布局，
# 所以直接按 32 通道使用。
_FLUX2_RGB_FACTORS: tuple[tuple[float, float, float], ...] = (
    (0.0058, 0.0113, 0.0073),
    (0.0495, 0.0443, 0.0836),
    (-0.0099, 0.0096, 0.0644),
    (0.2144, 0.3009, 0.3652),
    (0.0166, -0.0039, -0.0054),
    (0.0157, 0.0103, -0.0160),
    (-0.0398, 0.0902, -0.0235),
    (-0.0052, 0.0095, 0.0109),
    (-0.3527, -0.2712, -0.1666),
    (-0.0301, -0.0356, -0.0180),
    (-0.0107, 0.0078, 0.0013),
    (0.0746, 0.0090, -0.0941),
    (0.0156, 0.0169, 0.0070),
    (-0.0034, -0.0040, -0.0114),
    (0.0032, 0.0181, 0.0080),
    (-0.0939, -0.0008, 0.0186),
    (0.0018, 0.0043, 0.0104),
    (0.0284, 0.0056, -0.0127),
    (-0.0024, -0.0022, -0.0030),
    (0.1207, -0.0026, 0.0065),
    (0.0128, 0.0101, 0.0142),
    (0.0137, -0.0072, -0.0007),
    (0.0095, 0.0092, -0.0059),
    (0.0000, -0.0077, -0.0049),
    (-0.0465, -0.0204, -0.0312),
    (0.0095, 0.0012, -0.0066),
    (0.0290, -0.0034, 0.0025),
    (0.0220, 0.0169, -0.0048),
    (-0.0332, -0.0457, -0.0468),
    (-0.0085, 0.0389, 0.0609),
    (-0.0076, 0.0003, -0.0043),
    (-0.0111, -0.0460, -0.0614),
)
_FLUX2_RGB_BIAS: tuple[float, float, float] = (-0.0329, -0.0718, -0.0851)

#: FLUX.2 VAE latent，BN 归一化后、未 patchify 的排布：f8、32ch（见模块 docstring）。
FLUX2_F8C32_BN = LatentSpec(
    fingerprint="flux2-f8c32-bn",
    channels=32,
    spatial_stride=8,
    patch_spatial=2,
    patch_temporal=1,
    temporal=False,
    rgb_factors=_FLUX2_RGB_FACTORS,
    rgb_bias=_FLUX2_RGB_BIAS,
)
