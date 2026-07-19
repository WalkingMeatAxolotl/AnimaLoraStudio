"""族无关的 latent 空间常量（共享外部事实层）。

latent 空间是 VAE 的属性，不是模型族的属性——多个族可以共用同一空间
（D6：同指纹自动共享 npz 缓存）。空间的定义（归一化统计 / latent2rgb
投影系数）来自上游单一事实源，在此定义一次，族 spec 直接引用同一实例，
使「跨族共享」成为结构事实（同一性）而非各族副本 + 测试维持的相等性。

Wan2.1 空间：由 ComfyUI `comfy/latent_formats.py` 的 `Wan21` 定义；
Qwen-Image VAE 复用它（`comfy/supported_models.py`
`QwenImage.latent_format = Wan21`）。Anima 与 Krea 2 同用此空间（D6/D17）。
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
