"""Anima 族的声明式常量（PR-1 只有 ANIMA_SPEC；行为适配器随 PR-2b 落地）。"""

from __future__ import annotations

from training.families.spec import (
    ConstantShift,
    LatentSpec,
    LoraOutputSpec,
    ModelSpec,
    SamplingDefaults,
    TextSpec,
)

# latent2rgb 快速预览的线性投影系数（"模糊但能看出图"）。取自 ComfyUI
# comfy/latent_formats.py 的 `Wan21.latent_rgb_factors[_bias]` —— Anima 的
# Qwen-Image VAE 与 Wan2.1 同 latent 空间（docs/design/multi-model/00 §2、D17）。
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


ANIMA_SPEC = ModelSpec(
    family_id="anima",
    display_name="Anima",
    objective="rectified_flow",
    latent=LatentSpec(
        # latent 空间身份：Qwen-Image VAE = Wan2.1 latent 空间，f8、16ch。
        # Krea 2 同值 → latent 缓存跨族共享（D6）。
        fingerprint="wan21-f8c16",
        channels=16,
        spatial_stride=8,
        patch_spatial=2,
        patch_temporal=1,
        temporal=False,
        rgb_factors=_WAN21_RGB_FACTORS,
        rgb_bias=_WAN21_RGB_BIAS,
    ),
    text=TextSpec(
        # 每步在线编码（Qwen3-0.6B 末层 + T5 IDs 进 LLMAdapter），无文本缓存
        strategy="online",
        max_seq_len=512,
        fingerprint="anima-qwen3-0.6b-t5xxl",
    ),
    sampling=SamplingDefaults(
        # 白名单与默认值对应 sampling.py Comfy KSampler parity 现状
        samplers=("er_sde", "dpmpp_3m_sde"),
        schedulers=("simple", "sgm_uniform"),
        default_sampler="er_sde",
        default_scheduler="simple",
        default_steps=25,
        default_cfg=4.0,
        shift_policy=ConstantShift(shift=3.0),
    ),
    # D5：Anima = 全量能力减 text_cache
    capabilities=frozenset({
        "navit", "sra", "leap", "compile_blocks",
        "caption_tag_ops", "online_text", "masked_loss",
    }),
    lora=LoraOutputSpec(prefix="lora_unet", preset_name="anima_full"),
    config_defaults={},
)
