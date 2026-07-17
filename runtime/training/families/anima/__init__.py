"""Anima 族的声明式常量（PR-1 只有 ANIMA_SPEC；行为适配器随 PR-2b 落地）。"""

from __future__ import annotations

# 能力矩阵 / 族默认 / 采样白名单单源在 studio/domain/common.py（config 管线
# 刀 1 / R3）：studio 不 import runtime（server sys.path 无 runtime/），反向
# runtime → studio 是既有依赖方向（bootstrap 已 import studio.schema），
# common.py 是零依赖纯数据叶子模块，server / 裸 CLI / tests 三场景都可达。
from studio.domain.common import (
    FAMILY_CAPABILITIES,
    FAMILY_CONFIG_DEFAULTS,
    FAMILY_SAMPLING,
)

# 相对导入：studio server 经 `runtime.training.dataset` 间接 import 本模块，
# 那边 sys.path 没有 runtime/，`training.*` 绝对导入会 ModuleNotFoundError。
from ..latent_spaces import WAN21_F8C16
from ..spec import (
    ConstantShift,
    LoraOutputSpec,
    ModelSpec,
    SamplingDefaults,
    TextSpec,
)


ANIMA_SPEC = ModelSpec(
    family_id="anima",
    display_name="Anima",
    objective="rectified_flow",
    # Qwen-Image VAE = Wan2.1 latent 空间；与 Krea 2 引用同一实例 →
    # latent 缓存跨族共享（D6）是结构事实。
    latent=WAN21_F8C16,
    text=TextSpec(
        # 每步在线编码（Qwen3-0.6B 末层 + T5 IDs 进 LLMAdapter），无文本缓存
        strategy="online",
        max_seq_len=512,
        fingerprint="anima-qwen3-0.6b-t5xxl",
    ),
    sampling=SamplingDefaults(
        # 白名单对应 sampling.py Comfy KSampler parity 现状；首项 = 族默认
        samplers=FAMILY_SAMPLING["anima"]["samplers"],
        schedulers=FAMILY_SAMPLING["anima"]["schedulers"],
        default_sampler=FAMILY_SAMPLING["anima"]["samplers"][0],
        default_scheduler=FAMILY_SAMPLING["anima"]["schedulers"][0],
        default_steps=25,
        default_cfg=4.0,
        shift_policy=ConstantShift(shift=3.0),
    ),
    # D5：Anima = 全量能力减 text_cache
    capabilities=FAMILY_CAPABILITIES["anima"],
    lora=LoraOutputSpec(prefix="lora_unet", preset_name="anima_full"),
    config_defaults=FAMILY_CONFIG_DEFAULTS["anima"],
)
