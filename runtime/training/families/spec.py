"""ModelSpec —— 模型族的声明式常量（多模型支持 PR-1）。

纯数据、frozen、无行为。设计出处：docs/design/multi-model/03-interface-evolution.md
§2.1（字段集为 v1 冻结面）+ 04-synthesis.md D11/D12/D17。

本文件在 PR-1 阶段先承担两件事：
- 把散落在 dataset / sampling / timestep_sampling / phases 的 latent 规格
  （z_dim / stride / patch / 对齐单元 / latent2rgb 系数）收敛为单一来源；
- 为 latent npz 缓存提供指纹（fingerprint 是 **latent 空间身份**而非族名：
  Anima 与 Krea 2 同为 ``wan21-f8c16`` → 缓存跨族共享，D6）。

ModelFamily 行为接口（八方法 + convert_lora_state_dict）在 PR-2b 落地。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Union


@dataclass(frozen=True)
class ConstantShift:
    """固定 timestep shift（Anima：3.0，对应 Comfy parity sampling_settings）。"""

    shift: float


@dataclass(frozen=True)
class ResolutionAwareShift:
    """分辨率感知动态 shift（Krea 2：base 0.5 → max 1.15 按 image_seq_len 插值）。

    只影响采样端 sigma 时刻表与 schema 默认值 overlay；训练端 t 采样走
    timestep_samplers plugin（04-synthesis D14）。
    """

    base_shift: float
    max_shift: float
    base_image_seq_len: int
    max_image_seq_len: int


ShiftPolicy = Union[ConstantShift, ResolutionAwareShift]


@dataclass(frozen=True)
class LatentSpec:
    #: latent 空间身份（非族名），进 npz 缓存指纹键
    fingerprint: str
    #: z_dim（VAE latent 通道数）
    channels: int
    #: VAE 空间下采样倍数（f8）
    spatial_stride: int
    #: DiT patchify 空间边长（2 → 1 token = 16×16 px）
    patch_spatial: int
    patch_temporal: int
    #: 视频拒绝位（03 §3.2）：v1 恒 False，registry 注册期校验
    temporal: bool
    #: latent2rgb 预览线性投影 [channels][3]（D17，取自 ComfyUI LatentFormat）
    rgb_factors: tuple[tuple[float, float, float], ...]
    rgb_bias: tuple[float, float, float]

    @property
    def align_px(self) -> int:
        """桶 / 采样尺寸的像素对齐单元 = spatial_stride × patch_spatial。"""
        return self.spatial_stride * self.patch_spatial


@dataclass(frozen=True)
class TextSpec:
    #: Anima=online（每步在线编码）；K2=cached_varlen（varlen 预缓存，D3）
    strategy: Literal["online", "cached_varlen"]
    max_seq_len: int
    #: TE 指纹（cached_varlen 的缓存键成分；online 族仅记录用）
    fingerprint: str


@dataclass(frozen=True)
class SamplingDefaults:
    #: sampler / scheduler 白名单与默认值（schema overlay 与采样端消费）
    samplers: tuple[str, ...]
    schedulers: tuple[str, ...]
    default_sampler: str
    default_scheduler: str
    default_steps: int
    default_cfg: float
    shift_policy: ShiftPolicy


@dataclass(frozen=True)
class LoraOutputSpec:
    #: 保存键名前缀（两族统一 "lora_unet"，04-synthesis §7.1）
    prefix: str
    #: lycoris preset 标识（进 ss_network_args.preset）
    preset_name: str


@dataclass(frozen=True)
class ModelSpec:
    #: registry 键 & schema enum 值，永不改名（进 yaml / LoRA metadata / resume state）
    family_id: str
    display_name: str
    #: 非 rectified-flow 拒绝位（03 §3.3）：v1 唯一合法值
    objective: Literal["rectified_flow"]
    latent: LatentSpec
    text: TextSpec
    sampling: SamplingDefaults
    capabilities: frozenset[str]
    lora: LoraOutputSpec
    #: 创建 version 时叠进初始 yaml 的 per-family 默认值 overlay（作者写时落盘）
    config_defaults: Mapping[str, Any] = field(default_factory=dict)


#: 能力词表（03 §2.4）。加词零成本；删词 / 改语义须过 04-synthesis 评审。
KNOWN_CAPABILITIES = frozenset({
    "navit", "sra", "leap", "compile_blocks",
    "caption_tag_ops", "online_text", "text_cache", "masked_loss",
})


def validate_spec(spec: ModelSpec) -> None:
    """registry 注册期自洽校验（违反 → ValueError，进程启动即死）。"""
    unknown = spec.capabilities - KNOWN_CAPABILITIES
    if unknown:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] 未知能力位: {sorted(unknown)}"
        )
    if spec.text.strategy == "cached_varlen" and "caption_tag_ops" in spec.capabilities:
        # 交叉不变量（03 §2.4）：缓存键 = caption 内容 hash，tag shuffle/dropout
        # 会让每步 caption 漂移、缓存永 miss。
        raise ValueError(
            f"ModelSpec[{spec.family_id}] cached_varlen 与 caption_tag_ops 互斥"
        )
    if spec.latent.temporal:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] temporal=True：v1 不支持视频族（03 §3.2）"
        )
    if len(spec.latent.rgb_factors) != spec.latent.channels:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] rgb_factors 行数 "
            f"{len(spec.latent.rgb_factors)} != channels {spec.latent.channels}"
        )
    if any(len(row) != 3 for row in spec.latent.rgb_factors) or len(spec.latent.rgb_bias) != 3:
        raise ValueError(f"ModelSpec[{spec.family_id}] rgb_factors/bias 必须是 RGB 三元")
    if spec.sampling.default_sampler not in spec.sampling.samplers:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] default_sampler 不在白名单 {spec.sampling.samplers}"
        )
    if spec.sampling.default_scheduler not in spec.sampling.schedulers:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] default_scheduler 不在白名单 {spec.sampling.schedulers}"
        )
