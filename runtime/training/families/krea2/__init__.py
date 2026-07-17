"""Krea2 ModelSpec and ModelFamily implementation (multi-model Phase 3)."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch

from .preset import KREA2_PRESET
from .sampling import (
    KREA2_BASE_IMAGE_SEQ_LEN,
    KREA2_BASE_SHIFT,
    KREA2_MAX_IMAGE_SEQ_LEN,
    KREA2_MAX_SHIFT,
    KREA2_RAW_GUIDANCE,
    KREA2_RAW_STEPS,
    KREA2_SAMPLER,
    KREA2_SCHEDULER,
    Krea2SamplingCondition,
    prepare_sampling_condition,
    resolve_sampling_settings,
    sample_image,
)
from .text_encoding import (
    KREA2_TEXT_FINGERPRINT,
    Krea2TextCondition,
    Krea2TextStack,
    load_krea2_text_stack,
)
from ..latent_spaces import WAN21_F8C16
from ..spec import (
    LoraOutputSpec,
    ModelSpec,
    ConstantShift,
    SamplingDefaults,
    TextSpec,
)
# 单源数据（刀 1 / R3）：见 anima/__init__.py 同位注释的依赖方向说明
from studio.domain.common import (
    FAMILY_CAPABILITIES,
    FAMILY_CONFIG_DEFAULTS,
    FAMILY_SAMPLING,
)


logger = logging.getLogger(__name__)


_GIB = 1024 ** 3
#: TE 上卡所需：fp16 权重 8.9GB + embed 表逐层 cast fp32 瞬时 ~1.5GB + 缓冲
_TE_LOAD_NEED_BYTES = int(11 * _GIB)


def _cuda_free_bytes(device) -> int | None:
    """目标 CUDA 设备当前空闲显存；非 CUDA / 查询失败返回 None。"""
    try:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(dev)
        return int(free)
    except Exception:
        return None


def _sampling_headroom_bytes(height: int, width: int) -> int:
    # 采样余量粗估不做 per-model 记账（编排方案 D1 用户拍板）：
    # 2GB 底 + 3GB × 面积比（1024² → 5GB，1536² → 8.75GB）
    area_ratio = (height * width) / (1024 * 1024)
    return int((2.0 + 3.0 * area_ratio) * _GIB)


def _should_yield_dit(policy: str, device) -> bool:
    """编码前 TE 需要搬上 GPU 时，DiT 是否先撤到 CPU（comfy free_memory
    的「装不下才让位」语义）。"""
    if policy == "performance":
        return False
    if policy == "save_vram":
        return True
    free = _cuda_free_bytes(device)
    return free is not None and free < _TE_LOAD_NEED_BYTES


def _should_offload_te(policy: str, device, height: int, width: int,
                       dit_yielded: bool) -> bool:
    """采样前是否把 TE 卸到 CPU。auto 只在采样余量不足时卸——32GB fp8
    三者同驻 free 充裕，从此零搬运。"""
    if policy == "performance":
        return False
    if policy == "save_vram":
        return True
    if dit_yielded:
        # DiT 刚让过位 = 显存装不下三者同驻，采样期必须让 DiT 独占；
        # 且此刻 DiT 还在 CPU，free 虚高不可作判据
        return True
    free = _cuda_free_bytes(device)
    return free is None or free < _sampling_headroom_bytes(height, width)


KREA2_SPEC = ModelSpec(
    family_id="krea2",
    display_name="Krea 2",
    objective="rectified_flow",
    # 与 Anima 共用 Qwen-Image VAE / Wan2.1 latent 空间——引用同一实例（D6）。
    latent=WAN21_F8C16,
    text=TextSpec(
        strategy="cached_varlen",
        max_seq_len=512,
        fingerprint=KREA2_TEXT_FINGERPRINT,
    ),
    sampling=SamplingDefaults(
        # 白名单单源 studio/domain/common.py FAMILY_SAMPLING（刀 1 / R3）；
        # KREA2_SAMPLER 等常量仍归 sampling.py（生成侧 parity 共用），
        # 与单源数据的一致性由 tests/test_model_family_gating.py 锁死
        samplers=FAMILY_SAMPLING["krea2"]["samplers"],
        schedulers=FAMILY_SAMPLING["krea2"]["schedulers"],
        default_sampler=KREA2_SAMPLER,
        default_scheduler=KREA2_SCHEDULER,
        default_steps=KREA2_RAW_STEPS,
        default_cfg=KREA2_RAW_GUIDANCE,
        # Comfy parity 口径：固定 mu=1.15（ComfyUI ModelSamplingFlux 同款；
        # 注意本值是 **mu（exp 前）**，与 Anima ConstantShift 的直接因子语义
        # 不同——shift_policy 语义归 family 解释）。diffusers 的分辨率感知
        # 动态 mu 保留为 build_krea2_sigmas(dynamic_mu=True) 非默认路径。
        shift_policy=ConstantShift(shift=1.15),
    ),
    capabilities=FAMILY_CAPABILITIES["krea2"],
    lora=LoraOutputSpec(prefix="lora_unet", preset_name="krea2_full"),
    config_defaults=FAMILY_CONFIG_DEFAULTS["krea2"],
)


class Krea2Family:
    spec = KREA2_SPEC

    def load_dit(self, path, device, dtype, *,
                 attention_backend: str = "flash_attn", repo_root=None,
                 purpose: str = "train"):
        from training.families.krea2.loader import load_krea2_model

        if attention_backend != "none":
            logger.info(
                "Krea2 当前固定使用 PyTorch SDPA；忽略 attention_backend=%s",
                attention_backend,
            )
        return load_krea2_model(path, device, dtype, purpose=purpose)

    def load_vae(self, path, device, dtype, *, tiling: str = "auto"):
        from training.vae import load_vae

        return load_vae(path, device, dtype, None, tiling=tiling)

    def load_text(self, text_encoder_path, device, dtype, *,
                  t5_tokenizer_path: str = "", comfy_qwen: bool = False,
                  t5_fast: bool = False, purpose: str = "train",
                  cache_enabled: bool = True):
        if purpose == "generate":
            # Comfy parity（sd.py:258）：生成场景 TE 固定 fp16 存储 + fp32
            # compute（text_encoder_dtype 默认 fp16 + set_model_compute_dtype
            # fp32），忽略调用方 dtype、无旋钮——与 TE offload 同款固定行为。
            # 训练侧维持调用方 dtype（bf16）不动。
            return load_krea2_text_stack(
                text_encoder_path,
                device=device,
                dtype=torch.float16,
                compute_dtype=torch.float32,
                cache_enabled=cache_enabled,
            )
        return load_krea2_text_stack(
            text_encoder_path,
            device=device,
            dtype=dtype,
            cache_enabled=cache_enabled,
        )

    def prepare_text_cache(self, captions: Iterable[str],
                           extra_prompts: Iterable[str], *, cache_entries=(),
                           cache_root=None, text=None, device=None,
                           dtype=None) -> None:
        if text is None:
            raise ValueError("Krea2 prepare_text_cache 需要 Krea2TextStack")
        text.prepare_text_cache(
            captions,
            extra_prompts,
            cache_entries=cache_entries,
            cache_root=cache_root,
        )

    def encode_text_for_batch(self, text, dit, captions, device, dtype, *,
                              comfy_encoding: bool = True,
                              kv_trim: bool = True):
        return text.encode_text_for_batch(captions, device=device, dtype=dtype)

    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint: bool = False):
        return dit(
            noisy,
            t,
            cond.context,
            attention_mask=cond.attention_mask,
            use_checkpoint=use_checkpoint,
        )

    def sample_image(self, model, vae, text, prompt, *,
                     height: int = 1024, width: int = 1024,
                     steps: int | None = None,
                     cfg_scale: float | None = None,
                     negative_prompt: str = "",
                     sampler_name: str | None = None,
                     scheduler: str | None = None,
                     distilled: bool = False,
                     device="cuda", dtype=None, step_callback=None,
                     phase_callback=None, seed: int | None = None,
                     vram_policy: str | None = None):
        # 先按 Raw/Turbo 解析步数与 guidance（steps/cfg 未显式给时用族默认：
        # Raw 28 步 / 4.5，Turbo 8 步 / 0.0——TDM 蒸馏无 uncond，guidance=0
        # 时 prepare 不编码 negative、采样跳过 uncond forward）。
        resolved_steps, guidance = resolve_sampling_settings(
            distilled=distilled,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler,
        )
        # —— 显存编排（vram_policy=None 时维持旧行为，训练预览等调用面
        # 绝不动 model：优化器状态引用会被 .to() 破坏）。
        # 按需让位（comfy free_memory 语义）：编码需要把 TE 搬上 GPU 且
        # 装不下时，DiT 先撤 CPU；prompt 全部命中在线 LRU 则整体跳过。
        dit_yielded = False
        if vram_policy is not None:
            needed = [prompt] + ([negative_prompt] if guidance > 0 else [])
            cached = getattr(text, "online_conditions_cached", None)
            te_resident = bool(getattr(text, "is_model_on_device", False))
            need_te_move = not te_resident and not (
                callable(cached) and cached(needed)
            )
            if need_te_move and _should_yield_dit(vram_policy, device):
                logger.info("krea2 显存编排：编码前 DiT 让位到 CPU")
                model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                dit_yielded = True
        condition = prepare_sampling_condition(
            text,
            prompt,
            negative_prompt=negative_prompt,
            cfg_scale=guidance,
            device=device,
            dtype=dtype,
            phase_callback=phase_callback,
        )
        # 采样前 TE 处置：None=旧行为无条件卸（训练缓存模式 no-op）；
        # auto 只在采样余量不足时卸——显存充裕则同驻，零搬运。
        if vram_policy is None or _should_offload_te(
            vram_policy, device, height, width, dit_yielded,
        ):
            offload = getattr(text, "offload_model", None)
            if callable(offload):
                offload()
        if dit_yielded:
            model.to(torch.device(device))
        return sample_image(
            model,
            vae,
            condition,
            height=height,
            width=width,
            steps=resolved_steps,
            cfg_scale=guidance,
            sampler_name=sampler_name,
            scheduler=scheduler,
            distilled=distilled,
            device=device,
            dtype=dtype,
            step_callback=step_callback,
            phase_callback=phase_callback,
            seed=seed,
        )

    def lora_preset(self) -> dict[str, Any]:
        return KREA2_PRESET

    def lora_metadata(self) -> dict[str, str]:
        return {
            "model_family": self.spec.family_id,
            "preset": self.spec.lora.preset_name,
        }

    def convert_lora_state_dict(self, sd: dict) -> dict:
        return sd


__all__ = [
    "KREA2_SPEC",
    "Krea2Family",
    "Krea2SamplingCondition",
    "Krea2TextCondition",
    "Krea2TextStack",
    "load_krea2_text_stack",
    "prepare_sampling_condition",
    "sample_image",
]
