"""Krea2 ModelSpec and ModelFamily implementation (multi-model Phase 3)."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from training.families.krea2.preset import KREA2_PRESET
from training.families.krea2.sampling import (
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
from training.families.krea2.text_encoding import (
    KREA2_TEXT_FINGERPRINT,
    Krea2TextCondition,
    Krea2TextStack,
    load_krea2_text_stack,
)
from training.families.latent_spaces import WAN21_F8C16
from training.families.spec import (
    LoraOutputSpec,
    ModelSpec,
    ConstantShift,
    SamplingDefaults,
    TextSpec,
)


logger = logging.getLogger(__name__)


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
        samplers=(KREA2_SAMPLER,),
        schedulers=(KREA2_SCHEDULER,),
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
    capabilities=frozenset({"masked_loss", "text_cache"}),
    lora=LoraOutputSpec(prefix="lora_unet", preset_name="krea2_full"),
    config_defaults={
        "shuffle_caption": False,
        "keep_tokens": 0,
        "tag_dropout": 0.0,
        "text_encoder_cache": True,
        "attention_backend": "none",
        "timestep_sampling": "krea2_shift",
        "timestep_shift_resolution_aware": False,
        "sample_sampler_name": KREA2_SAMPLER,
        "sample_scheduler": KREA2_SCHEDULER,
        "sample_infer_steps": KREA2_RAW_STEPS,
        "sample_cfg_scale": KREA2_RAW_GUIDANCE,
    },
)


class Krea2Family:
    spec = KREA2_SPEC

    def load_dit(self, path, device, dtype, *,
                 attention_backend: str = "flash_attn", repo_root=None):
        from training.families.krea2.loader import load_krea2_model

        if attention_backend != "none":
            logger.info(
                "Krea2 当前固定使用 PyTorch SDPA；忽略 attention_backend=%s",
                attention_backend,
            )
        return load_krea2_model(path, device, dtype)

    def load_vae(self, path, device, dtype, *, tiling: str = "auto"):
        from training.vae import load_vae

        return load_vae(path, device, dtype, None, tiling=tiling)

    def load_text(self, text_encoder_path, device, dtype, *,
                  t5_tokenizer_path: str = "", comfy_qwen: bool = False,
                  t5_fast: bool = False, purpose: str = "train",
                  cache_enabled: bool = True):
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
                     phase_callback=None, seed: int | None = None):
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
        condition = prepare_sampling_condition(
            text,
            prompt,
            negative_prompt=negative_prompt,
            cfg_scale=guidance,
            device=device,
            dtype=dtype,
            phase_callback=phase_callback,
        )
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
