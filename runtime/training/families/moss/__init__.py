"""MOSS-Image ModelSpec 与 ModelFamily（docs/design/multi-model/05-moss-image.md）。

结构 / 加载 / int8 运行时 / 文本栈分别在 ``modeling.moss``、``loader``、``quant_int8``、
``text_encoding``；这里只做声明与派发。
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch

# 能力矩阵 / 族默认 / 采样白名单单源在 studio/domain/common.py（同 anima/krea2）
from studio.domain.common import (
    FAMILY_CAPABILITIES,
    FAMILY_CONFIG_DEFAULTS,
    FAMILY_SAMPLING,
)
from studio.infrastructure.log_messages import msg

from ..latent_spaces import FLUX2_F8C32_BN
from ..spec import ConstantShift, LoraOutputSpec, ModelSpec, SamplingDefaults, TextSpec
from .forward import MossTextCondition, forward_train
from .preset import MOSS_PRESET
from .sampling import (
    MOSS_DEFAULT_CFG,
    MOSS_DEFAULT_STEPS,
    MOSS_SHIFT,
    MOSS_UNCOND_SKIP_LAYERS,
    MossSamplingCondition,
    decode_to_pil,
    prepare_sampling_condition,
    sample_latents,
)
from .text_encoding import MOSS_MAX_TOKENS, MOSS_TEXT_FINGERPRINT, MossTextStack

logger = logging.getLogger(__name__)


MOSS_SPEC = ModelSpec(
    family_id="moss",
    display_name="MOSS-Image",
    objective="rectified_flow",
    latent=FLUX2_F8C32_BN,
    text=TextSpec(
        strategy="cached_varlen",
        max_seq_len=MOSS_MAX_TOKENS,
        fingerprint=MOSS_TEXT_FINGERPRINT,
    ),
    sampling=SamplingDefaults(
        samplers=FAMILY_SAMPLING["moss"]["samplers"],
        schedulers=FAMILY_SAMPLING["moss"]["schedulers"],
        default_sampler=FAMILY_SAMPLING["moss"]["samplers"][0],
        default_scheduler=FAMILY_SAMPLING["moss"]["schedulers"][0],
        default_steps=MOSS_DEFAULT_STEPS,
        default_cfg=MOSS_DEFAULT_CFG,
        # ComfyUI supported_models.MOSSImage：ModelSamplingDiscreteFlow shift=5.0
        shift_policy=ConstantShift(shift=MOSS_SHIFT),
    ),
    capabilities=FAMILY_CAPABILITIES["moss"],
    lora=LoraOutputSpec(prefix="lora_unet", preset_name="moss_full"),
    config_defaults=FAMILY_CONFIG_DEFAULTS["moss"],
)


class MossFamily:
    spec = MOSS_SPEC

    def load_dit(self, path, device, dtype, *,
                 attention_backend: str = "flash_attn", repo_root=None,
                 purpose: str = "train", blocks_to_swap: int = 0):
        del repo_root, purpose
        from training.families.moss.loader import load_moss_model

        if blocks_to_swap:
            # 能力位里没有 block_swap（PR-5 接入），phases 已 fail-fast；这里兜底
            raise RuntimeError("MOSS 暂不支持 block swap，请将 blocks_to_swap 置 0")
        if attention_backend != "none":
            logger.info(msg("train.moss_sdpa_only", backend=attention_backend))
        return load_moss_model(path, device, dtype)

    def load_vae(self, path, device, dtype, *, tiling: str = "auto"):
        from training.vae_flux2 import load_flux2_vae

        return load_flux2_vae(path, device, dtype, tiling=tiling)

    def load_text(self, text_encoder_path, device, dtype, *,
                  t5_tokenizer_path: str = "", comfy_qwen: bool = False,
                  t5_fast: bool = False, purpose: str = "train",
                  cache_enabled: bool = True) -> MossTextStack:
        del t5_tokenizer_path, comfy_qwen, t5_fast, purpose
        return MossTextStack(text_encoder_path, device=device, dtype=dtype,
                             cache_enabled=cache_enabled)

    def prepare_text_cache(self, captions: Iterable[str],
                           extra_prompts: Iterable[str], *, cache_entries=(),
                           cache_root=None, text=None, device=None,
                           dtype=None) -> None:
        if text is None:
            raise ValueError("MOSS prepare_text_cache 需要 MossTextStack")
        text.prepare_text_cache(captions, extra_prompts,
                                cache_entries=cache_entries, cache_root=cache_root)

    def encode_text_for_batch(self, text, dit, captions, device, dtype, *,
                              comfy_encoding: bool = True,
                              kv_trim: bool = True) -> MossTextCondition:
        return text.encode_text_for_batch(captions, device=device, dtype=dtype)

    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint: bool = False):
        return forward_train(dit, noisy, t, cond, use_checkpoint=use_checkpoint)

    def sample_image(self, model, vae, text, prompt, *,
                     height: int = 1024, width: int = 1024,
                     steps: int | None = None,
                     cfg_scale: float | None = None,
                     negative_prompt: str = "",
                     sampler_name: str | None = None,
                     scheduler: str | None = None,
                     uncond_skip_layers=MOSS_UNCOND_SKIP_LAYERS,
                     uncond_skip_start_percent: float = 0.0,
                     uncond_skip_end_percent: float = 1.0,
                     device="cuda", dtype=None, step_callback=None,
                     phase_callback=None, seed: int | None = None,
                     vram_policy: str | None = None):
        del vram_policy  # 显存编排随 PR-5
        dtype = dtype or torch.bfloat16
        steps = int(steps or MOSS_DEFAULT_STEPS)
        cfg = MOSS_DEFAULT_CFG if cfg_scale is None else float(cfg_scale)
        condition: MossSamplingCondition = prepare_sampling_condition(
            text, prompt, negative_prompt=negative_prompt or "", cfg_scale=cfg,
            device=device, dtype=dtype, phase_callback=phase_callback,
        )
        offload = getattr(text, "offload_model", None)
        if callable(offload):
            offload()  # 训练缓存模式下 TE 本就已释放，no-op
        if phase_callback:
            phase_callback("sample")
        latents = sample_latents(
            model, condition, height=height, width=width, steps=steps, cfg_scale=cfg,
            sampler_name=sampler_name or self.spec.sampling.default_sampler,
            scheduler=scheduler or self.spec.sampling.default_scheduler,
            uncond_skip_layers=uncond_skip_layers,
            uncond_skip_start_percent=uncond_skip_start_percent,
            uncond_skip_end_percent=uncond_skip_end_percent,
            device=device, dtype=dtype, seed=seed, step_callback=step_callback,
        )
        if phase_callback:
            phase_callback("vae")
        return decode_to_pil(vae, latents)

    def lora_preset(self) -> dict[str, Any]:
        return MOSS_PRESET

    def lora_metadata(self) -> dict[str, str]:
        return {"model_family": self.spec.family_id, "preset": self.spec.lora.preset_name}

    def convert_lora_state_dict(self, sd: dict) -> dict:
        return sd


__all__ = ["MOSS_SPEC", "MossFamily", "MossTextCondition", "MossTextStack"]
