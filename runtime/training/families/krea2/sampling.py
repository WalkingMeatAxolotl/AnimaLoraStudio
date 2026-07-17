"""Krea2 FlowMatchEuler sampling with resolution-aware dynamic shifting.

The Euler loop, resolution alignment, and shifted timestep schedule are adapted
from kohya-ss/musubi-tuner's Krea2 sampler at fixed commit
8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6 (Apache-2.0).
Copyright 2026 Kohya S. and musubi-tuner contributors.
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2/krea2_sampling.py

The Krea guidance convention (``cond + g * (cond - uncond)``), Raw/TDM
defaults, and FlowMatchEuler scheduler behavior are cross-checked against
Hugging Face diffusers at commit bc529a5f677db9c4b3fc72c76962c4e2f61567e1
(Apache-2.0):
https://github.com/huggingface/diffusers/blob/bc529a5f677db9c4b3fc72c76962c4e2f61567e1/src/diffusers/pipelines/krea2/pipeline_krea2.py
https://github.com/huggingface/diffusers/blob/bc529a5f677db9c4b3fc72c76962c4e2f61567e1/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
"""

from __future__ import annotations

import contextlib
import gc
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from training.families.krea2.text_encoding import Krea2TextCondition


KREA2_SAMPLER = "euler"
KREA2_SCHEDULER = "krea2_shift"
KREA2_RAW_STEPS = 28
KREA2_RAW_GUIDANCE = 4.5
KREA2_TURBO_STEPS = 8
KREA2_TURBO_GUIDANCE = 0.0
# 推理 sigma 的固定 mu（Comfy parity：ModelSamplingFlux shift=1.15，Raw/Turbo 同）。
# 曾名 KREA2_DISTILLED_MU——统一口径后它不再是蒸馏专属。
KREA2_FIXED_MU = 1.15
KREA2_DISTILLED_MU = KREA2_FIXED_MU  # 兼容别名
KREA2_BASE_IMAGE_SEQ_LEN = 256
KREA2_MAX_IMAGE_SEQ_LEN = 6400
KREA2_BASE_SHIFT = 0.5
KREA2_MAX_SHIFT = 1.15


@dataclass(frozen=True)
class Krea2SamplingCondition:
    """Positive and optional negative Qwen3-VL conditions for one image."""

    positive: Krea2TextCondition
    negative: Krea2TextCondition | None = None


def _validate_guidance(value: float) -> float:
    guidance = float(value)
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Krea2 guidance scale 必须为有限非负数")
    return guidance


def resolve_sampling_settings(
    *,
    distilled: bool,
    steps: int | None,
    cfg_scale: float | None,
    sampler_name: str | None,
    scheduler: str | None,
) -> tuple[int, float]:
    """Resolve Raw/Turbo defaults and reject sampler names from another family."""

    sampler = KREA2_SAMPLER if sampler_name is None else str(sampler_name).lower().strip()
    schedule = KREA2_SCHEDULER if scheduler is None else str(scheduler).lower().strip()
    if sampler != KREA2_SAMPLER or schedule != KREA2_SCHEDULER:
        raise ValueError(
            "Krea2 仅支持 euler+krea2_shift，实际 "
            f"{sampler or '<empty>'}+{schedule or '<empty>'}"
        )

    default_steps = KREA2_TURBO_STEPS if distilled else KREA2_RAW_STEPS
    default_guidance = KREA2_TURBO_GUIDANCE if distilled else KREA2_RAW_GUIDANCE
    resolved_steps = default_steps if steps is None else int(steps)
    if resolved_steps <= 0:
        raise ValueError("Krea2 sampling steps 必须为正数")
    guidance = _validate_guidance(
        default_guidance if cfg_scale is None else cfg_scale,
    )
    return resolved_steps, guidance


def calculate_krea2_mu(
    image_seq_len: int,
    *,
    base_image_seq_len: int = KREA2_BASE_IMAGE_SEQ_LEN,
    max_image_seq_len: int = KREA2_MAX_IMAGE_SEQ_LEN,
    base_shift: float = KREA2_BASE_SHIFT,
    max_shift: float = KREA2_MAX_SHIFT,
) -> float:
    """Linearly interpolate Krea2 ``mu`` from image token count, without clamp."""

    seq_len = int(image_seq_len)
    if seq_len <= 0:
        raise ValueError("Krea2 image_seq_len 必须为正数")
    if max_image_seq_len <= base_image_seq_len:
        raise ValueError("Krea2 max_image_seq_len 必须大于 base_image_seq_len")
    values = (float(base_shift), float(max_shift))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Krea2 shift 端点必须为有限数")
    slope = (values[1] - values[0]) / (max_image_seq_len - base_image_seq_len)
    return slope * seq_len + (values[0] - slope * base_image_seq_len)


def build_krea2_sigmas(
    image_seq_len: int,
    steps: int,
    *,
    distilled: bool = False,
    dynamic_mu: bool = False,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Build the shifted ``1 → 0`` FlowMatchEuler sigma grid.

    默认固定 mu=1.15（Comfy parity 口径）：ComfyUI 把 Krea2 注册为
    ModelType.FLUX → ModelSamplingFlux(shift=1.15)，其 flux_time_shift(mu, 1, t)
    与本函数的 Möbius 形式恒等——Raw / Turbo 一律固定 mu，训练预览与 Generate
    两面共用同一口径（与 Anima 的 ConstantShift 先例同构）。

    ``dynamic_mu=True`` 保留 diffusers/musubi Raw pipeline 的分辨率感知插值
    （1024² 约等效 mu≈0.906），非默认，供对照实验。``distilled`` 不再影响
    sigma（只决定 resolve_sampling_settings 的步数 / guidance 默认）。
    """

    if int(steps) <= 0:
        raise ValueError("Krea2 sampling steps 必须为正数")
    del distilled  # sigma 口径统一后仅存于签名兼容；见 docstring
    mu = (
        calculate_krea2_mu(image_seq_len)
        if dynamic_mu
        else KREA2_FIXED_MU
    )
    base = torch.linspace(
        1.0, 0.0, int(steps) + 1, device=device, dtype=torch.float32,
    )
    shift = math.exp(mu)
    return (shift * base) / (1.0 + (shift - 1.0) * base)


def align_krea2_resolution(value: int, *, align: int = 16) -> int:
    """Round a requested pixel dimension up to the VAE-stride × DiT-patch grid."""

    dimension = int(value)
    if dimension <= 0 or int(align) <= 0:
        raise ValueError("Krea2 resolution 与 align 必须为正数")
    return ((dimension + int(align) - 1) // int(align)) * int(align)


def prepare_sampling_condition(
    text_stack,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    cfg_scale: float = KREA2_RAW_GUIDANCE,
    device: torch.device | str,
    dtype: torch.dtype,
    phase_callback=None,
) -> Krea2SamplingCondition:
    """Resolve cached/online text for sampling before entering the Euler loop."""

    guidance = _validate_guidance(cfg_scale)
    captions = [str(prompt)]
    if guidance > 0:
        captions.append("" if negative_prompt is None else str(negative_prompt))
    if phase_callback is not None:
        phase_callback("clip")
    encoded = text_stack.encode_text_for_batch(captions, device=device, dtype=dtype)
    positive = Krea2TextCondition(
        context=encoded.context[0:1],
        attention_mask=encoded.attention_mask[0:1],
    )
    negative = None
    if guidance > 0:
        negative = Krea2TextCondition(
            context=encoded.context[1:2],
            attention_mask=encoded.attention_mask[1:2],
        )
    return Krea2SamplingCondition(positive=positive, negative=negative)


def _move_condition(
    condition: Krea2TextCondition,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Krea2TextCondition:
    if condition.context.ndim != 4 or condition.attention_mask.ndim != 2:
        raise ValueError("Krea2 sampling condition 形状必须为 context 4D + mask 2D")
    if condition.context.shape[:2] != condition.attention_mask.shape:
        raise ValueError("Krea2 sampling context 与 attention mask 的 batch/seq 不一致")
    if condition.context.shape[0] != 1:
        raise ValueError("Krea2 sample_image 当前一次只生成一张图")
    return Krea2TextCondition(
        context=condition.context.to(device=device, dtype=dtype),
        attention_mask=condition.attention_mask.to(device=device, dtype=torch.bool),
    )


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "cpu"} and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def sample_latents(
    model,
    condition: Krea2SamplingCondition,
    *,
    height: int = 1024,
    width: int = 1024,
    steps: int | None = None,
    cfg_scale: float | None = None,
    sampler_name: str | None = None,
    scheduler: str | None = None,
    distilled: bool = False,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int | None = None,
    latents: Tensor | None = None,
    step_callback=None,
) -> Tensor:
    """Run Krea2's deterministic shifted FlowMatchEuler loop and return 5D latents."""

    steps, guidance = resolve_sampling_settings(
        distilled=distilled,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_name=sampler_name,
        scheduler=scheduler,
    )
    target = torch.device(device)
    patch = int(model.config.patch)
    channels = int(model.config.channels)
    align = 8 * patch
    aligned_height = align_krea2_resolution(height, align=align)
    aligned_width = align_krea2_resolution(width, align=align)
    latent_height = aligned_height // 8
    latent_width = aligned_width // 8
    image_seq_len = (latent_height // patch) * (latent_width // patch)

    positive = _move_condition(condition.positive, device=target, dtype=dtype)
    negative = None
    if guidance > 0:
        if condition.negative is None:
            raise ValueError("Krea2 guidance > 0 时必须提供 negative condition")
        negative = _move_condition(condition.negative, device=target, dtype=dtype)

    expected_shape = (1, channels, 1, latent_height, latent_width)
    if latents is None:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        latents = torch.randn(expected_shape, generator=generator, dtype=torch.float32)
    elif tuple(latents.shape) != expected_shape:
        raise ValueError(
            f"Krea2 初始 latent 形状应为 {expected_shape}，实际 {tuple(latents.shape)}"
        )
    image = latents.to(device=target, dtype=dtype)
    sigmas = build_krea2_sigmas(
        image_seq_len, steps, distilled=distilled, device=target,
    )

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with _autocast_context(target, dtype):
            for index, (sigma, next_sigma) in enumerate(zip(sigmas[:-1], sigmas[1:])):
                timestep = sigma.expand(image.shape[0]).to(device=target, dtype=image.dtype)
                velocity = model(
                    image,
                    timestep,
                    positive.context,
                    attention_mask=positive.attention_mask,
                )
                if guidance > 0:
                    uncond_velocity = model(
                        image,
                        timestep,
                        negative.context,
                        attention_mask=negative.attention_mask,
                    )
                    # Krea convention: g=0 is conditional-only; equivalent standard
                    # CFG scale is (1 + g).
                    velocity = velocity + guidance * (velocity - uncond_velocity)

                if step_callback is not None:
                    denoised = image - sigma.to(dtype=image.dtype) * velocity
                    try:
                        step_callback(index, steps, denoised)
                    except Exception:
                        pass
                image = image + (next_sigma - sigma).to(dtype=image.dtype) * velocity
    finally:
        model.train(was_training)
    return image


def _decode_to_pil(vae, latents: Tensor):
    import numpy as np
    from PIL import Image

    pixels = vae.decode(latents)
    if pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError(
            "Krea2 VAE decode 应返回单张 (1,3,T,H,W)，实际 "
            f"{tuple(pixels.shape)}"
        )
    pixels = (pixels[:, :, 0].clamp(-1, 1) + 1) / 2
    array = pixels[0].permute(1, 2, 0).cpu().float().numpy()
    return Image.fromarray((array * 255).clip(0, 255).astype(np.uint8))


@torch.no_grad()
def sample_image(
    model,
    vae,
    condition: Krea2SamplingCondition,
    *,
    height: int = 1024,
    width: int = 1024,
    steps: int | None = None,
    cfg_scale: float | None = None,
    sampler_name: str | None = None,
    scheduler: str | None = None,
    distilled: bool = False,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    step_callback=None,
    phase_callback=None,
    seed: int | None = None,
):
    """Generate one PIL image from pre-encoded Krea2 sampling conditions."""

    if phase_callback is not None:
        phase_callback("sample")
    latents = sample_latents(
        model,
        condition,
        height=height,
        width=width,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_name=sampler_name,
        scheduler=scheduler,
        distilled=distilled,
        device=device,
        dtype=dtype,
        seed=seed,
        step_callback=step_callback,
    )
    if phase_callback is not None:
        phase_callback("vae")
    image = _decode_to_pil(vae, latents)
    del latents
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return image
