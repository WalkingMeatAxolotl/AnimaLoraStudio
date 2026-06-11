"""推理采样：sigma 调度 + ER-SDE solver + sample_image（训练/生成共用）。

抽自原 runtime/anima_train.py L822-961 + L1677-1815（ADR 0003 PR-A）。

公开：
- sample_image — 训练时采样预览 + 生成 CLI 共用入口（被 sister script 调）

内部：
- _time_snr_shift / _flow_sigmas_simple — ComfyUI ModelSamplingDiscreteFlow 对齐
- _default_noise_sampler / _sample_er_sde_const_x0 — ER-SDE-Solver-3 在 CONST flow 下的实现

注：sample_t / make_noise / compute_loss_weight 是 *训练 step* 用的采样工具，
不在本模块——见 training.timestep_sampling / training.noise / training.loss_weighting。
"""

from __future__ import annotations

import logging
import sys
import torch
import torch.nn.functional as F

from training.text_encoding import (
    _build_qwen_text_from_prompt,
    apply_t5_token_weights,
    build_comfy_anima_conditioning_inputs,
    encode_qwen,
    tokenize_t5_weighted,
)


logger = logging.getLogger(__name__)


_COMFY_PARITY_SAMPLERS = {"dpmpp_3m_sde", "er_sde"}
_COMFY_PARITY_SCHEDULERS = {"sgm_uniform", "simple"}
DEFAULT_NEGATIVE_PROMPT = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, bad hands, bad feet, missing fingers, extra fingers, text, watermark, logo, signature, username, artist name, copyright name"


def _resolve_parity_sampler_scheduler(sampler_name: str, scheduler: str) -> tuple[str, str]:
    sampler = str(sampler_name).lower().strip()
    sched = str(scheduler).lower().strip()
    if sampler not in _COMFY_PARITY_SAMPLERS or sched not in _COMFY_PARITY_SCHEDULERS:
        raise ValueError(f"unsupported Comfy parity sampler/scheduler: {sampler}+{sched}")
    return sampler, sched


def _normalize_negative_prompt(negative_prompt: str | None, *, comfy_parity: bool) -> str:
    if negative_prompt is None:
        return "" if comfy_parity else DEFAULT_NEGATIVE_PROMPT
    return str(negative_prompt)


def _set_model_xformers_enabled(model, enabled: bool) -> bool:
    """Toggle model-module xformers switches. Returns True if it had been enabled."""
    module_names = {
        cls.__module__
        for cls in type(model).__mro__
        if getattr(cls, "__module__", None)
    }
    module_names.update({
        "models.cosmos_predict2_modeling",
        "cosmos_predict2_modeling",
    })

    was_enabled = False
    for module_name in sorted(module_names):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        was_enabled = bool(getattr(module, "_USE_XFORMERS", False)) or was_enabled
        fn = getattr(module, "set_xformers_enabled", None)
        if fn is not None:
            fn(enabled)
    return was_enabled


def _module_device(module) -> torch.device | None:
    if module is None or not hasattr(module, "parameters"):
        return None
    try:
        param = next(module.parameters())
    except StopIteration:
        return None
    except Exception:
        return None
    return param.device


def _offload_modules_for_vae_decode(*modules) -> list[tuple[object, torch.device]]:
    """Move large, inactive modules off GPU while fp32 VAE decode runs."""
    offloaded: list[tuple[object, torch.device]] = []
    for module in modules:
        device = _module_device(module)
        if device is None or device.type != "cuda" or not hasattr(module, "to"):
            continue
        module.to("cpu")
        offloaded.append((module, device))
    if offloaded and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return offloaded


def _restore_offloaded_modules(offloaded: list[tuple[object, torch.device]]) -> None:
    for module, device in offloaded:
        module.to(device)
    if offloaded and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_modules_to_device(device: str | torch.device, *modules) -> None:
    target = torch.device(device)
    moved = False
    for module in modules:
        current = _module_device(module)
        if current is None or current == target or not hasattr(module, "to"):
            continue
        module.to(target)
        moved = True
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _decode_vae(vae, latents: torch.Tensor) -> torch.Tensor:
    if hasattr(vae, "decode"):
        return vae.decode(latents)
    return vae.model.decode(latents, vae.scale)


def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    """ComfyUI ModelSamplingDiscreteFlow.time_snr_shift"""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def _flow_sigmas_simple(steps: int, *, shift: float = 3.0, timesteps: int = 1000, device: str = "cpu") -> torch.Tensor:
    """
    复刻 ComfyUI:
    - supported_models.Anima 的 sampling_settings: shift=3.0, multiplier=1.0
    - ModelSamplingDiscreteFlow + simple_scheduler(model_sampling, steps)

    返回：sigmas (steps+1,) float32，从高到低，末尾带 0.0。
    注意：ComfyUI 的 simple_scheduler 原样返回首项 1.0；KSampler 在进入
    具体 sampler 后才做 offset_first_sigma_for_snr。不要在 scheduler 层提前
    offset，否则 txt2img 初始 noise_scaling 会和 ComfyUI 不同。
    """
    ts = torch.arange(1, timesteps + 1, device=device, dtype=torch.float32) / float(timesteps)  # (0, 1]
    sigmas_full = _time_snr_shift(float(shift), ts)  # (0, 1]

    ss = len(sigmas_full) / float(steps)
    sigmas = [float(sigmas_full[-(1 + int(i * ss))]) for i in range(steps)]
    sigmas.append(0.0)
    sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
    return sigmas


def _flow_sigmas_sgm_uniform(steps: int, *, shift: float = 3.0, timesteps: int = 1000, multiplier: int = 1000, device: str = "cpu") -> torch.Tensor:
    """SGM uniform scheduler —— 逐行对齐 ComfyUI normal_scheduler(sgm=True)。

    ModelSamplingDiscreteFlow 语义：
      sigma_max/min = sigma 表两端 = time_snr_shift(shift, {1, 1/timesteps})
      timestep(σ)   = σ * multiplier
      sigma(ts)     = time_snr_shift(shift, ts / multiplier)
    sgm 分支：linspace(timestep(σ_max), timestep(σ_min), steps+1)[:-1] 再各自
    sigma() 回 σ，末尾 append 0。注意这里对 σ_max/σ_min 做了「二次 shift」——
    σ 已经是 shift 后的值，timestep 不反 shift，sigma() 又 shift 一次。这是
    ComfyUI 的既有行为，刻意复刻以对齐出图。
    """
    # sigma 表两端（已 shift）
    sigma_max = float(_time_snr_shift(float(shift), torch.tensor(1.0)))
    sigma_min = float(_time_snr_shift(float(shift), torch.tensor(1.0 / timesteps)))
    start = sigma_max * multiplier  # timestep(sigma_max)
    end = sigma_min * multiplier    # timestep(sigma_min)
    tl = torch.linspace(start, end, steps + 1, device=device, dtype=torch.float32)[:-1]
    sigmas = _time_snr_shift(float(shift), tl / multiplier)  # sigma(ts)
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
    return sigmas


def _prepare_comfy_t2i_noise(
    shape: tuple[int, ...],
    sigmas: torch.Tensor,
    *,
    device: str | torch.device,
    seed: int | None,
) -> torch.Tensor:
    """ComfyUI txt2img initial noise for CONST flow.

    ComfyUI KSampler creates initial noise on CPU with a CPU generator seeded by
    the requested seed, then model_sampling.noise_scaling() moves the sample into
    the sampler path. For CONST + empty txt2img latent, noise_scaling is simply
    `sigma * noise`.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

    noise = torch.randn(
        shape,
        dtype=torch.float32,
        layout=torch.strided,
        device="cpu",
        generator=generator,
    )
    sigma0 = float(sigmas[0].detach().cpu()) if sigmas.numel() > 0 else 0.0
    return (noise * sigma0).to(device=device, dtype=torch.float32)


def _fix_comfy_empty_latent_channels(
    latent_image: torch.Tensor,
    *,
    latent_channels: int,
    latent_dimensions: int,
) -> torch.Tensor:
    """Mirror ComfyUI `fix_empty_latent_channels` for txt2img empty latents.

    ResolutionMaster commonly emits a 4-channel empty latent. ComfyUI KSampler
    repeats all-zero empty latents to the model latent channel count, then adds
    the temporal dimension for video/Anima-style latent formats.
    """
    if torch.count_nonzero(latent_image) == 0 and latent_image.shape[1] != latent_channels:
        repeats = (latent_channels + latent_image.shape[1] - 1) // latent_image.shape[1]
        latent_image = latent_image.repeat(1, repeats, *([1] * (latent_image.ndim - 2)))
        latent_image = latent_image.narrow(1, 0, latent_channels)

    if latent_dimensions == 3 and latent_image.ndim == 4:
        latent_image = latent_image.unsqueeze(2)
    return latent_image


def _prepare_comfy_ksampler_txt2img_latent(
    height: int,
    width: int,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build the same empty latent shape path as the target Comfy workflow."""
    latent = torch.zeros(
        (1, 4, height // 8, width // 8),
        device=device,
        dtype=torch.float32,
    )
    return _fix_comfy_empty_latent_channels(
        latent,
        latent_channels=16,
        latent_dimensions=3,
    )


# ER-SDE-Solver 实现 + _default_noise_sampler 已搬到 training.inference_samplers.er_sde
# （ADR 0003 PR-C plugin registry）。sample_image 通过 build_inference_sampler 派发。


@torch.no_grad()
def sample_image(
    model, vae, qwen_model, qwen_tokenizer, t5_tokenizer,
    prompt, height=1024, width=1024, steps=25, cfg_scale=4.0,
    negative_prompt=None,
    sampler_name: str = "er_sde",
    scheduler: str = "simple",
    device="cuda",
    dtype=torch.bfloat16,
    step_callback=None,
    seed: int | None = None,
    comfy_parity: bool = False,
):
    """训练时采样预览（尽量对齐 ComfyUI KSampler）

    Args:
        negative_prompt: 负面提示词，默认使用标准负面提示词
        sampler_name: 采样器（推荐：er_sde）
        scheduler: 调度器（推荐：simple）
    """
    import numpy as np
    from PIL import Image

    if comfy_parity and str(device).startswith("cuda"):
        _move_modules_to_device(device, model, qwen_model)

    model.eval()

    if comfy_parity:
        sampler_name, scheduler = _resolve_parity_sampler_scheduler(sampler_name, scheduler)

    logger.info(f"[Debug] Sampling start. Prompt: {prompt[:50]}...")

    # Check VAE scale
    if isinstance(vae.scale, list) and len(vae.scale) == 2:
        m, s = vae.scale
        logger.info(f"[Debug] VAE scale: mean_shape={m.shape}, std_inv_shape={s.shape}")
        logger.info(f"[Debug] VAE scale values: mean={m.mean().item():.4f}, std_inv={s.mean().item():.4f}")

    # 默认负面提示词（参考 Anima Prompt Guide）；Generate 的 Comfy parity
    # 路径会保留空负面提示词以对齐 ComfyUI。
    negative_prompt = _normalize_negative_prompt(negative_prompt, comfy_parity=comfy_parity)

    # 文本编码
    try:
        def build_cross(prompt_text: str):
            if comfy_parity:
                qwen_text, t5_ids, t5_attn, t5_w = build_comfy_anima_conditioning_inputs(
                    t5_tokenizer,
                    prompt_text,
                    max_length=512,
                )
                qwen_embeds, qwen_attn = encode_qwen(
                    qwen_model,
                    qwen_tokenizer,
                    [qwen_text],
                    device,
                    preserve_empty_text=True,
                )
                logger.info(f"[Debug] Qwen embeds: {qwen_embeds.shape}, mean={qwen_embeds.mean().item():.4f}")
                qwen_embeds = qwen_embeds.to(device=device, dtype=dtype)
                t5_ids = t5_ids.to(device)
                t5_attn = t5_attn.to(device)
                t5_w = t5_w.to(device, dtype=dtype)
                cross = model.preprocess_text_embeds(qwen_embeds, t5_ids, t5xxl_weights=t5_w)
            else:
                qwen_text = _build_qwen_text_from_prompt(prompt_text)
                qwen_embeds, qwen_attn = encode_qwen(qwen_model, qwen_tokenizer, [qwen_text], device)
                logger.info(f"[Debug] Qwen embeds: {qwen_embeds.shape}, mean={qwen_embeds.mean().item():.4f}")
                t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tokenizer, [prompt_text], max_length=512)
                t5_ids = t5_ids.to(device)
                t5_attn = t5_attn.to(device)
                t5_w = t5_w.to(device, dtype=torch.float32)
                cross = model.preprocess_text_embeds(qwen_embeds, t5_ids)
                cross = apply_t5_token_weights(cross, t5_w)
            if cross.shape[1] < 512:
                cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))
            return cross

        # 有条件 (positive prompt)
        cross_cond = build_cross(prompt)

        # 无条件/负面提示词 (negative prompt)
        cross_uncond = build_cross(negative_prompt)

    except Exception as e:
        logger.error(f"[Debug] Encoding failed: {e}")
        raise e

    # sigmas（对齐 ComfyUI supported_models.Anima: shift=3.0, multiplier=1.0）
    lat_h, lat_w = height // 8, width // 8
    _scheduler_builders = {
        "simple": _flow_sigmas_simple,
        "sgm_uniform": _flow_sigmas_sgm_uniform,
    }
    sched_fn = _scheduler_builders.get(str(scheduler).lower().strip())
    if sched_fn is None:
        if comfy_parity:
            raise ValueError(
                f"unsupported Comfy parity sampler/scheduler: "
                f"{str(sampler_name).lower().strip()}+{str(scheduler).lower().strip()}"
            )
        logger.warning(f"采样 scheduler={scheduler} 未注册，回退 simple")
        sched_fn = _flow_sigmas_simple
    sigmas = sched_fn(steps, shift=3.0, device=device)

    # 初始化噪声（ComfyUI CONST.noise_scaling: x = sigma*noise + (1-sigma)*latent_image；txt2img latent_image=0）
    empty_latent = _prepare_comfy_ksampler_txt2img_latent(height, width, device="cpu")
    x = _prepare_comfy_t2i_noise(tuple(empty_latent.shape), sigmas, device=device, seed=seed)
    logger.info(f"[Debug] Latents init: {x.shape}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

    pad_mask = torch.zeros(1, 1, lat_h, lat_w, device=device, dtype=dtype)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def denoise_fn(x_in: torch.Tensor, sigma_in: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(sigma_in):
            sigma_in = torch.tensor(float(sigma_in), device=x_in.device, dtype=torch.float32)
        sigma_5d = sigma_in.view(1, 1, 1, 1, 1).to(device=x_in.device, dtype=torch.float32)

        def _run_model_forward() -> torch.Tensor:
            with torch.autocast(device_type=device_type, dtype=dtype):
                if comfy_parity:
                    # ComfyUI's CFGGuider batches negative/positive conds through one
                    # model forward in the common txt2img path, then chunks outputs.
                    # It also passes ModelSamplingDiscreteFlow.timestep(sigma) as
                    # float32. For Anima multiplier=1.0, timestep == sigma.
                    x_model = x_in.to(device=x_in.device, dtype=dtype)
                    sigma_1d = sigma_in.reshape(1).to(device=x_in.device, dtype=torch.float32)
                    if float(cfg_scale) == 1.0:
                        return model(
                            x_model,
                            sigma_1d.expand(x_model.shape[0]),
                            cross_cond,
                            padding_mask=pad_mask.expand(x_model.shape[0], -1, -1, -1).contiguous(),
                        )
                    x_batch = torch.cat([x_model, x_model], dim=0)
                    cross_batch = torch.cat([cross_uncond, cross_cond], dim=0)
                    pad_batch = pad_mask.expand(x_batch.shape[0], -1, -1, -1).contiguous()
                    sigma_batch = sigma_1d.expand(x_batch.shape[0])
                    v_uncond, v_cond = model(
                        x_batch,
                        sigma_batch,
                        cross_batch,
                        padding_mask=pad_batch,
                    ).chunk(2)
                    return v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    sigma_b = sigma_in.view(1, 1).to(device=x_in.device, dtype=dtype)
                    v_cond = model(x_in.to(device=x_in.device, dtype=dtype), sigma_b, cross_cond, padding_mask=pad_mask)
                    v_uncond = model(x_in.to(device=x_in.device, dtype=dtype), sigma_b, cross_uncond, padding_mask=pad_mask)
                    return v_uncond + cfg_scale * (v_cond - v_uncond)

        v = _run_model_forward()

        if torch.isnan(v).any():
            if comfy_parity and _set_model_xformers_enabled(model, False):
                logger.warning("xformers attention produced NaN; retrying denoise with SDPA fallback")
                v = _run_model_forward()
            if torch.isnan(v).any():
                raise RuntimeError("v contains NaN during sampling")

        # CONST(flow): denoised x0 = x - sigma * v
        return x_in - sigma_5d * v.float()

    sampler_name_l = str(sampler_name).lower().strip()
    logger.info(f"[Debug] Sampler={sampler_name_l}, Scheduler={scheduler}, steps={steps}, cfg={cfg_scale}")

    # PR-C：通过 inference_samplers plugin registry 派发；未注册名走下面 inline Euler 兜底
    from training.inference_samplers import build_inference_sampler
    sampler_fn = build_inference_sampler(sampler_name_l)
    if sampler_fn is None and comfy_parity:
        raise ValueError(
            f"unsupported Comfy parity sampler/scheduler: "
            f"{sampler_name_l}+{str(scheduler).lower().strip()}"
        )
    if sampler_fn is not None:
        sampler_kwargs = {
            "seed": seed,
            "s_noise": 1.0,
            "max_stage": 3,
            "step_callback": step_callback,
        }
        if comfy_parity and sampler_name_l == "dpmpp_3m_sde":
            sampler_kwargs["require_brownian_tree"] = True
        x = sampler_fn(denoise_fn, x, sigmas, **sampler_kwargs)
    else:
        # fallback: 简化 Euler ODE（deterministic），与 flow 兼容
        total = len(sigmas) - 1
        for i in range(total):
            sigma = float(sigmas[i])
            sigma_next = float(sigmas[i + 1])
            denoised = denoise_fn(x, sigmas[i])
            if step_callback is not None:
                try:
                    step_callback(i, total, denoised)
                except Exception:
                    pass
            d = (x - denoised) / max(sigma, 1e-6)
            x = x + d * (sigma_next - sigma)

    # VAE 解码
    latents = x.to(device=device, dtype=dtype)
    logger.info(f"[Debug] Final latents: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
    del denoise_fn, x, cross_cond, cross_uncond, pad_mask, sigmas, empty_latent
    offloaded_modules: list[tuple[object, torch.device]] = []
    try:
        # VAEWrapper.decode：整图 OOM 时自动 tile+cosine blend 回退（issue #200）。
        # 大显存路径在 try 一次就过，零成本；fallback 路径每张图慢 ~30%。
        if comfy_parity and device_type == "cuda":
            logger.info("[Debug] VAE decode: offloading inactive modules before fp32 decode")
            offloaded_modules = _offload_modules_for_vae_decode(model, qwen_model)
        images = _decode_vae(vae, latents)
        images = images.squeeze(2)  # [B,C,H,W]
        images = (images.clamp(-1, 1) + 1) / 2

        # 转 PIL
        img = images[0].permute(1, 2, 0).cpu().float().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img)

        del images, latents
        if comfy_parity and device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"[Debug] VAE decode failed: {e}")
        raise
    finally:
        if offloaded_modules:
            logger.info("[Debug] VAE decode: restoring offloaded modules after cleanup")
            _restore_offloaded_modules(offloaded_modules)

    model.train()
    return pil_image
