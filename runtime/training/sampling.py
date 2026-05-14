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
import torch
import torch.nn.functional as F

from training.text_encoding import (
    _build_qwen_text_from_prompt,
    encode_qwen,
    tokenize_t5_weighted,
)


logger = logging.getLogger(__name__)


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

    返回：sigmas (steps+1,) float32，从高到低，末尾带 0.0
    """
    ts = torch.arange(1, timesteps + 1, device=device, dtype=torch.float32) / float(timesteps)  # (0, 1]
    sigmas_full = _time_snr_shift(float(shift), ts)  # (0, 1]

    ss = len(sigmas_full) / float(steps)
    sigmas = [float(sigmas_full[-(1 + int(i * ss))]) for i in range(steps)]
    sigmas.append(0.0)
    sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)

    # ComfyUI offset_first_sigma_for_snr: CONST 下避免 sigma=1 导致 logit inf
    if sigmas.numel() > 0 and sigmas[0] >= 1.0:
        sigmas[0] = float(_time_snr_shift(float(shift), torch.tensor(1.0 - 1e-4, device=device, dtype=torch.float32)))
    return sigmas


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
):
    """训练时采样预览（尽量对齐 ComfyUI KSampler）

    Args:
        negative_prompt: 负面提示词，默认使用标准负面提示词
        sampler_name: 采样器（推荐：er_sde）
        scheduler: 调度器（推荐：simple）
    """
    import numpy as np
    from PIL import Image
    model.eval()

    logger.info(f"[Debug] Sampling start. Prompt: {prompt[:50]}...")

    # Check VAE scale
    if isinstance(vae.scale, list) and len(vae.scale) == 2:
        m, s = vae.scale
        logger.info(f"[Debug] VAE scale: mean_shape={m.shape}, std_inv_shape={s.shape}")
        logger.info(f"[Debug] VAE scale values: mean={m.mean().item():.4f}, std_inv={s.mean().item():.4f}")

    # 默认负面提示词 (参考 Anima Prompt Guide)
    if negative_prompt is None:
        negative_prompt = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, bad hands, bad feet, missing fingers, extra fingers, text, watermark, logo, signature, username, artist name, copyright name"

    # 文本编码
    try:
        # 有条件 (positive prompt)
        qwen_text = _build_qwen_text_from_prompt(prompt)
        qwen_embeds, qwen_attn = encode_qwen(qwen_model, qwen_tokenizer, [qwen_text], device)
        logger.info(f"[Debug] Qwen embeds: {qwen_embeds.shape}, mean={qwen_embeds.mean().item():.4f}")

        t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tokenizer, [prompt], max_length=512)
        t5_ids = t5_ids.to(device)
        t5_attn = t5_attn.to(device)
        t5_w = t5_w.to(device, dtype=torch.float32)
        cross_cond = model.preprocess_text_embeds(qwen_embeds, t5_ids)
        if cross_cond.shape[1] < 512:
            cross_cond = F.pad(cross_cond, (0, 0, 0, 512 - cross_cond.shape[1]))

        # 无条件/负面提示词 (negative prompt)
        qwen_text_uncond = _build_qwen_text_from_prompt(negative_prompt)
        qwen_embeds_uncond, qwen_attn_uncond = encode_qwen(qwen_model, qwen_tokenizer, [qwen_text_uncond], device)
        t5_ids_uncond, t5_attn_uncond, t5_w_uncond = tokenize_t5_weighted(t5_tokenizer, [negative_prompt], max_length=512)
        t5_ids_uncond = t5_ids_uncond.to(device)
        t5_attn_uncond = t5_attn_uncond.to(device)
        t5_w_uncond = t5_w_uncond.to(device, dtype=torch.float32)
        cross_uncond = model.preprocess_text_embeds(qwen_embeds_uncond, t5_ids_uncond)
        if cross_uncond.shape[1] < 512:
            cross_uncond = F.pad(cross_uncond, (0, 0, 0, 512 - cross_uncond.shape[1]))

    except Exception as e:
        logger.error(f"[Debug] Encoding failed: {e}")
        raise e

    # sigmas（对齐 ComfyUI supported_models.Anima: shift=3.0, multiplier=1.0）
    lat_h, lat_w = height // 8, width // 8
    if str(scheduler).lower() != "simple":
        logger.warning(f"采样 scheduler={scheduler} 未实现，回退 simple")
    sigmas = _flow_sigmas_simple(steps, shift=3.0, device=device)

    # 初始化噪声（ComfyUI CONST.noise_scaling: x = sigma*noise + (1-sigma)*latent_image；txt2img latent_image=0）
    x = torch.randn(1, 16, 1, lat_h, lat_w, device=device, dtype=torch.float32) * float(sigmas[0])
    logger.info(f"[Debug] Latents init: {x.shape}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

    pad_mask = torch.zeros(1, 1, lat_h, lat_w, device=device, dtype=dtype)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def denoise_fn(x_in: torch.Tensor, sigma_in: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(sigma_in):
            sigma_in = torch.tensor(float(sigma_in), device=x_in.device, dtype=torch.float32)
        sigma_b = sigma_in.view(1, 1).to(device=x_in.device, dtype=dtype)
        sigma_5d = sigma_in.view(1, 1, 1, 1, 1).to(device=x_in.device, dtype=torch.float32)

        with torch.autocast(device_type=device_type, dtype=dtype):
            v_cond = model(x_in.to(device=x_in.device, dtype=dtype), sigma_b, cross_cond, padding_mask=pad_mask)
            v_uncond = model(x_in.to(device=x_in.device, dtype=dtype), sigma_b, cross_uncond, padding_mask=pad_mask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

        if torch.isnan(v).any():
            raise RuntimeError("v contains NaN during sampling")

        # CONST(flow): denoised x0 = x - sigma * v
        return x_in - sigma_5d * v.float()

    sampler_name_l = str(sampler_name).lower().strip()
    logger.info(f"[Debug] Sampler={sampler_name_l}, Scheduler=simple, steps={steps}, cfg={cfg_scale}")

    # PR-C：通过 inference_samplers plugin registry 派发；未注册名走下面 inline Euler 兜底
    from training.inference_samplers import build_inference_sampler
    sampler_fn = build_inference_sampler(sampler_name_l)
    if sampler_fn is not None:
        x = sampler_fn(
            denoise_fn, x, sigmas,
            seed=None, s_noise=1.0, max_stage=3,
            step_callback=step_callback,
        )
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
    try:
        images = vae.model.decode(latents, vae.scale)
        images = images.squeeze(2)  # [B,C,H,W]
        images = (images.clamp(-1, 1) + 1) / 2

        # 转 PIL
        img = images[0].permute(1, 2, 0).cpu().float().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)

        model.train()
        return Image.fromarray(img)
    except Exception as e:
        logger.error(f"[Debug] VAE decode failed: {e}")
        raise e
