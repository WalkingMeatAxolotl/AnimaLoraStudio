"""MOSS-Image 采样（ComfyUI parity，MOSS 设计稿 §5.6）。

对齐 ComfyUI 原生 MOSS 支持（``supported_models.MOSSImage``）+ KSampler：

- ``ModelSamplingDiscreteFlow``：shift 5.0、multiplier 1.0（DiT 内部 timestep ×1000）
- ``simple`` 调度：1000 档 σ 表 ``shift·t/(1+(shift−1)t)`` 按步数等距取
- 初始噪声：EmptyLatentImage 经 ``fix_empty_latent_channels`` 变成 Flux2 latent
  （128ch、1/16），CPU generator 按 seed 生成；``er_sde`` 每步噪声同样按这个形状生成
  ——所以整个采样循环在 Comfy 的 128ch 排布里跑，DiT 前后各一次纯 reshape
  （本项目缓存 / DiT 吃的是 unpatchify 后的 32ch f8 排布，见 latent_spaces）
- CFG：``uncond + cfg·(cond − uncond)``；uncond 跳层语义对齐 Comfy
  ``SkipLayerGuidanceDiTSimple``：σ ∈ [percent_to_sigma(end), percent_to_sigma(start)]
  时 uncond 前向跳过指定 block，其余步正常
- 默认值取官方推荐：euler / simple / 50 步 / CFG 3 / uncond 跳第 22 层（全程）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from modeling.moss.moss_modeling import patchify, unpatchify
from studio.domain.common import FAMILY_SAMPLING

from .forward import MossTextCondition

logger = logging.getLogger(__name__)

MOSS_SHIFT = 5.0
MOSS_DEFAULT_STEPS = 50
MOSS_DEFAULT_CFG = 3.0
MOSS_SAMPLERS: tuple[str, ...] = FAMILY_SAMPLING["moss"]["samplers"]
MOSS_SCHEDULERS: tuple[str, ...] = FAMILY_SAMPLING["moss"]["schedulers"]
MOSS_DEFAULT_SAMPLER = MOSS_SAMPLERS[0]
MOSS_DEFAULT_SCHEDULER = MOSS_SCHEDULERS[0]
#: 官方 pipeline 默认 ``uncond_skip_layers=(22,)``，全程生效
MOSS_UNCOND_SKIP_LAYERS: tuple[int, ...] = (22,)
_ALIGN = 16  # VAE f8 × patch 2


@dataclass(frozen=True)
class MossSamplingCondition:
    cond: MossTextCondition
    uncond: MossTextCondition | None


def _time_snr_shift(shift: float, t: Tensor) -> Tensor:
    return shift * t / (1 + (shift - 1) * t)


def build_sigmas(steps: int, *, shift: float = MOSS_SHIFT, scheduler: str = "simple") -> Tensor:
    """ComfyUI ``simple_scheduler`` on ``ModelSamplingDiscreteFlow``；末尾补 0。"""
    if scheduler != "simple":
        raise ValueError(f"MOSS 只支持 simple 调度，收到 {scheduler!r}")
    if steps < 1:
        raise ValueError("MOSS 采样步数必须 ≥ 1")
    table = _time_snr_shift(float(shift), torch.arange(1, 1001, dtype=torch.float32) / 1000.0)
    stride = len(table) / float(steps)
    sigmas = [float(table[-(1 + int(i * stride))]) for i in range(steps)] + [0.0]
    return torch.tensor(sigmas, dtype=torch.float32)


def percent_to_sigma(percent: float, *, shift: float = MOSS_SHIFT) -> float:
    """ComfyUI ``ModelSamplingDiscreteFlow.percent_to_sigma``：Python 浮点（double）计算。

    精度要与 Comfy 一致——SLG 窗口两端闭区间，边界 σ（如 end_percent=0.5 → 5/6）
    在 double 下比 fp32 σ 表里的同一档略大，Comfy 在该步不跳层；用 fp32 算边界会
    恰好相等而多跳一步。
    """
    if percent <= 0.0:
        return 1.0
    if percent >= 1.0:
        return 0.0
    t = 1.0 - float(percent)
    shift = float(shift)
    return t if shift == 1.0 else shift * t / (1 + (shift - 1) * t)


def comfy_initial_noise(height: int, width: int, *, seed: int | None) -> Tensor:
    """Comfy txt2img 初始噪声（128ch、1/16、CPU generator），未乘 σ₀。"""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    return torch.randn(
        (1, 128, height // _ALIGN, width // _ALIGN),
        dtype=torch.float32, device="cpu", generator=generator,
    )


def prepare_sampling_condition(text, prompt: str, *, negative_prompt: str = "",
                               cfg_scale: float, device, dtype,
                               phase_callback=None) -> MossSamplingCondition:
    if phase_callback:
        phase_callback("clip")
    prompts = [str(prompt)] + ([str(negative_prompt)] if float(cfg_scale) != 1.0 else [])
    # 逐条编码（变长），不拼 batch：cond / uncond 的文本长度通常不同
    conds = [text.encode_text_for_batch([p], device=device, dtype=dtype) for p in prompts]
    return MossSamplingCondition(cond=conds[0], uncond=conds[1] if len(conds) > 1 else None)


def _validate(sampler_name: str, scheduler: str) -> tuple[str, str]:
    sampler = str(sampler_name or MOSS_DEFAULT_SAMPLER).lower().strip()
    sched = str(scheduler or MOSS_DEFAULT_SCHEDULER).lower().strip()
    if sampler not in MOSS_SAMPLERS or sched not in MOSS_SCHEDULERS:
        raise ValueError(
            f"MOSS 不支持 sampler/scheduler {sampler}+{sched}"
            f"（可选 sampler {list(MOSS_SAMPLERS)}，scheduler {list(MOSS_SCHEDULERS)}）"
        )
    return sampler, sched


@torch.no_grad()
def sample_latents(
    model,
    condition: MossSamplingCondition,
    *,
    height: int,
    width: int,
    steps: int = MOSS_DEFAULT_STEPS,
    cfg_scale: float = MOSS_DEFAULT_CFG,
    sampler_name: str = MOSS_DEFAULT_SAMPLER,
    scheduler: str = MOSS_DEFAULT_SCHEDULER,
    uncond_skip_layers: Sequence[int] = MOSS_UNCOND_SKIP_LAYERS,
    uncond_skip_start_percent: float = 0.0,
    uncond_skip_end_percent: float = 1.0,
    device="cuda",
    dtype=torch.bfloat16,
    seed: int | None = None,
    step_callback=None,
) -> Tensor:
    """返回 ``[1, 32, 1, H/8, W/8]``（本项目 latent 排布，可直接交给 VAE decode）。"""
    sampler, sched = _validate(sampler_name, scheduler)
    if height % _ALIGN or width % _ALIGN:
        raise ValueError(f"MOSS 出图尺寸须为 {_ALIGN} 的倍数，收到 {width}×{height}")
    device = torch.device(device)
    sigmas = build_sigmas(int(steps), scheduler=sched)
    # CONST noise_scaling：x = σ₀·noise（txt2img 空 latent）
    x = (comfy_initial_noise(height, width, seed=seed) * float(sigmas[0])).to(device)
    skip = tuple(int(i) for i in (uncond_skip_layers or ()))
    sigma_hi = percent_to_sigma(uncond_skip_start_percent)
    sigma_lo = percent_to_sigma(uncond_skip_end_percent)
    use_cfg = condition.uncond is not None and float(cfg_scale) != 1.0
    cond = condition.cond
    uncond = condition.uncond

    comfy_numerics = bool(getattr(model, "moss_comfy_forward", False))

    def velocity(x128: Tensor, sigma: Tensor, text: MossTextCondition, skip_layers) -> Tensor:
        if comfy_numerics:
            # 出图：ComfyUI 口径前向（逐算子复刻其舍入路径，见 comfy_forward）
            from training.families.moss.comfy_forward import comfy_forward

            mask = text.attention_mask[0].to(device).bool()
            ctx = text.context.to(device=device, dtype=dtype)[:, mask]
            v = comfy_forward(
                model, x128.to(dtype), sigma.reshape(1).to(device=device, dtype=torch.float32),
                ctx, skip_layers=skip_layers or (),
            )
            return v.float()
        x_model = unpatchify(x128).to(dtype)  # → 32ch f8（DiT 输入排布）
        v = model(
            x_model, sigma.reshape(1).to(device=device, dtype=torch.float32),
            text.context.to(device=device, dtype=dtype),
            attention_mask=text.attention_mask.to(device),
            skip_layers=skip_layers or None,
        )
        return patchify(v.float())

    def denoise_fn(x_in: Tensor, sigma_in) -> Tensor:
        sigma = sigma_in if torch.is_tensor(sigma_in) else torch.tensor(float(sigma_in))
        sigma = sigma.to(device=device, dtype=torch.float32)
        # ComfyUI 运算顺序（浮点舍入逐位一致）：每个分支先 CONST calculate_denoised
        # ``x − v·σ``（fp32），再在去噪结果上做 cfg_function ``u + (c − u)·cfg``
        x0 = x_in - velocity(x_in, sigma, cond, ()) * sigma
        if use_cfg:
            s = float(sigma)
            skip_now = skip if (skip and sigma_lo <= s <= sigma_hi) else ()
            x0_uncond = x_in - velocity(x_in, sigma, uncond, skip_now) * sigma
            x0 = x0_uncond + (x0 - x0_uncond) * float(cfg_scale)
        return x0

    from training.inference_samplers import build_inference_sampler

    sampler_fn = build_inference_sampler(sampler)
    kwargs = {"seed": seed, "shift": MOSS_SHIFT, "step_callback": step_callback}
    if sampler == "er_sde":
        kwargs.update(s_noise=1.0, max_stage=3)
    x = sampler_fn(denoise_fn, x, sigmas.to(device), **kwargs)
    return unpatchify(x.float()).unsqueeze(2)


def decode_to_pil(vae, latents: Tensor):
    import numpy as np
    from PIL import Image

    pixels = vae.decode(latents)
    if pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError(f"MOSS VAE decode 应返回单张 (1,3,T,H,W)，实际 {tuple(pixels.shape)}")
    pixels = (pixels[:, :, 0].float().clamp(-1, 1) + 1) / 2
    array = pixels[0].permute(1, 2, 0).cpu().numpy()
    # ComfyUI SaveImage 同口径：×255 后 clip 再截断取整（不四舍五入）
    return Image.fromarray((array * 255).clip(0, 255).astype(np.uint8))
