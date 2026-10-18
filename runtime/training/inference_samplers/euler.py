"""Euler（一阶 ODE）在 CONST(flow) 噪声日程下的实现。

逐行对齐 ComfyUI ``k_diffusion/sampling.py`` 的 ``sample_euler``（s_churn=0 默认路径）：
``d = (x − denoised) / σ``，``x ← x + d · (σ_next − σ)``。CONST 下 σ=1 时
``d`` 就是速度预测。确定性采样，``seed`` 不参与（接口与其它 sampler 一致）。
"""

from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def sample(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    seed: Optional[int] = None,
    shift: float = 3.0,
    step_callback=None,
) -> torch.Tensor:
    """``denoise_fn(x, sigma) -> x0``；``step_callback(step, total, denoised)`` 同 er_sde。"""
    del seed, shift  # 确定性；Comfy euler 不做 offset_first_sigma
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    total = len(sigmas) - 1
    for i in range(total):
        sigma = sigmas[i]
        denoised = denoise_fn(x, sigma)
        if step_callback is not None:
            try:
                step_callback(i, total, denoised)
            except Exception:
                pass  # 预览失败不该影响采样
        d = (x - denoised) / sigma
        x = x + d * (sigmas[i + 1] - sigma)
    return x
