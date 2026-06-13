"""DPM-Solver++(3M) SDE for CONST flow —— 逐行对齐 ComfyUI k_diffusion.sample_dpmpp_3m_sde。

对齐要点（comfy/k_diffusion/sampling.py + comfy/model_sampling.py，GPL-3.0）：
- λ (CONST half-log-SNR) = sigma.logit().neg() = log((1-σ)/σ)
- alpha_t = σ_{i+1} * exp(λ_t)  （CONST 下恒等于 1-σ_{i+1}，保留 ComfyUI 写法）
- offset_first_sigma_for_snr：σ_0 ≥ 1 时替换为 percent_to_sigma(1e-4)
- 噪声：BrownianTreeNoiseSampler，transform=identity（按 σ 直接做时间轴，
  **非** -log(σ)），BatchedBrownianTree 在 CPU 上跑（cpu=True）以对齐 RNG。

这让 dpmpp_3m_sde 出图对齐 ComfyUI，与用独立 Gaussian 的 [[er_sde]] 拉开差距。
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch

# torchsde 在边界 σ（== tree t0/t1）查询时，因 float32 往返误差产生 ta<t0 / tb>t1
# 几个 ULP 的越界，打 UserWarning 后 clamp 回边界（数值正确）。ComfyUI 同样命中
# 此告警、只是不 promote 成 error。这里精准静音该条，避免刷屏。
warnings.filterwarnings(
    "ignore",
    message=r"Should have t[ab][<>]=t[01] but got",
    module="torchsde",
)


class _BrownianTreeNoiseSampler:
    """对齐 ComfyUI BrownianTreeNoiseSampler + BatchedBrownianTree。

    transform=identity：t0/t1 直接取 σ_min/σ_max（不做 -log）。tree 在 CPU
    上构建、查询后搬回原 device，复刻 ComfyUI cpu=True 路径的 RNG 行为。
    """

    def __init__(self, x: torch.Tensor, sigma_min: float, sigma_max: float, seed: Optional[int] = None):
        from torchsde import BrownianTree

        # ComfyUI: transform=identity，t0=σ_min, t1=σ_max。统一用 float32 tensor
        # 存边界（与 __call__ 查询时的 .float() 同精度），避免 float64 边界点
        # 触发 torchsde 的 ta<t0 / tb>t1 越界告警。
        t0 = torch.as_tensor(sigma_min, dtype=torch.float32)
        t1 = torch.as_tensor(sigma_max, dtype=torch.float32)
        self._sign = 1
        if float(t0) > float(t1):
            t0, t1, self._sign = t1, t0, -1
        if seed is None:
            seed = int(torch.randint(0, 2**63 - 1, ()).item())
        # cpu=True：tree 在 CPU 上，w0 也在 CPU
        self._device = x.device
        self._dtype = x.dtype
        self._w0 = torch.zeros_like(x, device="cpu")
        self._tree = BrownianTree(t0, self._w0, t1, entropy=int(seed))

    def __call__(self, sigma: float, sigma_next: float) -> torch.Tensor:
        t0 = torch.as_tensor(sigma, dtype=torch.float32)
        t1 = torch.as_tensor(sigma_next, dtype=torch.float32)
        sign = 1
        if float(t0) > float(t1):
            t0, t1, sign = t1, t0, -1
        delta = float(t1) - float(t0)
        if delta < 1e-12:
            return self._w0.to(device=self._device, dtype=self._dtype)
        w = self._tree(t0, t1).to(device=self._device, dtype=self._dtype) * (self._sign * sign)
        return w / (delta ** 0.5)


def _gaussian_noise_sampler(x: torch.Tensor, seed: Optional[int]):
    """torchsde 不可用时的兜底：独立 Gaussian。"""
    if seed is not None:
        if x.device.type == "cpu":
            seed = int(seed) + 1
        g = torch.Generator(device=x.device)
        g.manual_seed(int(seed))
    else:
        g = None

    def _sample(_sigma, _sigma_next):
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=g)

    return _sample


def _build_noise_sampler(
    x: torch.Tensor,
    sigmas: torch.Tensor,
    seed: Optional[int],
    *,
    require_brownian_tree: bool = False,
):
    """优先 BrownianTree（对齐 ComfyUI）；torchsde 缺失时回退独立 Gaussian。"""
    positive = sigmas[sigmas > 0]
    sigma_min = float(positive.min()) if positive.numel() > 0 else 1e-3
    sigma_max = float(sigmas.max())
    try:
        return _BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)
    except ImportError as exc:
        if require_brownian_tree:
            raise RuntimeError(
                "Comfy parity dpmpp_3m_sde requires torchsde BrownianTree noise"
            ) from exc
        import logging
        logging.getLogger(__name__).warning(
            "torchsde 未安装，dpmpp_3m_sde 回退独立 Gaussian 噪声（与 ComfyUI / er_sde 差异变化）"
        )
        return _gaussian_noise_sampler(x, seed=seed)


def _offset_first_sigma_for_snr(sigmas: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """对齐 ComfyUI offset_first_sigma_for_snr（CONST）。

    σ_0 ≥ 1 会让 logit 爆 inf；ComfyUI 用 percent_to_sigma(1e-4) 替换：
    = time_snr_shift(shift, 1 - 1e-4)。
    """
    if sigmas.numel() <= 1:
        return sigmas
    if float(sigmas[0]) >= 1.0:
        sigmas = sigmas.clone()
        t = 1.0 - 1e-4
        sigmas[0] = shift * t / (1.0 + (shift - 1.0) * t)
    return sigmas


@torch.no_grad()
def sample(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    seed: Optional[int] = None,
    s_noise: float = 1.0,
    eta: float = 1.0,
    shift: float = 3.0,
    require_brownian_tree: bool = False,
    step_callback=None,
    **_unused,
) -> torch.Tensor:
    """DPM-Solver++(3M) SDE —— 对齐 ComfyUI。

    Args:
        denoise_fn: 输入 (x, sigma) 返回 x0 估计。
        sigmas: 从高到低的 sigma 序列，末尾带 0.0。
        eta: SDE 噪声强度（1.0 = ComfyUI 默认；0.0 退化为 ODE 多步）。
        s_noise: 噪声缩放。
        shift: ModelSamplingDiscreteFlow shift（用于 σ_0 offset，默认 3.0）。
        require_brownian_tree: True 时缺少 torchsde 直接失败，不回退 Gaussian。
        step_callback: 每步回调 (step, total, denoised)。
    """
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    if sigmas.numel() <= 1:
        return x

    noise_sampler = _build_noise_sampler(
        x,
        sigmas,
        seed=seed,
        require_brownian_tree=require_brownian_tree,
    )
    # offset_first_sigma_for_snr（在算 noise_sampler 的 σ_min/σ_max 之后，
    # 与 ComfyUI 顺序一致：BrownianTreeNoiseSampler 用原始 sigmas[sigmas>0]）
    sigmas = _offset_first_sigma_for_snr(sigmas, shift=shift)
    eps = 1e-7

    def half_log_snr(sigma: torch.Tensor) -> torch.Tensor:
        # CONST: log((1-σ)/σ) = logit(σ).neg()
        s = sigma.clamp(min=eps, max=1.0 - eps)
        return torch.log((1.0 - s) / s)

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in range(len(sigmas) - 1):
        sigma_i = sigmas[i]
        denoised = denoise_fn(x, sigma_i)

        if step_callback is not None:
            try:
                step_callback(i, len(sigmas) - 1, denoised)
            except Exception:
                pass

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lambda_s = half_log_snr(sigma_i)
            lambda_t = half_log_snr(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1.0)
            alpha_t = sigmas[i + 1] * lambda_t.exp()  # = 1 - σ_{i+1} (CONST)

            x = sigmas[i + 1] / sigma_i * (-h * eta).exp() * x \
                + alpha_t * (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1.0
                phi_3 = phi_2 / h_eta - 0.5
                x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1.0
                x = x + (alpha_t * phi_2) * d

            if eta:
                x = x + noise_sampler(float(sigma_i), float(sigmas[i + 1])) * sigmas[i + 1] \
                    * (-2 * h * eta).expm1().neg().sqrt() * float(s_noise)

            denoised_1, denoised_2 = denoised, denoised_1
            h_1, h_2 = h, h_1

    return x
