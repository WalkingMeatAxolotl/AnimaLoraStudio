"""FFL: Focal Frequency Loss for Anima LoRA training.

Based on: "Focal Frequency Loss for Image Reconstruction and Synthesis"
(Jiang et al., ICCV 2021, arXiv:2012.12821). The model's clean-latent estimate
x̂₀ and the real latent x₀ are transformed to the frequency domain (2D FFT) and
compared there, with a *focal* per-frequency weight that adaptively up-weights
the frequencies that are hardest to synthesize (large error) and down-weights the
easy ones. This directly penalises "high / hard frequencies not learned well",
which bare spatial MSE tends to smooth over — the exact failure mode behind poor
detail/texture learning in DiT LoRA.

Design notes (mirrors SRA / LPL — an additive term on the standard RF path):
- Operates purely in *latent* space (Wan VAE is /8 spatial, the latent keeps
  spatial structure), so — unlike LPL — there is NO VAE decode, no decoder in the
  autograd graph, ~0 extra VRAM. This is the cheap, orthogonal counterpart.
- x̂₀ = noisy − t·pred is free on the rectified-flow path (no extra transformer
  forward); grads flow through pred to the LoRA.
- Zero trainable parameters. No optimizer group / state / resume wiring.

Faithful to the official implementation (github.com/EndlessSora/focal-frequency-loss):
- ``torch.fft.fft2(x, norm="ortho")``, real/imag stacked.
- frequency distance = (Δreal)² + (Δimag)²  (squared Euclidean per frequency).
- focal weight = ‖Δ‖^alpha, normalised per (sample, channel) by its spatial max,
  clamped to [0,1], and **detached** (a per-frequency importance mask, not part of
  the gradient). ``alpha`` default 1.0.
- loss = mean(weight · freq_distance). Here reduced *per-sample* so the caller can
  apply reg-set ``loss_weight`` down-weighting, same as SRA / LPL.

Usage:
    ffl = FocalFrequencyLoss(alpha=1.0)
    ffl_per_sample = ffl.compute(x_hat0, x0_ref)     # (n,) per-sample
    loss = loss + ffl_weight * ffl_per_sample.mean()
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class FocalFrequencyLoss:
    """Latent-space Focal Frequency Loss. Zero trainable parameters.

    Args:
        alpha: focal exponent on the spectrum weight matrix (paper default 1.0;
            larger = sharper focus on the hardest frequencies, 0 = uniform weight).
        log_matrix: apply ``log(1 + w)`` to the weight matrix (paper option, off).
        ave_spectrum: average the spectra over the mini-batch before comparing
            (paper option, off).
    """

    def __init__(self, alpha: float = 1.0, log_matrix: bool = False, ave_spectrum: bool = False):
        self.alpha = float(alpha)
        self.log_matrix = bool(log_matrix)
        self.ave_spectrum = bool(ave_spectrum)
        logger.info(
            "FFL: alpha=%.3f log_matrix=%s ave_spectrum=%s (latent-space, 0 trainable params)",
            self.alpha, self.log_matrix, self.ave_spectrum,
        )

    @staticmethod
    def _to_freq(x: torch.Tensor) -> torch.Tensor:
        """``[n, C, H, W]`` → ``[n, C, H, W, 2]`` (real, imag), orthonormal 2D FFT."""
        freq = torch.fft.fft2(x, norm="ortho")
        return torch.stack([freq.real, freq.imag], dim=-1)

    def compute(self, recon: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Per-sample focal frequency loss.

        Args:
            recon: predicted clean latent x̂₀ ``[n, C, (T,) H, W]`` (requires grad).
            real:  real latent x₀ ``[n, C, (T,) H, W]`` (target; grads disabled).
        Returns:
            Per-sample loss ``(n,)``.
        """
        n = recon.shape[0]
        h, w = recon.shape[-2:]
        # 折叠所有非 batch/非空间维当通道（兼容图像 T=1 的 5D 与 4D）；FFT 走 fp32 保数值稳定。
        recon = recon.reshape(n, -1, h, w).float()
        real = real.reshape(n, -1, h, w).float().detach()

        with torch.autocast("cuda", enabled=False):
            recon_freq = self._to_freq(recon)
            real_freq = self._to_freq(real)
            if self.ave_spectrum:
                recon_freq = recon_freq.mean(0, keepdim=True).expand_as(recon_freq)
                real_freq = real_freq.mean(0, keepdim=True).expand_as(real_freq)

            tmp = (recon_freq - real_freq) ** 2
            freq_distance = tmp[..., 0] + tmp[..., 1]            # [n, C, H, W]

            # focal 权重矩阵：detach（是重要性 mask，不进梯度），逐 (样本,通道) 按空间 max 归一。
            with torch.no_grad():
                weight = freq_distance ** (self.alpha / 2.0)     # = ‖Δ‖^alpha
                if self.log_matrix:
                    weight = torch.log(weight + 1.0)
                norm = weight.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
                weight = (weight / norm).clamp(0.0, 1.0)

            loss = weight * freq_distance                        # [n, C, H, W]
        return loss.mean(dim=(1, 2, 3))                          # per-sample (n,)
