"""Region-balance schedule and APT-inspired adaptive training utilities.

The APT controller implements the paper's timestep-bin overfitting indicator
and adaptive loss weighting.  Anima uses continuous rectified-flow time and
latent-space affine augmentation, so this is explicitly an engineering
adaptation rather than a claim of bit-for-bit SDXL APT reproduction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionBalanceStats:
    scale: float
    coverage: float
    annotated_fraction: float
    mean_weight: float


def region_schedule_scale(
    global_step: int,
    total_steps: Optional[int],
    *,
    hold_ratio: float = 0.45,
    end_ratio: float = 0.55,
) -> float:
    """Full strength, cosine transition, then exact whole-image training."""
    if not total_steps or total_steps <= 0:
        return 1.0
    progress = max(0.0, min(1.0, float(global_step) / float(total_steps)))
    hold = max(0.0, min(1.0, float(hold_ratio)))
    end = max(hold, min(1.0, float(end_ratio)))
    if progress <= hold:
        return 1.0
    if progress >= end or end <= hold:
        return 0.0
    x = (progress - hold) / (end - hold)
    return 0.5 * (1.0 + math.cos(math.pi * x))


def build_region_spatial_weight(
    regions: Optional[torch.Tensor],
    ignore_mask: Optional[torch.Tensor],
    region_weights: Optional[torch.Tensor],
    *,
    global_step: int,
    total_steps: Optional[int],
    max_weight: float,
    hold_ratio: float,
    end_ratio: float,
) -> tuple[Optional[torch.Tensor], RegionBalanceStats]:
    """Combine positive focus and negative ignore masks in latent space."""
    scale = region_schedule_scale(
        global_step, total_steps, hold_ratio=hold_ratio, end_ratio=end_ratio,
    )
    spatial = ignore_mask
    coverage = 0.0
    annotated_fraction = 0.0
    if regions is not None:
        focus = regions.float().clamp(0, 1)
        dims = tuple(range(1, focus.dim()))
        per_sample_coverage = focus.mean(dim=dims)
        annotated = per_sample_coverage > 0
        coverage = float(
            per_sample_coverage[annotated].mean().item() if annotated.any() else 0.0
        )
        annotated_fraction = float(annotated.float().mean().item())
        if scale > 0.0:
            local = torch.ones(focus.shape[0], device=focus.device, dtype=focus.dtype)
            if region_weights is not None:
                local = region_weights.to(focus.device, dtype=focus.dtype).view(-1)
            boost = max(0.0, float(max_weight) - 1.0) * float(scale)
            region_spatial = 1.0 + focus * boost * local.view(-1, 1, 1)
            spatial = region_spatial if spatial is None else spatial * region_spatial
    mean_weight = float(spatial.mean().item()) if spatial is not None else 1.0
    return spatial, RegionBalanceStats(
        scale=scale,
        coverage=coverage,
        annotated_fraction=annotated_fraction,
        mean_weight=mean_weight,
    )


class AptAdaptiveController:
    """EMA loss-gap indicator over continuous-timestep bins."""

    def __init__(self, bins: int = 10, ema_alpha: float = 0.1, denoising_steps: int = 1000):
        self.bins = max(1, int(bins))
        self.alpha = max(0.0, min(1.0, float(ema_alpha)))
        self.denoising_steps = max(1, int(denoising_steps))
        self.base_ema = torch.zeros(self.bins, dtype=torch.float64)
        self.tuned_ema = torch.zeros(self.bins, dtype=torch.float64)
        self.initialized = torch.zeros(self.bins, dtype=torch.bool)

    def bin_indices(self, t: torch.Tensor) -> torch.Tensor:
        return (t.detach().float().clamp(0, 1) * self.bins).long().clamp_max(self.bins - 1)

    def gamma_bins(self) -> torch.Tensor:
        gap = self.base_ema - self.tuned_ema
        gamma = 1.0 - torch.exp(-float(self.denoising_steps) * gap)
        gamma = gamma.clamp(0.0, 1.0)
        return torch.where(self.initialized, gamma, torch.zeros_like(gamma))

    def gamma_for(self, t: torch.Tensor, *, device=None) -> torch.Tensor:
        values = self.gamma_bins()[self.bin_indices(t).cpu()].float()
        return values.to(device=device or t.device)

    @torch.no_grad()
    def update(
        self,
        t: torch.Tensor,
        base_loss: torch.Tensor,
        tuned_loss: torch.Tensor,
        main_mask: Optional[torch.Tensor] = None,
    ) -> None:
        indices = self.bin_indices(t).cpu()
        base = base_loss.detach().float().cpu().view(-1)
        tuned = tuned_loss.detach().float().cpu().view(-1)
        keep = (
            main_mask.detach().bool().cpu().view(-1)
            if main_mask is not None else torch.ones_like(indices, dtype=torch.bool)
        )
        for idx in range(self.bins):
            selected = keep & (indices == idx)
            if not selected.any():
                continue
            b = base[selected].mean().double()
            q = tuned[selected].mean().double()
            if not self.initialized[idx]:
                self.base_ema[idx] = b
                self.tuned_ema[idx] = q
                self.initialized[idx] = True
            else:
                a = self.alpha
                self.base_ema[idx].lerp_(b, a)
                self.tuned_ema[idx].lerp_(q, a)

    def loss_weights(
        self, t: torch.Tensor, main_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weights = 1.0 - self.gamma_for(t, device=t.device)
        if main_mask is not None:
            weights = torch.where(main_mask.to(t.device).bool(), weights, torch.ones_like(weights))
        return weights

    def status(self) -> dict[str, object]:
        gamma = self.gamma_bins().float()
        return {
            "gamma_mean": float(gamma.mean().item()),
            "gamma_max": float(gamma.max().item()),
            "gamma_bins": [float(x) for x in gamma.tolist()],
            "initialized_bins": int(self.initialized.sum().item()),
        }

    def state_dict(self) -> dict[str, torch.Tensor | int | float]:
        return {
            "bins": self.bins,
            "ema_alpha": self.alpha,
            "denoising_steps": self.denoising_steps,
            "base_ema": self.base_ema.clone(),
            "tuned_ema": self.tuned_ema.clone(),
            "initialized": self.initialized.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("bins", self.bins)) != self.bins:
            raise ValueError("APT checkpoint bins 与当前配置不一致")
        self.base_ema.copy_(torch.as_tensor(state["base_ema"], dtype=torch.float64))
        self.tuned_ema.copy_(torch.as_tensor(state["tuned_ema"], dtype=torch.float64))
        self.initialized.copy_(torch.as_tensor(state["initialized"], dtype=torch.bool))


def apply_adaptive_affine(
    latents: torch.Tensor,
    regions: Optional[torch.Tensor],
    ignore_mask: Optional[torch.Tensor],
    gamma: torch.Tensor,
    *,
    p_max: float = 0.8,
    zoom_max: float = 3.0,
    rotation_degrees: float = 15.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float]:
    """Apply the same sampled latent-space affine grid to data and maps."""
    if latents.dim() != 5 or latents.shape[2] != 1:
        return latents, regions, ignore_mask, 0.0
    b = latents.shape[0]
    probability = gamma.to(latents.device).float().clamp(0, float(p_max))
    selected = torch.rand(b, device=latents.device) < probability
    if not selected.any():
        return latents, regions, ignore_mask, 0.0
    zoom = 1.0 + torch.rand(b, device=latents.device) * (max(1.0, zoom_max) - 1.0)
    angle = (torch.rand(b, device=latents.device) * 2.0 - 1.0) * math.radians(rotation_degrees)
    zoom = torch.where(selected, zoom, torch.ones_like(zoom))
    angle = torch.where(selected, angle, torch.zeros_like(angle))
    cos, sin = torch.cos(angle) * zoom, torch.sin(angle) * zoom
    theta = torch.zeros(b, 2, 3, device=latents.device, dtype=torch.float32)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    flat = latents[:, :, 0]
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    transformed = F.grid_sample(
        flat.float(), grid, mode="bilinear", padding_mode="reflection", align_corners=False,
    ).to(latents.dtype)
    out = torch.where(selected.view(-1, 1, 1, 1), transformed, flat).unsqueeze(2)

    def transform_map(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        mapped = F.grid_sample(
            value.float().unsqueeze(1), grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )[:, 0]
        mapped = torch.where(selected.view(-1, 1, 1), mapped, value.float())
        return mapped.clamp(0, 1)

    return out, transform_map(regions), transform_map(ignore_mask), float(selected.float().mean().item())
