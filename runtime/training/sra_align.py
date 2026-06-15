"""SRA v2: VAE Self-Representation Alignment for efficient LoRA training.

Based on: "SRA 2: Variational Autoencoder Self-Representation Alignment for
Efficient Diffusion Training" (CVPR 2026, arXiv:2601.17830).

Aligns intermediate transformer block hidden states to the clean VAE latent
via a lightweight projection MLP. Accelerates convergence and regularizes
representations with ~4% extra GFLOPs and zero additional model forward passes.

Usage:
    aligner = SRAAligner(model, block_idx=4, patch_spatial=2, model_channels=2048)
    # ... after model forward ...
    align_loss = aligner.compute(clean_latents)
    loss = denoising_loss + sra_weight * align_loss
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class SRAProjectionHead(nn.Module):
    """5-layer MLP projecting block hidden states to VAE latent space."""

    def __init__(self, in_dim: int, out_per_token: int):
        super().__init__()
        d1 = in_dim // 2
        d2 = in_dim // 4
        d3 = in_dim // 8
        self.net = nn.Sequential(
            nn.Linear(in_dim, d1),
            nn.SiLU(),
            nn.Linear(d1, d2),
            nn.SiLU(),
            nn.Linear(d2, d3),
            nn.SiLU(),
            nn.Linear(d3, d3),
            nn.SiLU(),
            nn.Linear(d3, out_per_token),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SRAAligner:
    """Captures intermediate block output and computes VAE alignment loss.

    Registers a forward hook on model.blocks[block_idx]. After each model
    forward pass, call .compute(clean_latents) to get the alignment loss.

    The projection MLP is trained alongside the LoRA parameters but discarded
    after training (not saved into the LoRA safetensors).
    """

    def __init__(
        self,
        model: nn.Module,
        block_idx: int,
        patch_spatial: int,
        model_channels: int,
        vae_channels: int = 16,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.block_idx = block_idx
        self.patch_spatial = patch_spatial
        self.vae_channels = vae_channels
        self._cached_hidden: Optional[Tensor] = None

        out_per_token = vae_channels * patch_spatial * patch_spatial
        self.proj = SRAProjectionHead(model_channels, out_per_token).to(
            device=device, dtype=dtype
        )

        self._hook_handle = model.blocks[block_idx].register_forward_hook(
            self._hook_fn
        )
        n_params = sum(p.numel() for p in self.proj.parameters())
        logger.info(
            f"SRA v2: hook on block[{block_idx}], "
            f"proj {model_channels} → {out_per_token}, "
            f"{n_params / 1e6:.1f}M params"
        )

    def _hook_fn(self, module, input, output):
        self._cached_hidden = output

    def compute(self, clean_latents: Tensor) -> Tensor:
        """Compute alignment loss between projected hidden state and VAE latents.

        Args:
            clean_latents: (B, C, T, H_lat, W_lat) — the original VAE-encoded latent.

        Returns:
            Scalar alignment loss.
        """
        hidden = self._cached_hidden
        if hidden is None:
            raise RuntimeError("SRAAligner.compute() called before model forward")

        # hidden: (B, T_p, H_p, W_p, D) from block output
        B, T_p, H_p, W_p, D = hidden.shape
        ps = self.patch_spatial
        C = self.vae_channels

        # Project: (B, T_p, H_p, W_p, D) → (B, T_p, H_p, W_p, C*ps*ps)
        projected = self.proj(hidden.float())

        # Unpatchify to (B, C, T, H_lat, W_lat)
        # reshape to (B, T_p, H_p, W_p, C, ps, ps)
        projected = projected.view(B, T_p, H_p, W_p, C, ps, ps)
        # rearrange: b t h w c p q -> b c t (h p) (w q)
        projected = projected.permute(0, 4, 1, 2, 5, 3, 6)  # (B, C, T_p, H_p, ps, W_p, ps)
        projected = projected.reshape(B, C, T_p, H_p * ps, W_p * ps)

        target = clean_latents.detach().float()

        # Handle potential shape mismatch (e.g. if patch_temporal != 1)
        if projected.shape != target.shape:
            raise RuntimeError(
                f"SRA shape mismatch: projected {projected.shape} vs target {target.shape}"
            )

        return F.smooth_l1_loss(projected, target, beta=0.05)

    def get_param_groups(self, lr: Optional[float] = None) -> list[dict]:
        """Return optimizer param groups for the projection MLP."""
        group = {"params": list(self.proj.parameters()), "weight_decay": 0.0}
        if lr is not None:
            group["lr"] = lr
        return [group]

    def remove_hooks(self):
        """Remove the forward hook and free cached state."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._cached_hidden = None

    def train(self):
        self.proj.train()

    def eval(self):
        self.proj.eval()
