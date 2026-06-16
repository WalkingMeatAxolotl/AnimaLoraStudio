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
        patch_temporal: int = 1,
        vae_channels: int = 16,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.block_idx = block_idx
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.vae_channels = vae_channels
        self._cached_hidden: Optional[Tensor] = None

        out_per_token = vae_channels * patch_temporal * patch_spatial * patch_spatial
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

    def _hidden_as_target_grid(self, hidden: Tensor, target: Tensor) -> Tensor:
        """Reshape captured tokens to the VAE latent patch grid.

        In torch.compile native-flatten mode, block outputs are shaped as
        (B, 1, seq_len, 1, D). Rebuilding the grid from clean_latents keeps SRA
        independent of the model's internal flattening strategy.
        """
        if not isinstance(hidden, Tensor):
            raise RuntimeError(f"SRA expected block tensor output, got {type(hidden)!r}")

        B, _C, T, H, W = target.shape
        ps = self.patch_spatial
        pt = self.patch_temporal
        if T % pt != 0 or H % ps != 0 or W % ps != 0:
            raise RuntimeError(
                f"SRA target shape {target.shape} is not divisible by "
                f"patch_temporal={pt}, patch_spatial={ps}"
            )

        T_p, H_p, W_p = T // pt, H // ps, W // ps
        B_h, T_h, H_h, W_h, D = hidden.shape
        if B_h != B:
            raise RuntimeError(f"SRA batch mismatch: hidden B={B_h} vs target B={B}")

        expected_tokens = T_p * H_p * W_p
        actual_tokens = T_h * H_h * W_h
        if actual_tokens != expected_tokens:
            raise RuntimeError(
                f"SRA token mismatch: hidden grid {(T_h, H_h, W_h)} "
                f"({actual_tokens}) vs target grid {(T_p, H_p, W_p)} "
                f"({expected_tokens})"
            )

        if (T_h, H_h, W_h) == (T_p, H_p, W_p):
            return hidden
        return hidden.reshape(B, T_p, H_p, W_p, D)

    def compute(self, clean_latents: Tensor, sample_weight: Optional[Tensor] = None) -> Tensor:
        """Compute alignment loss between projected hidden state and VAE latents.

        Args:
            clean_latents: (B, C, T, H_lat, W_lat) — the original VAE-encoded latent.
            sample_weight: optional per-sample training weights, e.g. reg loss weights.

        Returns:
            Scalar alignment loss.
        """
        hidden = self._cached_hidden
        if hidden is None:
            raise RuntimeError("SRAAligner.compute() called before model forward")

        target = clean_latents.detach().float()
        hidden = self._hidden_as_target_grid(hidden, target)

        # hidden: (B, T_p, H_p, W_p, D) from block output
        B, T_p, H_p, W_p, D = hidden.shape
        ps = self.patch_spatial
        pt = self.patch_temporal
        C = self.vae_channels

        # Project: (B, T_p, H_p, W_p, D) → (B, T_p, H_p, W_p, C*pt*ps*ps)
        projected = self.proj(hidden.float())

        # Unpatchify to (B, C, T, H_lat, W_lat)
        # reshape to (B, T_p, H_p, W_p, C, pt, ps, ps)
        projected = projected.view(B, T_p, H_p, W_p, C, pt, ps, ps)
        # rearrange: b t h w c r p q -> b c (t r) (h p) (w q)
        projected = projected.permute(0, 4, 1, 5, 2, 6, 3, 7)
        projected = projected.reshape(B, C, T_p * pt, H_p * ps, W_p * ps)

        # Handle potential shape mismatch (e.g. if patch_temporal != 1)
        if projected.shape != target.shape:
            raise RuntimeError(
                f"SRA shape mismatch: projected {projected.shape} vs target {target.shape}"
            )

        loss_per_sample = F.smooth_l1_loss(
            projected, target, beta=0.05, reduction="none"
        ).mean(dim=tuple(range(1, projected.dim())))
        if sample_weight is not None:
            weight = sample_weight.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype).view(-1)
            if weight.numel() != loss_per_sample.numel():
                raise RuntimeError(
                    f"SRA sample_weight shape mismatch: {tuple(weight.shape)} "
                    f"vs batch={loss_per_sample.numel()}"
                )
            loss_per_sample = loss_per_sample * weight
        return loss_per_sample.mean()

    def get_param_groups(self, lr: Optional[float] = None) -> list[dict]:
        """Return optimizer param groups for the projection MLP."""
        group = {"params": list(self.proj.parameters()), "weight_decay": 0.0}
        if lr is not None:
            group["lr"] = lr
        return [group]

    def state_dict(self) -> dict:
        """Return checkpointable SRA state.

        Hooks and cached activations are runtime-only; only the trained
        projection head and shape metadata need to survive resume.
        """
        return {
            "block_idx": self.block_idx,
            "patch_spatial": self.patch_spatial,
            "patch_temporal": self.patch_temporal,
            "vae_channels": self.vae_channels,
            "proj": self.proj.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore projection-head weights from a checkpoint."""
        proj_state = state.get("proj", state)
        self.proj.load_state_dict(proj_state)

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
