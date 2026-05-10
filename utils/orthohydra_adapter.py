"""Ortho-Hydra adapter for Anima (DiT).

Multi-expert LoRA with orthogonal SVD-initialized expert subspaces and
Cayley-parameterized rotations. Solves the HydraLoRA cold-start deadlock
where zero-initialized experts never specialize.

Reference: "Ortho-Hydra: Orthogonalized Experts for DiT LoRA" (arXiv 2605.03252)

Architecture per layer:
  - Frozen buffers: Q_basis (r, d_in), P_bases (E, d_out, r)  — from top-(E·r) SVD of W₀
  - Learnable: S_q (r,r), S_p (E,r,r)  — skew-symmetric seeds for Cayley rotations
  - Learnable: lambda_layer (1,r)       — shared scale, ZERO-INIT (ensures ΔW=0 at step 0)
  - Learnable: router Linear(r+4, E)    — per-layer, routes from pre-λ RMS-pooled bottleneck

Forward:
  1. Batched Cayley: R = solve(I+A, I-A) for A = S - Sᵀ, all (E+1) matrices in one call
  2. Q_eff = R_q @ Q_basis,  P_eff = bmm(P_bases, R_p)
  3. lx = x @ Q_effᵀ  (shared bottleneck, pre-λ)
  4. rms_pool(lx) → router(concat(pool, σ_feat)) → softmax gate g ∈ Δ^(E-1)
  5. lx_scaled = lx * lambda_layer * timestep_mask
  6. delta = einsum(g, P_eff, lx_scaled) * scale

Key design decisions (from upstream sorryhyun/anima_lora):
  - Router input uses pre-λ bottleneck: λ is zero-init, post-λ routing would freeze router
  - RMS pool (not mean): mean collapses for zero-mean activations at long sequence lengths
  - Batched linalg.solve: single kernel for all (E+1) Cayley inverses
  - σ-feature columns of router weight are zero-init: step-0 routing is σ-agnostic
  - P_bases are disjoint left singular subspaces → P_eff[i]ᵀ P_eff[j] = 0, no deadlock
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

_TARGET_SUBPATHS: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.k_proj",
    "cross_attn.v_proj",
    "cross_attn.output_proj",
    "mlp.layer1",
    "mlp.layer2",
)

_SIGMA_FEATURE_DIM = 4  # 2 log-spaced freqs → cos+sin → 4 features


def _sigma_to_features(sigma: float, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sinusoidal σ features matching DiT t_embedder (log-10000 frequencies)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=device) / half
    )
    args = torch.tensor(sigma, dtype=torch.float32, device=device) * freqs
    return torch.cat([torch.cos(args), torch.sin(args)]).to(dtype)


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------

class OrthoHydraLinearLayer(nn.Module):
    """Replaces a single nn.Linear with an Ortho-Hydra multi-expert LoRA."""

    def __init__(
        self,
        original: nn.Linear,
        rank: int,
        alpha: float,
        num_experts: int,
        sigma_feature_dim: int = _SIGMA_FEATURE_DIM,
    ) -> None:
        super().__init__()
        out_f, in_f = original.out_features, original.in_features
        self.scale = alpha / rank
        self.sigma_feature_dim = sigma_feature_dim

        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)

        # --- SVD initialization ---
        # Need E*r left + r right singular vectors.
        # Fall back if layer is too narrow.
        max_q = min(out_f, in_f)
        desired_q = num_experts * rank + 6
        if max_q < rank:
            raise ValueError(
                f"Layer ({out_f}×{in_f}) smaller than rank {rank}; cannot inject OrthoHydra."
            )
        actual_experts = num_experts
        if max_q < desired_q:
            actual_experts = max(1, max_q // rank)
            logger.warning(
                "Layer (%d×%d) supports only %d experts at rank %d (requested %d). "
                "Falling back to %d experts.",
                out_f, in_f, actual_experts, rank, num_experts, actual_experts,
            )
        q = min(actual_experts * rank + 6, max_q)

        W = original.weight.data.float()
        U, _S, V = torch.svd_lowrank(W, q=q, niter=2)  # faster than full SVD for r << min(m,n)

        Q_init = V[:, :rank].T.contiguous()                              # (r, in_f)
        target_cols = actual_experts * rank
        P_stack = U[:, :target_cols].reshape(out_f, actual_experts, rank)
        P_init = P_stack.permute(1, 0, 2).contiguous()                  # (E, out_f, r)

        # Persistent=True so they're saved/loaded with state_dict
        self.register_buffer("Q_basis", Q_init, persistent=True)
        self.register_buffer("P_bases", P_init, persistent=True)
        # Eye pre-allocated to avoid repeated kernel launches in forward
        self.register_buffer("_eye_r", torch.eye(rank), persistent=False)

        # --- Learnable parameters ---
        self.S_q = nn.Parameter(torch.zeros(rank, rank))
        self.S_p = nn.Parameter(torch.zeros(actual_experts, rank, rank))
        # Zero-init: ensures ΔW=0 at step 0; also why router reads pre-λ bottleneck
        self.lambda_layer = nn.Parameter(torch.zeros(1, rank))

        # --- Router ---
        # Input: rms-pooled pre-λ rank-R signal + σ sinusoidal features
        router_in = rank + sigma_feature_dim
        self.router = nn.Linear(router_in, actual_experts, bias=True)
        nn.init.normal_(self.router.weight[:, :rank], std=0.01)
        nn.init.zeros_(self.router.weight[:, rank:])  # σ columns zero-init
        nn.init.zeros_(self.router.bias)

        # --- Shared sigma buffers (aliased by adapter for O(1) set_sigma) ---
        self.register_buffer("_sigma", torch.tensor(0.5), persistent=False)
        self.register_buffer("_sigma_features", torch.zeros(sigma_feature_dim), persistent=False)
        # Timestep mask for T-LoRA composition; defaults to all-ones (identity)
        self.register_buffer("_timestep_mask", torch.ones(1, rank), persistent=False)

        self.rank = rank
        self.num_experts = actual_experts
        self._last_gate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original(x)
        eye = self._eye_r.to(x.dtype)

        # 1. Batched Cayley for all (E+1) rotation matrices in one linalg.solve call
        #    A = S - Sᵀ  (skew-symmetric)
        #    R = (I+A)⁻¹(I-A)  — right Cayley map, equivalent orthogonal rotation
        skew = torch.cat([self.S_q.unsqueeze(0), self.S_p], dim=0)  # (E+1, r, r)
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(eye + A, eye - A)                     # (E+1, r, r)
        R_q = R[0]   # (r, r)
        R_p = R[1:]  # (E, r, r)

        # 2. Effective bases
        Q_eff = R_q @ self.Q_basis.to(x.dtype)              # (r, in_f)
        P_eff = torch.bmm(self.P_bases.to(x.dtype), R_p)    # (E, out_f, r)

        # 3. Shared bottleneck (pre-λ; used for routing before scaling)
        lx = x @ Q_eff.T   # (B, L, r) or (B, r)

        # 4. RMS pool pre-λ signal → router
        #    RMS instead of mean: mean collapses zero-mean activations as 1/√L at long seqs
        if lx.dim() == 3:
            pooled = lx.pow(2).mean(dim=1).add(1e-8).sqrt()   # (B, r)
        else:
            pooled = lx                                         # (B, r)

        sf = self._sigma_features.to(x.dtype).expand(x.shape[0], -1)   # (B, d_σ)
        gate = torch.softmax(
            self.router(torch.cat([pooled, sf], dim=-1)), dim=-1
        )  # (B, E)
        self._last_gate = gate

        # 5. Scale bottleneck with λ and optional timestep mask
        lx_scaled = lx * self.lambda_layer.to(x.dtype) * self._timestep_mask.to(x.dtype)

        # 6. Soft expert mixture and output projection
        P_comb = torch.einsum("be,eor->bor", gate, P_eff)   # (B, out_f, r)
        if lx_scaled.dim() == 3:
            delta = torch.einsum("blr,bor->blo", lx_scaled, P_comb) * self.scale
        else:
            delta = torch.einsum("br,bor->bo", lx_scaled, P_comb) * self.scale

        return orig_out + delta


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class AnimaOrthoHydraAdapter:
    """Ortho-Hydra adapter with the same interface as AnimaTLoRAAdapter / AnimaLycorisAdapter.

    Injects OrthoHydraLinearLayer into all target Linear layers of model.blocks[*],
    shares sigma buffers across all layers via aliasing (O(1) set_sigma),
    and exposes get_balance_loss() for the Switch-Transformer auxiliary loss.
    """

    def __init__(
        self,
        rank: int = 32,
        alpha: float = 32.0,
        num_experts: int = 8,
        balance_loss_weight: float = 5e-7,
        balance_warmup_ratio: float = 0.4,
        router_lr_scale: float = 10.0,
    ) -> None:
        self.rank = rank
        self.alpha = alpha
        self.num_experts = num_experts
        self.balance_loss_weight = balance_loss_weight
        self.balance_warmup_ratio = balance_warmup_ratio
        self.router_lr_scale = router_lr_scale

        # Compatibility fields expected by anima_train.py
        self.algo = "orthohydra"
        self.use_lokr = False

        self._layers: list[OrthoHydraLinearLayer] = []
        self._layer_keys: dict[str, OrthoHydraLinearLayer] = {}
        self._injected_model: Optional[nn.Module] = None

        # Single shared tensors aliased across all layers; set in inject()
        self._sigma_buf: Optional[torch.Tensor] = None
        self._sigma_feat_buf: Optional[torch.Tensor] = None

    # --------------------------------------------------------------- inject

    def inject(self, model: nn.Module) -> dict[str, OrthoHydraLinearLayer]:
        if not hasattr(model, "blocks"):
            raise RuntimeError("AnimaOrthoHydraAdapter: model has no .blocks")

        try:
            ref = next(model.parameters())
            _device, _dtype = ref.device, ref.dtype
        except StopIteration:
            _device, _dtype = torch.device("cpu"), torch.float32

        # Shared sigma buffers on model device (float32; forward casts as needed)
        self._sigma_buf = torch.tensor(0.5, device=_device)
        self._sigma_feat_buf = torch.zeros(_SIGMA_FEATURE_DIM, device=_device)

        injected: dict[str, OrthoHydraLinearLayer] = {}
        failed = 0

        for block_idx, block in enumerate(model.blocks):
            for subpath in _TARGET_SUBPATHS:
                parts = subpath.split(".")
                parent: nn.Module = block
                try:
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    attr = parts[-1]
                    original: nn.Linear = getattr(parent, attr)
                except AttributeError:
                    continue

                if not isinstance(original, nn.Linear):
                    continue

                key = f"lora_unet_blocks_{block_idx}_{subpath.replace('.', '_')}"

                try:
                    layer = OrthoHydraLinearLayer(
                        original, self.rank, self.alpha, self.num_experts,
                    )
                except ValueError as e:
                    logger.warning("Skipping %s: %s", key, e)
                    failed += 1
                    continue

                layer.to(device=_device, dtype=_dtype)

                # Alias shared sigma buffers AFTER .to() so device is consistent
                layer._buffers["_sigma"] = self._sigma_buf
                layer._buffers["_sigma_features"] = self._sigma_feat_buf

                setattr(parent, attr, layer)
                self._layers.append(layer)
                self._layer_keys[key] = layer
                injected[key] = layer

        self._injected_model = model
        logger.info(
            "Ortho-Hydra 注入 %d 层（rank=%d, experts=%d）%s",
            len(injected), self.rank, self.num_experts,
            f"，跳过 {failed} 层（维度不足）" if failed else "",
        )
        return injected

    # --------------------------------------------------------------- sigma

    def set_sigma(self, sigma: float) -> None:
        """Call once per training step before forward. Updates all layers via buffer aliasing."""
        if self._sigma_buf is None or not self._layers:
            return
        device = self._sigma_buf.device
        dtype = self._layers[0].S_q.dtype
        self._sigma_buf.fill_(sigma)
        feat = _sigma_to_features(sigma, _SIGMA_FEATURE_DIM, device, dtype)
        self._sigma_feat_buf.copy_(feat.to(self._sigma_feat_buf.device))

    # --------------------------------------------------------------- balance loss

    def get_balance_loss(self) -> torch.Tensor:
        """Switch-Transformer balance loss averaged across all layers.

        L = E * Σ_e (frac_e * mean_gate_e)
        where frac_e = fraction of batch hard-assigned to expert e.
        """
        total: Optional[torch.Tensor] = None
        count = 0
        for layer in self._layers:
            if layer._last_gate is None:
                continue
            gate = layer._last_gate   # (B, E)
            B, E = gate.shape
            expert_idx = gate.argmax(dim=-1)  # (B,) hard assignment
            frac = torch.zeros(E, device=gate.device, dtype=gate.dtype)
            frac.scatter_add_(0, expert_idx, torch.ones(B, device=gate.device, dtype=gate.dtype))
            frac = frac / B
            gate_mean = gate.mean(dim=0)   # (E,)
            layer_loss = E * (frac * gate_mean).sum()
            total = layer_loss if total is None else total + layer_loss
            count += 1

        if total is None or count == 0:
            return torch.tensor(0.0)
        return total / count

    # --------------------------------------------------------------- detach

    def detach(self) -> bool:
        if self._injected_model is None:
            return True
        model = self._injected_model
        if not hasattr(model, "blocks"):
            return False
        try:
            for block_idx, block in enumerate(model.blocks):
                for subpath in _TARGET_SUBPATHS:
                    parts = subpath.split(".")
                    parent: nn.Module = block
                    try:
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        attr = parts[-1]
                    except AttributeError:
                        continue
                    layer = getattr(parent, attr, None)
                    if isinstance(layer, OrthoHydraLinearLayer):
                        setattr(parent, attr, layer.original)
        except Exception as exc:
            logger.warning("Ortho-Hydra detach 部分失败: %s", exc)
            return False
        self._layers.clear()
        self._layer_keys.clear()
        self._injected_model = None
        return True

    # --------------------------------------------------------------- params

    def get_params(self) -> list[nn.Parameter]:
        params = []
        for layer in self._layers:
            params.extend([layer.S_q, layer.S_p, layer.lambda_layer,
                           layer.router.weight, layer.router.bias])
        return params

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        """Two groups: adapter params (with wd) and router params (no wd, higher lr).

        Router group is marked with '_is_router_group': True so the training
        script can set lr = base_lr * router_lr_scale after optimizer creation.
        """
        adapter_params: list[nn.Parameter] = []
        router_params: list[nn.Parameter] = []
        for layer in self._layers:
            adapter_params.extend([layer.S_q, layer.S_p, layer.lambda_layer])
            router_params.extend([layer.router.weight, layer.router.bias])
        return [
            {"params": adapter_params, "weight_decay": weight_decay},
            {"params": router_params, "weight_decay": 0.0, "_is_router_group": True},
        ]

    def named_trainable_params(self) -> list[tuple[str, nn.Parameter]]:
        """For OrthoGrad: returns (name, param) pairs.

        lambda_layer is zero-init and magnitude-bearing — callers should add
        'lambda_layer' to OrthoGrad exclude_param_keywords to skip it.
        """
        result = []
        for key, layer in self._layer_keys.items():
            result.extend([
                (f"{key}.S_q", layer.S_q),
                (f"{key}.S_p", layer.S_p),
                (f"{key}.lambda_layer", layer.lambda_layer),
                (f"{key}.router.weight", layer.router.weight),
                (f"{key}.router.bias", layer.router.bias),
            ])
        return result

    # --------------------------------------------------------------- state I/O

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        for key, layer in self._layer_keys.items():
            sd[f"{key}.S_q"] = layer.S_q.data.clone()
            sd[f"{key}.S_p"] = layer.S_p.data.clone()
            sd[f"{key}.lambda_layer"] = layer.lambda_layer.data.clone()
            sd[f"{key}.router.weight"] = layer.router.weight.data.clone()
            sd[f"{key}.router.bias"] = layer.router.bias.data.clone()
            # Save frozen bases so the file is self-contained for inference
            sd[f"{key}.Q_basis"] = layer.Q_basis.clone()
            sd[f"{key}.P_bases"] = layer.P_bases.clone()
            sd[f"{key}.alpha"] = torch.tensor(float(self.alpha))
        return sd

    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True):
        _LEARNABLE = ("S_q", "S_p", "lambda_layer", "router.weight", "router.bias")
        _BUFFERS   = ("Q_basis", "P_bases")
        _ALL = _LEARNABLE + _BUFFERS + ("alpha",)

        missing, unexpected = [], []
        expected: set[str] = set()

        for key, layer in self._layer_keys.items():
            for suffix in _ALL:
                full = f"{key}.{suffix}"
                expected.add(full)
                if full not in sd:
                    if strict and suffix not in ("alpha",):
                        missing.append(full)
                    continue
                t = sd[full]
                if suffix == "S_q":
                    layer.S_q.data.copy_(t)
                elif suffix == "S_p":
                    layer.S_p.data.copy_(t)
                elif suffix == "lambda_layer":
                    layer.lambda_layer.data.copy_(t)
                elif suffix == "router.weight":
                    layer.router.weight.data.copy_(t)
                elif suffix == "router.bias":
                    layer.router.bias.data.copy_(t)
                elif suffix == "Q_basis":
                    layer.Q_basis.copy_(t)
                elif suffix == "P_bases":
                    layer.P_bases.copy_(t)

        for k in sd:
            if k not in expected:
                unexpected.append(k)

        return type("Result", (), {"missing_keys": missing, "unexpected_keys": unexpected})()

    def save(self, path: str | Path) -> None:
        sd = self.state_dict()
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": "orthohydra",
            "ss_network_args": json.dumps({
                "algo": "orthohydra",
                "rank": self.rank,
                "alpha": self.alpha,
                "num_experts": self.num_experts,
            }),
        }
        save_file(sd, str(path), metadata=meta)
        logger.info("Ortho-Hydra 保存到: %s", path)

    def load(self, path: str | Path) -> None:
        logger.info("加载 Ortho-Hydra 权重: %s", path)
        sd: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        result = self.load_state_dict(sd, strict=False)
        missing = len(getattr(result, "missing_keys", []))
        unexpected = len(getattr(result, "unexpected_keys", []))
        logger.info("加载 %d 个张量，missing=%d, unexpected=%d", len(sd), missing, unexpected)
