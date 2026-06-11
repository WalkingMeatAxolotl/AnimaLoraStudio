"""OrthoLoRA adapter for Anima training.

This is the PSOFT-style OrthoLoRA path used by the T-LoRA stack: train
Cayley-rotated SVD bases, optionally apply a timestep rank mask, and save a
plain LoRA checkpoint for inference.
"""
from __future__ import annotations

import json
import logging
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from utils.lokr_preset import ANIMA_PRESET

logger = logging.getLogger(__name__)


def _lora_name(module_name: str) -> str:
    return "lora_unet_" + module_name.replace(".", "_")


def _matches_anima_preset(module_name: str) -> bool:
    if any(fnmatch(module_name, pat) for pat in ANIMA_PRESET.get("exclude_name", [])):
        return False
    return any(fnmatch(module_name, pat) for pat in ANIMA_PRESET.get("target_name", []))


def _split_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class OrthoLoRALinear(nn.Module):
    """Linear wrapper with Cayley-rotated SVD bases.

    Trainable parameters are ``S_p``, ``S_q`` and ``lambda_layer``. The wrapped
    base Linear stays frozen and contributes the original forward path.
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Linear,
        *,
        rank: int,
        alpha: float,
        multiplier: float = 1.0,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, nn.Linear):
            raise TypeError(f"OrthoLoRALinear only supports nn.Linear, got {type(org_module)!r}")

        self.lora_name = lora_name
        self.org_module = org_module
        self.org_module.requires_grad_(False)

        out_dim, in_dim = org_module.weight.shape
        self.lora_dim = max(1, min(int(rank), int(out_dim), int(in_dim)))
        alpha = float(self.lora_dim if alpha is None or alpha == 0 else alpha)
        self.scale = alpha / self.lora_dim
        self.multiplier = float(multiplier)
        self.dropout = float(dropout or 0.0)
        self.rank_dropout = float(rank_dropout or 0.0)
        self.module_dropout = float(module_dropout or 0.0)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        device = org_module.weight.device
        basis_dtype = org_module.weight.dtype if org_module.weight.dtype in (
            torch.float16,
            torch.bfloat16,
        ) else torch.float32

        with torch.no_grad():
            W = org_module.weight.detach().float()
            q = min(self.lora_dim + 6, min(W.shape))
            try:
                U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
                P_init = U[:, : self.lora_dim].contiguous()
                Q_init = V[:, : self.lora_dim].T.contiguous()
            except RuntimeError:
                U, _S_vals, Vh = torch.linalg.svd(W, full_matrices=False)
                P_init = U[:, : self.lora_dim].contiguous()
                Q_init = Vh[: self.lora_dim, :].contiguous()

        self.register_buffer("P_basis", P_init.to(device=device, dtype=basis_dtype))
        self.register_buffer("Q_basis", Q_init.to(device=device, dtype=basis_dtype))
        self.S_p = nn.Parameter(torch.zeros(self.lora_dim, self.lora_dim, device=device))
        self.S_q = nn.Parameter(torch.zeros(self.lora_dim, self.lora_dim, device=device))
        self.lambda_layer = nn.Parameter(torch.zeros(1, self.lora_dim, device=device))
        self.register_buffer(
            "_eye_r",
            torch.eye(self.lora_dim, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_timestep_mask",
            torch.ones(1, self.lora_dim, dtype=torch.float32, device=device),
            persistent=False,
        )

    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        A = S.float() - S.float().T
        eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return torch.linalg.solve(eye + A, eye - A)

    def _rank_dropout(self, lx: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.rank_dropout > 0.0 and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if lx.dim() == 3:
                mask = mask.unsqueeze(1)
            elif lx.dim() == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            lx = lx * mask.to(dtype=lx.dtype)
            return lx, self.scale * (1.0 / max(1e-6, 1.0 - self.rank_dropout))
        return lx, self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_forwarded = self.org_module(x)
        # 用 torch RNG（而非 python random）：grad checkpoint 重算只恢复 torch
        # RNG 状态，python random 会让重算分支与首次前向不一致 → recompute mismatch。
        if self.module_dropout > 0.0 and self.training and bool(torch.rand(()) < self.module_dropout):
            return org_forwarded

        work = self.P_basis.dtype
        skew = torch.stack([self.S_q, self.S_p])
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)
        R_q = R[0].to(work)
        R_p = R[1].to(work)

        Q_eff = R_q @ self.Q_basis
        lx = F.linear(x.to(work), Q_eff)
        lx = lx * self.lambda_layer.to(work) * self._timestep_mask.to(
            device=lx.device,
            dtype=work,
        )
        if self.dropout > 0.0 and self.training:
            lx = F.dropout(lx, p=self.dropout)
        lx, scale = self._rank_dropout(lx)

        P_eff = self.P_basis @ R_p
        out = F.linear(lx, P_eff)
        return org_forwarded + (out * self.multiplier * scale).to(org_forwarded.dtype)

    def distilled_lora_state(self, dtype: Optional[torch.dtype] = None) -> dict[str, torch.Tensor]:
        save_dtype = dtype or self.P_basis.dtype
        R_p = self._cayley(self.S_p.detach())
        R_q = self._cayley(self.S_q.detach())
        P_eff = self.P_basis.float() @ R_p
        Q_eff = R_q @ self.Q_basis.float()
        lam = self.lambda_layer.detach().squeeze(0).float()
        lam_abs = lam.abs()
        lam_sign = lam.sign()
        lam_sqrt = lam_abs.sqrt()
        lora_up = (P_eff * (lam_sqrt * lam_sign).unsqueeze(0)).to(save_dtype).cpu().contiguous()
        lora_down = (Q_eff * lam_sqrt.unsqueeze(1)).to(save_dtype).cpu().contiguous()
        return {
            f"{self.lora_name}.lora_down.weight": lora_down,
            f"{self.lora_name}.lora_up.weight": lora_up,
            f"{self.lora_name}.alpha": self.alpha.detach().cpu().float(),
        }

    def project_plain_lora_(self, lora_down: torch.Tensor, lora_up: torch.Tensor) -> None:
        """Best-effort plain-LoRA load by projecting into the frozen SVD basis."""
        delta = lora_up.float() @ lora_down.float()
        P = self.P_basis.float()
        Q = self.Q_basis.float()
        lam = torch.diagonal(P.T @ delta @ Q.T, 0).view(1, -1)
        with torch.no_grad():
            self.S_p.zero_()
            self.S_q.zero_()
            self.lambda_layer.copy_(lam.to(self.lambda_layer))


class OrthoLoRAAdapter:
    """AdapterProtocol implementation for OrthoLoRA and T-LoRA+Ortho."""

    def __init__(
        self,
        *,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_timestep_mask: bool = False,
        tlora_min_rank: int = 1,
        tlora_alpha_rank_scale: float = 1.0,
    ) -> None:
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout or 0.0)
        self.rank_dropout = float(rank_dropout or 0.0)
        self.module_dropout = float(module_dropout or 0.0)
        self.use_timestep_mask = bool(use_timestep_mask)
        self.tlora_min_rank = max(1, int(tlora_min_rank))
        self.tlora_alpha_rank_scale = max(0.0, float(tlora_alpha_rank_scale))
        self.loras = nn.ModuleList()
        self.use_lokr = False
        self._tlora_mask: Optional[torch.Tensor] = None
        self._tlora_arange: Optional[torch.Tensor] = None

    def inject(self, model: nn.Module) -> dict[str, nn.Module]:
        replacements: list[tuple[str, nn.Module, str, nn.Linear]] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and _matches_anima_preset(name):
                parent, child_name = _split_parent(model, name)
                replacements.append((name, parent, child_name, module))

        for name, parent, child_name, module in replacements:
            layer = OrthoLoRALinear(
                _lora_name(name),
                module,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                rank_dropout=self.rank_dropout,
                module_dropout=self.module_dropout,
            )
            setattr(parent, child_name, layer)
            self.loras.append(layer)

        logger.info(
            "注入 %s 到 %s 层（OrthoLoRA, save=baked LoRA）",
            "T-LORA+ORTHO" if self.use_timestep_mask else "ORTHO",
            len(self.loras),
        )
        return {layer.lora_name: layer for layer in self.loras}

    def _set_timestep_mask(self, sigma_t: torch.Tensor) -> None:
        if not self.loras:
            return
        max_rank = max(layer.lora_dim for layer in self.loras)
        device = sigma_t.device
        if self._tlora_mask is None or self._tlora_mask.device != device or self._tlora_mask.shape[1] != max_rank:
            self._tlora_mask = torch.ones(1, max_rank, device=device)
            self._tlora_arange = torch.arange(max_rank, device=device)
        t = sigma_t.float().mean().clamp(min=0.0, max=1.0)
        frac = (1.0 - t).pow(self.tlora_alpha_rank_scale)
        for layer in self.loras:
            min_rank = min(self.tlora_min_rank, layer.lora_dim)
            active_rank = frac * (layer.lora_dim - min_rank) + min_rank
            active_rank = active_rank.clamp(min=float(min_rank), max=float(layer.lora_dim))
            mask = (self._tlora_arange[: layer.lora_dim] < active_rank).to(self._tlora_mask.dtype)
            layer._timestep_mask = mask.view(1, -1).to(device=device)

    def clear_timestep_mask(self) -> None:
        for layer in self.loras:
            layer._timestep_mask.fill_(1)

    def on_step_begin(self, ctx) -> None:
        if self.use_timestep_mask:
            self._set_timestep_mask(ctx.sigma_t)
        return None

    def regularization_loss(self, ctx) -> None:
        return None

    def excludes_weight_decay(self, param_name: str) -> bool:
        return False

    def get_params(self) -> list[nn.Parameter]:
        return [p for layer in self.loras for p in layer.parameters() if p.requires_grad]

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        return [{"params": self.get_params(), "weight_decay": weight_decay}]

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        for layer in self.loras:
            prefix = layer.lora_name
            sd[f"{prefix}.S_p"] = layer.S_p.detach().cpu()
            sd[f"{prefix}.S_q"] = layer.S_q.detach().cpu()
            sd[f"{prefix}.P_basis"] = layer.P_basis.detach().cpu()
            sd[f"{prefix}.Q_basis"] = layer.Q_basis.detach().cpu()
            sd[f"{prefix}.lambda_layer"] = layer.lambda_layer.detach().cpu()
            sd[f"{prefix}.alpha"] = layer.alpha.detach().cpu()
        return sd

    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True) -> Any:
        missing: list[str] = []
        unexpected = set(sd)
        projected_layers = 0
        for layer in self.loras:
            prefix = layer.lora_name
            runtime_keys = {
                "S_p": layer.S_p,
                "S_q": layer.S_q,
                "P_basis": layer.P_basis,
                "Q_basis": layer.Q_basis,
                "lambda_layer": layer.lambda_layer,
            }
            has_runtime = any(f"{prefix}.{suffix}" in sd for suffix in runtime_keys)
            if has_runtime:
                for suffix, target in runtime_keys.items():
                    key = f"{prefix}.{suffix}"
                    if key not in sd:
                        missing.append(key)
                        continue
                    with torch.no_grad():
                        target.copy_(sd[key].to(device=target.device, dtype=target.dtype))
                    unexpected.discard(key)
                unexpected.discard(f"{prefix}.alpha")
                continue

            down_key = f"{prefix}.lora_down.weight"
            up_key = f"{prefix}.lora_up.weight"
            if down_key in sd and up_key in sd:
                layer.project_plain_lora_(sd[down_key], sd[up_key])
                projected_layers += 1
                unexpected.discard(down_key)
                unexpected.discard(up_key)
                unexpected.discard(f"{prefix}.alpha")
                continue

            missing.extend(f"{prefix}.{suffix}" for suffix in runtime_keys)

        if projected_layers:
            logger.warning(
                "OrthoLoRA: %s 层从 plain LoRA 投影恢复（仅取冻结 SVD 基上的对角分量，"
                "旋转信息丢失）—— 这是近似续训，非完整恢复。无损断点续训请用训练 state "
                "(.pt) 而非蒸馏后的 LoRA 文件。", projected_layers,
            )

        if strict and (missing or unexpected):
            raise RuntimeError(f"OrthoLoRA load_state_dict mismatch: missing={missing}, unexpected={sorted(unexpected)}")
        return SimpleNamespace(missing_keys=missing, unexpected_keys=sorted(unexpected))

    def save(self, path: str | Path) -> None:
        sd: dict[str, torch.Tensor] = {}
        for layer in self.loras:
            sd.update(layer.distilled_lora_state())
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": "lycoris.kohya",
            "ss_network_args": json.dumps({
                "algo": "lora",
                "source_algo": "tlora_ortho" if self.use_timestep_mask else "ortho",
                "factor": 8,
                "preset": "anima_full",
                "dropout": self.dropout,
                "rank_dropout": self.rank_dropout,
                "module_dropout": self.module_dropout,
                "weight_decompose": False,
                "rs_lora": False,
            }),
        }
        save_file(sd, str(path), metadata=meta)
        logger.info("OrthoLoRA 保存到: %s (baked as plain LoRA)", path)

    def load(self, path: str | Path) -> None:
        sd: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        self.load_state_dict(sd, strict=False)
