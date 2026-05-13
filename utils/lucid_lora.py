from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LucidLoRALinear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        rank: int,
        alpha: float,
        sig_type: str = "last",
    ) -> None:
        super().__init__()
        in_f, out_f = original.in_features, original.out_features
        self.rank = max(1, int(rank))
        self.scale = alpha / self.rank

        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)

        self.down = nn.Linear(in_f, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_f, bias=False)
        nn.init.zeros_(self.down.weight)

        if sig_type == "random":
            nn.init.normal_(self.up.weight, std=1.0 / self.rank)
        else:
            q = min(self.rank + 4, min(out_f, in_f))
            W = original.weight.data.float()
            U, _S, _Vh = torch.svd_lowrank(W, q=q, niter=2)
            if sig_type == "last":
                vecs = U[:, -self.rank:].contiguous()
            else:
                vecs = U[:, :self.rank].contiguous()
            self.up.weight.data.copy_(vecs.to(self.up.weight.dtype))

        self.current_mask: Optional[torch.Tensor] = None
        self.multiplier = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original(x)
        mask = self.current_mask
        if mask is None:
            mask = torch.ones(1, self.rank, device=x.device, dtype=x.dtype)
        else:
            mask = mask.to(device=x.device, dtype=x.dtype)
        delta = F.linear(
            F.linear(x, self.down.weight) * mask,
            self.up.weight,
        ) * self.scale
        return orig_out + delta * self.multiplier

    @torch.enable_grad()
    def aux_loss(
        self,
        ortho_reg: float,
        mag_reg: float,
        mag_amplify: float,
    ) -> torch.Tensor:
        a_norms = self.down.weight.norm(dim=1)
        b_norms = self.up.weight.norm(dim=0)
        rank_mags = a_norms * b_norms

        B = self.up.weight
        BtB = B.T @ B
        I = torch.eye(self.rank, device=B.device, dtype=B.dtype)
        ortho_loss = ortho_reg * ((BtB - I).pow(2).mean())

        rank_mags_norm = rank_mags / (rank_mags.mean().clamp(min=1e-8))
        soft_mask = torch.sigmoid(mag_amplify * (rank_mags_norm - 1.0))
        sparsity_loss = mag_reg * rank_mags.mean()
        amplify_loss = -mag_reg * (soft_mask * rank_mags).mean()

        return ortho_loss + sparsity_loss + amplify_loss


class LucidLoKrLinear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        factor: int | None = None,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        in_f, out_f = original.in_features, original.out_features
        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)

        factor = max(2, int(factor or 8))
        self.out_factor = self._choose_factor(out_f, factor)
        self.in_factor = self._choose_factor(in_f, factor)
        self.out_inner = out_f // self.out_factor
        self.in_inner = in_f // self.in_factor
        self.scale = alpha / float(max(1, self.out_factor * self.in_factor))

        self.lokr_w1 = nn.Parameter(torch.empty(self.out_factor, self.in_factor))
        self.lokr_w2_b = nn.Parameter(torch.empty(self.out_inner, self.in_inner))
        nn.init.normal_(self.lokr_w1, std=0.1)
        nn.init.zeros_(self.lokr_w2_b)
        self.multiplier = 1.0

    def _choose_factor(self, dim: int, target: int) -> int:
        target = min(target, dim)
        for value in range(target, 1, -1):
            if dim % value == 0:
                return value
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_weight = torch.kron(self.lokr_w1, self.lokr_w2_b).to(dtype=x.dtype) * self.scale
        return self.original(x) + F.linear(x, delta_weight) * self.multiplier


class AnimaLucidLoRAAdapter:
    def __init__(
        self,
        rank: int = 32,
        alpha: float = 16.0,
        min_rank: int | None = None,
        min_rank_ratio: float | None = 0.1,
        qk_rank_ratio: float = 0.25,
        lora_plus_ratio: float = 16.0,
        alpha_rank_scale: float = 2.0,
        sig_type: str = "last",
        ortho_reg: float = 0.01,
        mag_reg: float = 0.001,
        mag_amplify: float = 2.0,
        aux_loss_weight: float = 1.0,
        aux_warmup_ratio: float = 0.1,
        export_mode: str = "lycoris_compat",
        use_lokr_ffn: bool = False,
        lokr_factor: int | None = None,
    ) -> None:
        self.rank = max(1, int(rank))
        self.alpha = alpha
        if min_rank_ratio is None:
            min_rank_ratio = (float(min_rank) / float(self.rank)) if min_rank is not None else 0.1
        self.min_rank_ratio = max(0.0, min(1.0, float(min_rank_ratio)))
        if min_rank is None:
            self.min_rank = self._min_rank_for(self.rank)
        else:
            self.min_rank = max(1, min(self.rank, int(min_rank)))
            self.min_rank_ratio = self.min_rank / float(self.rank)
        self.qk_rank_ratio = max(0.0, min(1.0, float(qk_rank_ratio)))
        self.lora_plus_ratio = max(0.0, float(lora_plus_ratio))
        self.alpha_rank_scale = max(0.1, float(alpha_rank_scale))
        self.sig_type = sig_type
        self.ortho_reg = float(ortho_reg)
        self.mag_reg = float(mag_reg)
        self.mag_amplify = float(mag_amplify)
        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_warmup_ratio = float(aux_warmup_ratio)
        self.export_mode = export_mode if export_mode in {"native", "lycoris_compat"} else "lycoris_compat"
        self.use_lokr_ffn = bool(use_lokr_ffn)
        self.lokr_factor = max(2, int(lokr_factor or 8))

        self.algo = "lucid"
        self.use_lokr = self.use_lokr_ffn
        self.multiplier = 1.0

        self._lucid_layers: list[LucidLoRALinear] = []
        self._lokr_layers: list[LucidLoKrLinear] = []
        self._layer_keys: dict[str, LucidLoRALinear | LucidLoKrLinear] = {}
        self._layer_roles: dict[str, str] = {}
        self._injected_model: Optional[nn.Module] = None

    def _min_rank_for(self, rank: int) -> int:
        return max(1, min(int(rank), int(round(int(rank) * self.min_rank_ratio))))

    def _rank_for_subpath(self, subpath: str) -> int:
        if subpath.endswith("q_proj") or subpath.endswith("k_proj"):
            return max(1, int(round(self.rank * self.qk_rank_ratio)))
        return self.rank

    def _role_for_subpath(self, subpath: str) -> str:
        if subpath.endswith("q_proj") or subpath.endswith("k_proj"):
            return "qk"
        if subpath.startswith("mlp."):
            return "ffn"
        return "value_out"

    def inject(self, model: nn.Module) -> dict[str, LucidLoRALinear | LucidLoKrLinear]:
        if not hasattr(model, "blocks"):
            raise RuntimeError("AnimaLucidLoRAAdapter: model 没有 .blocks，是否加载了正确的 Anima 模型？")
        injected: dict[str, LucidLoRALinear | LucidLoKrLinear] = {}
        rank_count = 0

        try:
            ref = next(model.parameters())
            _device, _dtype = ref.device, ref.dtype
        except StopIteration:
            _device, _dtype = None, None

        for block_idx, block in enumerate(model.blocks):
            for subpath in _TARGET_SUBPATHS:
                parts = subpath.split(".")
                parent: nn.Module = block
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr = parts[-1]
                original: nn.Linear = getattr(parent, attr)

                if not isinstance(original, nn.Linear):
                    continue

                key = f"lora_unet_blocks_{block_idx}_{subpath.replace('.', '_')}"
                role = self._role_for_subpath(subpath)
                if role == "ffn" and self.use_lokr_ffn:
                    layer = LucidLoKrLinear(original, factor=self.lokr_factor, alpha=self.alpha)
                    self._lokr_layers.append(layer)
                else:
                    module_rank = self._rank_for_subpath(subpath)
                    layer = LucidLoRALinear(original, module_rank, float(module_rank), sig_type=self.sig_type)
                    self._lucid_layers.append(layer)
                if _device is not None:
                    layer.to(device=_device, dtype=_dtype)

                setattr(parent, attr, layer)
                self._layer_keys[key] = layer
                self._layer_roles[key] = role
                injected[key] = layer
                rank_count += 1

        self._injected_model = model
        logger.info(
            f"LucidLoRA 注入 {rank_count} 层 "
            f"(rank={self.rank}, qk_rank={self._rank_for_subpath('self_attn.q_proj')}, "
            f"min_rank_ratio={self.min_rank_ratio:.3f}, lora_plus={self.lora_plus_ratio:g}, "
            f"ffn_lokr={self.use_lokr_ffn})"
        )
        return injected

    def set_mask(self, sigma_mask: torch.Tensor) -> None:
        active_fraction = float(sigma_mask.float().mean().clamp(0.0, 1.0).item())
        for layer in self._lucid_layers:
            active_rank = max(
                self._min_rank_for(layer.rank),
                min(layer.rank, int(round(layer.rank * active_fraction))),
            )
            mask = torch.zeros(1, layer.rank, device=sigma_mask.device, dtype=sigma_mask.dtype)
            mask[:, :active_rank] = 1.0
            layer.current_mask = mask
            layer.multiplier = self.multiplier
        for layer in self._lokr_layers:
            layer.multiplier = self.multiplier

    def get_rank_by_t(self, t: float, rank: int | None = None) -> int:
        layer_rank = int(rank or self.rank)
        min_rank = self._min_rank_for(layer_rank)
        frac = (1.0 - t) ** self.alpha_rank_scale
        r = int(frac * (layer_rank - min_rank)) + min_rank
        return max(min_rank, min(layer_rank, r))

    def build_sigma_mask(self, t_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        t_mean = float(t_batch.mean().item())
        r = self.get_rank_by_t(t_mean, self.rank)
        mask = torch.zeros(1, self.rank, device=device)
        mask[:, :r] = 1.0
        return mask

    def compute_aux_loss(self) -> torch.Tensor:
        if not self._lucid_layers:
            return torch.tensor(0.0)

        device = self._lucid_layers[0].up.weight.device
        total = torch.zeros(1, device=device)
        for layer in self._lucid_layers:
            total = total + layer.aux_loss(self.ortho_reg, self.mag_reg, self.mag_amplify)
        return total.squeeze(0)

    def detach(self) -> bool:
        if self._injected_model is None:
            return True
        model = self._injected_model
        if not hasattr(model, "blocks"):
            return False
        try:
            for block in model.blocks:
                for subpath in _TARGET_SUBPATHS:
                    parts = subpath.split(".")
                    parent: nn.Module = block
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    attr = parts[-1]
                    layer = getattr(parent, attr)
                    if isinstance(layer, (LucidLoRALinear, LucidLoKrLinear)):
                        setattr(parent, attr, layer.original)
        except Exception as exc:
            logger.warning(f"LucidLoRA detach 部分失败: {exc}")
            return False
        self._lucid_layers.clear()
        self._lokr_layers.clear()
        self._layer_keys.clear()
        self._layer_roles.clear()
        self._injected_model = None
        return True

    def get_params(self) -> list[nn.Parameter]:
        params = []
        for layer in self._lucid_layers:
            params.extend([layer.down.weight, layer.up.weight])
        for layer in self._lokr_layers:
            params.extend([layer.lokr_w1, layer.lokr_w2_b])
        return params

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        down_params = [layer.down.weight for layer in self._lucid_layers]
        up_params = [layer.up.weight for layer in self._lucid_layers]
        lokr_params: list[nn.Parameter] = []
        for layer in self._lokr_layers:
            lokr_params.extend([layer.lokr_w1, layer.lokr_w2_b])
        groups = [
            {"params": down_params, "weight_decay": weight_decay},
            {"params": up_params, "weight_decay": weight_decay, "_lucid_lora_plus_group": True},
        ]
        if lokr_params:
            groups.append({"params": lokr_params, "weight_decay": weight_decay})
        return groups

    def named_trainable_params(self) -> list[Tuple[str, nn.Parameter]]:
        result = []
        for key, layer in self._layer_keys.items():
            if isinstance(layer, LucidLoRALinear):
                result.append((f"{key}.lora_down", layer.down.weight))
                result.append((f"{key}.lora_up", layer.up.weight))
            else:
                result.append((f"{key}.lokr_w1", layer.lokr_w1))
                result.append((f"{key}.lokr_w2_b", layer.lokr_w2_b))
        return result

    def _copy_tensor(self, target: nn.Parameter, source: torch.Tensor) -> bool:
        if target.ndim != source.ndim:
            return False
        source = source.to(device=target.device, dtype=target.dtype)
        if tuple(target.shape) == tuple(source.shape):
            target.data.copy_(source)
            return True
        slices = tuple(slice(0, min(dst, src)) for dst, src in zip(target.shape, source.shape))
        target.data[slices].copy_(source[slices])
        return True

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        for key, layer in self._layer_keys.items():
            if isinstance(layer, LucidLoRALinear):
                sd[f"{key}.lora_down.weight"] = layer.down.weight.data.clone()
                sd[f"{key}.lora_up.weight"] = layer.up.weight.data.clone()
            else:
                sd[f"{key}.lokr_w1"] = layer.lokr_w1.data.clone()
                sd[f"{key}.lokr_w2_b"] = layer.lokr_w2_b.data.clone()
        return sd

    def _pad_lora_down(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.shape[0] == self.rank:
            return weight.data.clone()
        padded = weight.new_zeros(self.rank, weight.shape[1])
        padded[: weight.shape[0], :] = weight.data
        return padded

    def _pad_lora_up(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.shape[1] == self.rank:
            return weight.data.clone()
        padded = weight.new_zeros(weight.shape[0], self.rank)
        padded[:, : weight.shape[1]] = weight.data
        return padded

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        if self.export_mode == "native":
            return self.state_dict()

        sd: dict[str, torch.Tensor] = {}
        for key, layer in self._layer_keys.items():
            if isinstance(layer, LucidLoRALinear):
                sd[f"{key}.lora_down.weight"] = self._pad_lora_down(layer.down.weight)
                sd[f"{key}.lora_up.weight"] = self._pad_lora_up(layer.up.weight)
            else:
                sd[f"{key}.lokr_w1"] = layer.lokr_w1.data.clone()
                sd[f"{key}.lokr_w2_b"] = layer.lokr_w2_b.data.clone()
        return sd

    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True):
        missing, unexpected = [], []
        expected: set[str] = set()
        for key, layer in self._layer_keys.items():
            if isinstance(layer, LucidLoRALinear):
                dk = f"{key}.lora_down.weight"
                uk = f"{key}.lora_up.weight"
                expected.update([dk, uk])
                if dk in sd:
                    self._copy_tensor(layer.down.weight, sd[dk])
                elif strict:
                    missing.append(dk)
                if uk in sd:
                    self._copy_tensor(layer.up.weight, sd[uk])
                elif strict:
                    missing.append(uk)
            else:
                w1 = f"{key}.lokr_w1"
                w2 = f"{key}.lokr_w2_b"
                expected.update([w1, w2])
                if w1 in sd:
                    self._copy_tensor(layer.lokr_w1, sd[w1])
                elif strict:
                    missing.append(w1)
                if w2 in sd:
                    self._copy_tensor(layer.lokr_w2_b, sd[w2])
                elif strict:
                    missing.append(w2)
        for k in sd:
            if k not in expected:
                unexpected.append(k)
        return type("Result", (), {"missing_keys": missing, "unexpected_keys": unexpected})()

    def _metadata(self) -> dict[str, str]:
        compat_mode = self.export_mode
        args = {
            "algo": "lucid" if compat_mode == "native" else "lora",
            "lucid_algo": "lucid",
            "rank": self.rank,
            "min_rank": self.min_rank,
            "min_rank_ratio": self.min_rank_ratio,
            "qk_rank_ratio": self.qk_rank_ratio,
            "lora_plus_ratio": self.lora_plus_ratio,
            "alpha": self.alpha,
            "alpha_rank_scale": self.alpha_rank_scale,
            "sig_type": self.sig_type,
            "ortho_reg": self.ortho_reg,
            "mag_reg": self.mag_reg,
            "mag_amplify": self.mag_amplify,
            "aux_loss_weight": self.aux_loss_weight,
            "aux_warmup_ratio": self.aux_warmup_ratio,
            "use_lokr_ffn": self.use_lokr_ffn,
            "lokr_factor": self.lokr_factor,
            "compat": compat_mode,
            "preset": "lucid_role_split",
        }
        export_alpha = self.alpha if compat_mode == "native" else float(self.rank)
        args["training_alpha"] = self.alpha
        return {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(export_alpha),
            "ss_network_module": "lucid" if compat_mode == "native" else "lycoris.kohya",
            "ss_network_args": json.dumps(args),
        }

    def save(self, path: str | Path) -> None:
        save_file(self.export_state_dict(), str(path), metadata=self._metadata())
        logger.info(f"LucidLoRA 保存到: {path}")

    def load(self, path: str | Path) -> None:
        logger.info(f"加载 LucidLoRA 权重: {path}")
        sd: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        result = self.load_state_dict(sd, strict=False)
        missing = len(getattr(result, "missing_keys", []))
        unexpected = len(getattr(result, "unexpected_keys", []))
        logger.info(f"加载 {len(sd)} 个张量，missing={missing}, unexpected={unexpected}")
