"""Riemannion 用的固定秩 LoRA adapter（POC）。

ΔW 以 X = A_L · Bᵀ 表示（A_L 列正交），配合 utils/riemannion_optimizer.py 在固定秩
流形上优化（arxiv 2507.12142）。前向带一路恒为 0 的 Z1 分支：

    y = W·x + A_L·(Bᵀ·x) + Z1·(B_Rᵀ·x)

Z1 不改变输出，只为一次反向同时拿到 ∇Z1 = G·B_R 与 ∇B = Gᵀ·A_L（论文 §6
gradient trick）。代价是 adapter 前向/反向约为同秩 LoRA 的 2 倍。

初始化（本实现取舍，论文用 LOI）：A_L / B_R 取随机正交基、B=0 → ΔW(0)=0，
第 0 步输出与底模一致；首步把梯度投影到这对随机子空间的切空间后回缩到秩 r。

保存：导出标准 LoRA（lora_up / lora_down / alpha=r，scale=1），对 B 做 QR 后把
核矩阵按奇异值开方对半分给 up / down，推理端无需任何改动。
"""

from __future__ import annotations

import json
import logging
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from studio.infrastructure.log_messages import msg
from utils.ortho_adapter import _lora_name, _matches_preset, _split_parent

logger = logging.getLogger(__name__)


def _random_orthonormal(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(rows, cols, generator=generator))
    return q


class RiemannLoRALinear(nn.Module):
    """nn.Linear 包装：ΔW = A_L·Bᵀ，可训练参数为 lora_b（B）与 lora_z1（恒 0）。"""

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Linear,
        *,
        rank: int,
        generator: torch.Generator,
        multiplier: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, nn.Linear):
            raise TypeError(f"RiemannLoRALinear only supports nn.Linear, got {type(org_module)!r}")
        self.lora_name = lora_name
        self.org_module = org_module
        self.org_module.requires_grad_(False)
        self.multiplier = float(multiplier)

        out_dim, in_dim = org_module.weight.shape
        self.lora_dim = max(1, min(int(rank), int(out_dim), int(in_dim)))
        # block swap 下被换出块的底模权重此时在 CPU（pinned），adapter 参数与正交基
        # 必须建在训练设备上（block swap 只搬冻结底模权重，不管它们）
        device = device if device is not None else org_module.weight.device
        r = self.lora_dim
        self.register_buffer("a_l", _random_orthonormal(out_dim, r, generator).to(device))
        self.register_buffer("b_r", _random_orthonormal(in_dim, r, generator).to(device))
        self.lora_b = nn.Parameter(torch.zeros(in_dim, r, device=device))
        self.lora_z1 = nn.Parameter(torch.zeros(out_dim, r, device=device))
        # optimizer 靠这两个标记找回所属层（weakref 避免参数↔模块循环引用）
        self.lora_b._riemann_role = "b"
        self.lora_z1._riemann_role = "z1"
        self.lora_b._riemann_layer = weakref.ref(self)
        self.lora_z1._riemann_layer = weakref.ref(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_forwarded = self.org_module(x)
        work = x.dtype
        h_b = F.linear(x, self.lora_b.mT.to(work))
        h_r = F.linear(x, self.b_r.mT.to(work))
        out = F.linear(h_b, self.a_l.to(work)) + F.linear(h_r, self.lora_z1.to(work))
        return org_forwarded + (out * self.multiplier).to(org_forwarded.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        return self.a_l.float() @ self.lora_b.float().mT

    @torch.no_grad()
    def distilled_lora_state(self, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
        """ΔW = A_L·Bᵀ = A_L·Rᵀ·Qᵀ（B=QR）；Rᵀ=U S Vᵀ → up=A_L U √S，down=√S Vᵀ Qᵀ。"""
        q, r = torch.linalg.qr(self.lora_b.float(), mode="reduced")
        u, s, vh = torch.linalg.svd(r.mT)
        root = s.sqrt()
        lora_up = (self.a_l.float() @ u) * root.unsqueeze(0)
        lora_down = root.unsqueeze(1) * (vh @ q.mT)
        return {
            f"{self.lora_name}.lora_down.weight": lora_down.to(dtype).cpu().contiguous(),
            f"{self.lora_name}.lora_up.weight": lora_up.to(dtype).cpu().contiguous(),
            f"{self.lora_name}.alpha": torch.tensor(float(self.lora_dim)),
        }

    @torch.no_grad()
    def load_plain_lora_(self, lora_down: torch.Tensor, lora_up: torch.Tensor, scale: float) -> None:
        """从标准 LoRA 精确还原：ΔW = s·up·down，up=QR → A_L=Q，B=(s·R·down)ᵀ。"""
        q, r = torch.linalg.qr(lora_up.float(), mode="reduced")
        b = (scale * r @ lora_down.float()).mT
        b_r, _ = torch.linalg.qr(b, mode="reduced")
        self.a_l.copy_(q.to(self.a_l))
        self.lora_b.copy_(b.to(self.lora_b))
        self.b_r.copy_(b_r.to(self.b_r))


class RiemannLoRAAdapter:
    """AdapterProtocol 实现：固定秩 LoRA，仅与 optimizer_type=riemannion 搭配。"""

    def __init__(self, *, rank: int = 32, preset: dict | None = None, seed: int = 0) -> None:
        self.rank = int(rank)
        self._preset = preset
        self._seed = int(seed)
        self.loras = nn.ModuleList()
        self.use_lokr = False

    def inject(self, model: nn.Module) -> dict[str, nn.Module]:
        generator = torch.Generator().manual_seed(self._seed)
        # 与 LycorisAdapter 同口径：按模型首个参数推断训练设备（block swap 换出的尾部
        # 块在 CPU，不能逐层跟随底模权重的设备）
        device = next(model.parameters()).device
        replacements = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and _matches_preset(name, self._preset or {}):
                parent, child_name = _split_parent(model, name)
                replacements.append((name, parent, child_name, module))
        for name, parent, child_name, module in replacements:
            layer = RiemannLoRALinear(
                _lora_name(name), module, rank=self.rank, generator=generator, device=device,
            )
            setattr(parent, child_name, layer)
            self.loras.append(layer)
        logger.info(msg("lora.injected", algo="RIEMANN", n=len(self.loras), detail="save=baked LoRA"))
        return {layer.lora_name: layer for layer in self.loras}

    def on_step_begin(self, ctx) -> None:
        return None

    def regularization_loss(self, ctx) -> None:
        return None

    def excludes_weight_decay(self, param_name: str) -> bool:
        return False

    def get_params(self) -> list[nn.Parameter]:
        return [p for layer in self.loras for p in (layer.lora_b, layer.lora_z1)]

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        return [{"params": self.get_params(), "weight_decay": weight_decay}]

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        for layer in self.loras:
            sd[f"{layer.lora_name}.a_l"] = layer.a_l.detach().cpu()
            sd[f"{layer.lora_name}.b_r"] = layer.b_r.detach().cpu()
            sd[f"{layer.lora_name}.lora_b"] = layer.lora_b.detach().cpu()
        return sd

    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True) -> Any:
        missing: list[str] = []
        unexpected = set(sd)
        for layer in self.loras:
            prefix = layer.lora_name
            if f"{prefix}.a_l" in sd:
                with torch.no_grad():
                    for key, target in (("a_l", layer.a_l), ("b_r", layer.b_r), ("lora_b", layer.lora_b)):
                        target.copy_(sd[f"{prefix}.{key}"].to(target))
                        unexpected.discard(f"{prefix}.{key}")
            elif f"{prefix}.lora_up.weight" in sd:
                up = sd[f"{prefix}.lora_up.weight"]
                down = sd[f"{prefix}.lora_down.weight"]
                alpha = float(sd.get(f"{prefix}.alpha", torch.tensor(float(up.shape[1]))))
                layer.load_plain_lora_(down.to(layer.a_l.device), up.to(layer.a_l.device), alpha / up.shape[1])
                for suffix in ("lora_up.weight", "lora_down.weight", "alpha"):
                    unexpected.discard(f"{prefix}.{suffix}")
            else:
                missing.append(prefix)
        if strict and (missing or unexpected):
            raise RuntimeError(f"RiemannLoRA load_state_dict mismatch: missing={missing}, unexpected={sorted(unexpected)}")
        return SimpleNamespace(missing_keys=missing, unexpected_keys=sorted(unexpected))

    def save(self, path: str | Path) -> None:
        sd: dict[str, torch.Tensor] = {}
        for layer in self.loras:
            sd.update(layer.distilled_lora_state())
        ss_args = {"algo": "lora", "source_algo": "riemannion", "weight_decompose": False, "rs_lora": False}
        ss_args.update(getattr(self, "metadata_extra", None) or {})
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(float(self.rank)),
            "ss_network_module": "lycoris.kohya",
            "ss_network_args": json.dumps(ss_args),
        }
        save_file(sd, str(path), metadata=meta)
        logger.info(msg("lora.saved_riemann", path=path))

    def load(self, path: str | Path) -> None:
        """热启动（resume_lora）：接受本 adapter 的续训键或标准 LoRA 键（lora_up/down）。

        一层都没对上（如 LoKr / LoHa 文件）时报错，而不是静默从零开始训练。
        """
        with safe_open(str(path), framework="pt", device="cpu") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
        result = self.load_state_dict(sd, strict=False)
        if self.loras and len(result.missing_keys) == len(self.loras):
            raise RuntimeError(
                f"resume_lora 与 Riemann LoRA 不匹配：{path} 中没有任何一层的 "
                "lora_up / lora_down 权重（Riemann LoRA 只能从普通 LoRA 或自身续训文件热启动）"
            )
        if result.missing_keys:
            logger.warning(
                "resume_lora matched %d/%d layers; unmatched layers start from zero: %s",
                len(self.loras) - len(result.missing_keys), len(self.loras),
                result.missing_keys[:5],
            )
