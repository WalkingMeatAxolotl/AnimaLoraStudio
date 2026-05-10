"""T-LoRA adapter for Anima (DiT).

Timestep-Dependent LoRA：根据扩散时间步动态调整有效秩。
- t=1（纯噪声，高噪阶段）→ 低秩（min_rank），限制记忆、防过拟合
- t=0（干净图像，低噪阶段）→ 高秩（max_rank），精细刻画细节

核心机制：sigma_mask [1, max_rank]，前 r 位为 1，其余为 0。
每次 forward 前由 set_mask() 注入，TLoRALinearLayer.forward 用它
乘 down-projection 输出，等效于只激活前 r 个秩维度。

接口与 AnimaLycorisAdapter 对齐（inject / get_params / get_param_groups /
state_dict / save / load / detach），lora_type='tlora' 直接接入 anima_train。

保存格式：safetensors，键名
  lora_unet_{模块路径}.lora_down.weight
  lora_unet_{模块路径}.lora_up.weight
与标准 LoRA safetensors 兼容（ComfyUI 可加载，需配套 T-LoRA AttnProcessor）。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

# 注入目标：Anima Block 内的 Linear 子模块（点路径）
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


# ---------------------------------------------------------------------------
# 核心层
# ---------------------------------------------------------------------------


class TLoRALinearLayer(nn.Module):
    """原始 Linear 的 T-LoRA 封装。

    forward 时：output = original(x) + up(down(x) * mask) * scale
    mask 由外部 set_mask() 在每次 forward 前注入；
    mask=None 时等效于全秩（退化为普通 LoRA）。
    """

    def __init__(self, original: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        in_f, out_f = original.in_features, original.out_features
        self.rank = rank
        self.scale = alpha / rank

        # 冻结原始层（梯度不流入原始权重）
        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)

        self.down = nn.Linear(in_f, rank, bias=False)
        self.up = nn.Linear(rank, out_f, bias=False)

        # T-LoRA 初始化：down ~ N(0, 1/rank)，up = 0（训练开始时 delta=0）
        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)

        self.current_mask: Optional[torch.Tensor] = None  # [1, rank]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original(x)
        mask = self.current_mask
        if mask is None:
            mask = torch.ones(1, self.rank, device=x.device, dtype=x.dtype)
        else:
            mask = mask.to(device=x.device, dtype=x.dtype)
        # down: [*, in] → [*, rank]；乘 mask 动态截断有效维度；up: [*, rank] → [*, out]
        delta = self.up(self.down(x) * mask) * self.scale
        return orig_out + delta


# ---------------------------------------------------------------------------
# 适配器
# ---------------------------------------------------------------------------


class AnimaTLoRAAdapter:
    """T-LoRA 适配器，接口与 AnimaLycorisAdapter 完全对齐。

    inject(model) 后，训练循环每步调用 set_mask(sigma_mask) 再 forward。
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: float = 8.0,
        min_rank: int = 1,
        reg_dims: Optional[dict[str, int]] = None,
        reg_lrs: Optional[dict[str, float]] = None,
    ) -> None:
        self.rank = rank
        self.alpha = alpha
        self.min_rank = max(1, min_rank)
        # per-layer rank / lr：key 为正则表达式，匹配模块全名（如 blocks_0_self_attn_q_proj）
        self.reg_dims: dict[str, int] = dict(reg_dims) if reg_dims else {}
        self.reg_lrs: dict[str, float] = dict(reg_lrs) if reg_lrs else {}
        # AnimaLycorisAdapter 兼容字段
        self.algo = "tlora"
        self.use_lokr = False

        self._tlora_layers: list[TLoRALinearLayer] = []
        # key_name → TLoRALinearLayer，供 state_dict / load 使用
        self._layer_keys: dict[str, TLoRALinearLayer] = {}
        self._module_lrs: dict[str, Optional[float]] = {}
        self._injected_model: Optional[nn.Module] = None

    def _get_reg_dim(self, key: str) -> int:
        """按正则匹配 key，返回对应 rank；无匹配则用全局 rank。"""
        for pat, dim in self.reg_dims.items():
            if re.fullmatch(pat, key):
                return int(dim)
        return self.rank

    def _get_reg_lr(self, key: str) -> Optional[float]:
        """按正则匹配 key，返回对应 lr；无匹配则 None（使用全局 lr）。"""
        for pat, lr in self.reg_lrs.items():
            if re.fullmatch(pat, key):
                return float(lr)
        return None

    # --------------------------------------------------------------- inject

    def inject(self, model: nn.Module) -> dict[str, TLoRALinearLayer]:
        """在 model.blocks[*] 的目标子路径上注入 TLoRALinearLayer。"""
        if not hasattr(model, "blocks"):
            raise RuntimeError("AnimaTLoRAAdapter: model 没有 .blocks，是否加载了正确的 Anima 模型？")

        injected: dict[str, TLoRALinearLayer] = {}
        rank_summary: dict[int, int] = {}

        try:
            ref = next(model.parameters())
            _device, _dtype = ref.device, ref.dtype
        except StopIteration:
            _device, _dtype = None, None

        for block_idx, block in enumerate(model.blocks):
            for subpath in _TARGET_SUBPATHS:
                # 按点路径递归找到目标 Linear
                parts = subpath.split(".")
                parent: nn.Module = block
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr = parts[-1]
                original: nn.Linear = getattr(parent, attr)

                if not isinstance(original, nn.Linear):
                    continue  # 跳过非 Linear（如已注入过）

                # 键名：lora_unet_blocks_{i}_{subpath（.→_）}
                key = f"lora_unet_blocks_{block_idx}_{subpath.replace('.', '_')}"

                # per-layer rank / lr
                mod_rank = self._get_reg_dim(key)
                mod_lr = self._get_reg_lr(key)
                mod_alpha = float(mod_rank)  # alpha 跟随 rank，保持 scale 一致

                tlora_layer = TLoRALinearLayer(original, mod_rank, mod_alpha)

                if _device is not None:
                    tlora_layer.to(device=_device, dtype=_dtype)

                setattr(parent, attr, tlora_layer)

                self._tlora_layers.append(tlora_layer)
                self._layer_keys[key] = tlora_layer
                self._module_lrs[key] = mod_lr
                injected[key] = tlora_layer
                rank_summary[mod_rank] = rank_summary.get(mod_rank, 0) + 1

        self._injected_model = model
        rank_dist = ", ".join(f"r{r}×{c}" for r, c in sorted(rank_summary.items()))
        logger.info(
            f"T-LoRA 注入 {len(injected)} 层（min_rank={self.min_rank}, rank分布: {rank_dist}）"
        )
        return injected

    # --------------------------------------------------------------- mask

    def set_mask(self, sigma_mask: torch.Tensor) -> None:
        """训练/推理每步前调用；sigma_mask 形如 [1, rank]，0/1 二值。"""
        for layer in self._tlora_layers:
            layer.current_mask = sigma_mask

    def get_rank_by_t(self, t: float) -> int:
        """Anima flow matching 约定：t=0 干净，t=1 纯噪声。

        高噪（t→1）→ 低秩（min_rank），防过拟合；
        低噪（t→0）→ 高秩（max_rank），精细细节。
        """
        r = int((1.0 - t) * (self.rank - self.min_rank)) + self.min_rank
        return max(self.min_rank, min(self.rank, r))

    def build_sigma_mask(self, t_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        """根据批次时间步（取均值）构建 sigma_mask [1, rank]。"""
        t_mean = float(t_batch.mean().item())
        r = self.get_rank_by_t(t_mean)
        mask = torch.zeros(1, self.rank, device=device)
        mask[:, :r] = 1.0
        return mask

    # --------------------------------------------------------------- detach

    def detach(self) -> bool:
        """撤销注入：把 TLoRALinearLayer 替换回原始 Linear。"""
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
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    attr = parts[-1]
                    layer = getattr(parent, attr)
                    if isinstance(layer, TLoRALinearLayer):
                        setattr(parent, attr, layer.original)
        except Exception as exc:
            logger.warning(f"T-LoRA detach 部分失败: {exc}")
            return False

        self._tlora_layers.clear()
        self._layer_keys.clear()
        self._injected_model = None
        return True

    # --------------------------------------------------------------- params

    def get_params(self) -> list[nn.Parameter]:
        params = []
        for layer in self._tlora_layers:
            params.extend([layer.down.weight, layer.up.weight])
        return params

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        if not self._module_lrs or not any(v is not None for v in self._module_lrs.values()):
            return [{"params": self.get_params(), "weight_decay": weight_decay}]

        # per-layer lr：按 custom_lr 分组，None 归入默认组
        groups: dict[Optional[float], list[nn.Parameter]] = {}
        for key, layer in self._layer_keys.items():
            lr = self._module_lrs.get(key)
            groups.setdefault(lr, []).extend([layer.down.weight, layer.up.weight])

        result = []
        for lr, params in groups.items():
            g: dict = {"params": params, "weight_decay": weight_decay}
            if lr is not None:
                g["lr"] = lr
            result.append(g)
        return result

    # --------------------------------------------------------------- state I/O

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        for key, layer in self._layer_keys.items():
            sd[f"{key}.lora_down.weight"] = layer.down.weight.data.clone()
            sd[f"{key}.lora_up.weight"] = layer.up.weight.data.clone()
        return sd

    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True):
        missing, unexpected = [], []
        for key, layer in self._layer_keys.items():
            dk = f"{key}.lora_down.weight"
            uk = f"{key}.lora_up.weight"
            if dk in sd:
                layer.down.weight.data.copy_(sd[dk])
            elif strict:
                missing.append(dk)
            if uk in sd:
                layer.up.weight.data.copy_(sd[uk])
            elif strict:
                missing.append(uk)
        for k in sd:
            if k not in {f"{key}.lora_down.weight" for key in self._layer_keys} | \
                        {f"{key}.lora_up.weight" for key in self._layer_keys}:
                unexpected.append(k)
        return type("Result", (), {"missing_keys": missing, "unexpected_keys": unexpected})()

    def save(self, path: str | Path) -> None:
        sd = self.state_dict()
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": "tlora",
            "ss_network_args": json.dumps({
                "algo": "tlora",
                "rank": self.rank,
                "min_rank": self.min_rank,
                "alpha": self.alpha,
            }),
        }
        save_file(sd, str(path), metadata=meta)
        logger.info(f"T-LoRA 保存到: {path}")

    def load(self, path: str | Path) -> None:
        logger.info(f"加载 T-LoRA 权重: {path}")
        sd: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        result = self.load_state_dict(sd, strict=False)
        missing = len(getattr(result, "missing_keys", []))
        unexpected = len(getattr(result, "unexpected_keys", []))
        logger.info(f"加载 {len(sd)} 个张量，missing={missing}, unexpected={unexpected}")
