"""StyleK-LoRA adapter for Anima (DiT).

单 LoRA 训练，最大化风格特征的集中程度，使学到的权重天然符合 K-LoRA 的
Top-K 分布假设。三个核心机制协同工作：

  1. up (B) = SVD 初始化（W₀ 的左奇异向量），down (A) = 零初始化
       ΔW = up @ down = 0 at step 0，不改变预训练输出。
       B 的列从正交基出发，配合持续正交正则保持 rank 利用率。

  2. T-LoRA 风格时间步 rank schedule（与 AnimaTLoRAAdapter 实现相同）
       alpha_rank_scale > 1 → 低噪声（风格段）获得更多 rank 容量。
       每步 forward 前由 set_mask() 注入 sigma_mask [1, rank]。

  3. Rank-component 辅助 loss（不实体化 W，代价 O(rank)）：
       ortho_loss   — (B^T B - I)^2 均值，防 rank 分量退化为同一方向
       bimodal_loss — 小 rank 分量被 L1 压缩，大分量受保护：
         sparsity = +mag_reg * rank_mags.mean()
         amplify  = -mag_reg * (soft_mask * rank_mags).mean()
         net: rank 幅值分布向"少数主导"极化，契合 K-LoRA Top-K 假设

保存格式：safetensors，键名
  lora_unet_{模块路径}.lora_down.weight
  lora_unet_{模块路径}.lora_up.weight
与标准 LoRA 兼容（ComfyUI 可加载）。

注意：StyleK 的 lora_up (B) 做 SVD 初始化、lora_down (A) 零初始化，
与 T-LoRA（down=SVD / up=0）方向相反；两者 ΔW=0 at init 均成立。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

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


# ---------------------------------------------------------------------------
# 核心层
# ---------------------------------------------------------------------------


class StyleKLoRALinear(nn.Module):
    """StyleK-LoRA 单层：带 B=SVD 初始化和 rank-component aux loss 的 LoRA。

    forward:  output = original(x) + up(down(x) * mask) * scale
    aux_loss: 返回 (ortho_loss + bimodal_loss)，不含全局权重，由 adapter 累加后加权。
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int,
        alpha: float,
        sig_type: str = "last",
    ) -> None:
        super().__init__()
        in_f, out_f = original.in_features, original.out_features
        self.rank = rank
        self.scale = alpha / rank

        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)

        # down (A): (rank, in_f)  —  zero-init，保证 ΔW=0 at step 0
        self.down = nn.Linear(in_f, rank, bias=False)
        # up   (B): (out_f, rank) —  SVD-init，保证正交起点
        self.up = nn.Linear(rank, out_f, bias=False)

        nn.init.zeros_(self.down.weight)

        if sig_type == "random":
            nn.init.normal_(self.up.weight, std=1.0 / rank)
        else:
            # 用 W₀ 的左奇异向量初始化 up.weight (out_f, rank)
            q = min(rank + 4, min(out_f, in_f))
            W = original.weight.data.float()
            U, _S, _Vh = torch.svd_lowrank(W, q=q, niter=2)  # U: (out_f, q)
            if sig_type == "last":
                vecs = U[:, -rank:].contiguous()   # (out_f, rank)，最小奇异方向
            else:  # "first"
                vecs = U[:, :rank].contiguous()    # (out_f, rank)，最大奇异方向
            self.up.weight.data.copy_(vecs.to(self.up.weight.dtype))

        self.current_mask: Optional[torch.Tensor] = None  # [1, rank]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original(x)
        mask = self.current_mask
        if mask is None:
            mask = torch.ones(1, self.rank, device=x.device, dtype=x.dtype)
        else:
            mask = mask.to(device=x.device, dtype=x.dtype)
        # down: [*, in_f] → [*, rank]；mask 动态截断；up: [*, rank] → [*, out_f]
        delta = F.linear(
            F.linear(x, self.down.weight) * mask,
            self.up.weight,
        ) * self.scale
        return orig_out + delta

    @torch.enable_grad()
    def aux_loss(
        self,
        ortho_reg: float,
        mag_reg: float,
        mag_amplify: float,
    ) -> torch.Tensor:
        """Rank-component 辅助 loss，不实体化完整权重矩阵 W。

        代价：O(out_f * rank) ortho + O(rank) bimodal，比 O(out_f * in_f) 便宜得多。
        """
        # --- rank-component 有效幅值 ---
        # down.weight: (rank, in_f)  → row norms → (rank,)
        # up.weight:   (out_f, rank) → col norms → (rank,)
        a_norms = self.down.weight.norm(dim=1)       # (rank,)
        b_norms = self.up.weight.norm(dim=0)         # (rank,)
        rank_mags = a_norms * b_norms                # (rank,) 奇异值代理

        # --- 正交正则 ---
        # up.weight 列应互相正交：B^T B ≈ I
        # up.weight shape (out_f, rank)，B^T = (rank, out_f)
        B = self.up.weight                           # (out_f, rank)
        BtB = B.T @ B                                # (rank, rank)
        I = torch.eye(self.rank, device=B.device, dtype=B.dtype)
        ortho_loss = ortho_reg * ((BtB - I).pow(2).mean())

        # --- 双极化 loss ---
        # sparsity: 推所有 rank 分量向零
        # amplify:  保护大分量（soft_mask≈1 时 amplify≈-sparsity，两者抵消）
        # 净效果：小分量被压缩，大分量相对受保护
        rank_mags_norm = rank_mags / (rank_mags.mean().clamp(min=1e-8))
        soft_mask = torch.sigmoid(mag_amplify * (rank_mags_norm - 1.0))
        sparsity_loss = mag_reg * rank_mags.mean()
        amplify_loss = -mag_reg * (soft_mask * rank_mags).mean()

        return ortho_loss + sparsity_loss + amplify_loss


# ---------------------------------------------------------------------------
# 适配器
# ---------------------------------------------------------------------------


class AnimaStyleKAdapter:
    """StyleK-LoRA 适配器，接口与 AnimaTLoRAAdapter / AnimaOrthoHydraAdapter 对齐。

    inject(model) 后，训练循环每步：
      1. set_mask(build_sigma_mask(t, device))   — 时间步 rank 调度
      2. forward(...)                             — 正常前向
      3. loss += aux_loss_weight * compute_aux_loss()  — 辅助 loss
    """

    def __init__(
        self,
        rank: int = 32,
        alpha: float = 16.0,
        min_rank: int = 4,
        alpha_rank_scale: float = 2.0,
        sig_type: str = "last",
        ortho_reg: float = 0.01,
        mag_reg: float = 0.001,
        mag_amplify: float = 2.0,
        aux_loss_weight: float = 1.0,
        aux_warmup_ratio: float = 0.1,
    ) -> None:
        self.rank = rank
        self.alpha = alpha
        self.min_rank = max(1, min_rank)
        self.alpha_rank_scale = max(0.1, float(alpha_rank_scale))
        self.sig_type = sig_type
        self.ortho_reg = float(ortho_reg)
        self.mag_reg = float(mag_reg)
        self.mag_amplify = float(mag_amplify)
        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_warmup_ratio = float(aux_warmup_ratio)

        # AnimaLycorisAdapter 兼容字段（训练 loop 检查）
        self.algo = "stylek"
        self.use_lokr = False

        self._style_layers: list[StyleKLoRALinear] = []
        self._layer_keys: dict[str, StyleKLoRALinear] = {}
        self._injected_model: Optional[nn.Module] = None

    # ---------------------------------------------------------------- inject

    def inject(self, model: nn.Module) -> dict[str, StyleKLoRALinear]:
        if not hasattr(model, "blocks"):
            raise RuntimeError(
                "AnimaStyleKAdapter: model 没有 .blocks，是否加载了正确的 Anima 模型？"
            )
        injected: dict[str, StyleKLoRALinear] = {}
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
                mod_alpha = float(self.rank)  # alpha 跟随 rank

                layer = StyleKLoRALinear(original, self.rank, mod_alpha, sig_type=self.sig_type)
                if _device is not None:
                    layer.to(device=_device, dtype=_dtype)

                setattr(parent, attr, layer)
                self._style_layers.append(layer)
                self._layer_keys[key] = layer
                injected[key] = layer
                rank_count += 1

        self._injected_model = model
        logger.info(
            f"StyleK-LoRA 注入 {rank_count} 层 "
            f"(rank={self.rank}, min_rank={self.min_rank}, "
            f"alpha_rank_scale={self.alpha_rank_scale}, sig_type={self.sig_type})"
        )
        return injected

    # ---------------------------------------------------------------- mask / schedule

    def set_mask(self, sigma_mask: torch.Tensor) -> None:
        """每步 forward 前调用；sigma_mask [1, rank]，0/1 二值。"""
        for layer in self._style_layers:
            layer.current_mask = sigma_mask

    def get_rank_by_t(self, t: float) -> int:
        """t=0 干净（风格），t=1 纯噪（结构）；alpha_rank_scale > 1 = 风格偏置。"""
        frac = (1.0 - t) ** self.alpha_rank_scale
        r = int(frac * (self.rank - self.min_rank)) + self.min_rank
        return max(self.min_rank, min(self.rank, r))

    def build_sigma_mask(self, t_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        """根据批次时间步（取均值）构建 sigma_mask [1, rank]。"""
        t_mean = float(t_batch.mean().item())
        r = self.get_rank_by_t(t_mean)
        mask = torch.zeros(1, self.rank, device=device)
        mask[:, :r] = 1.0
        return mask

    # ---------------------------------------------------------------- aux loss

    def compute_aux_loss(self) -> torch.Tensor:
        """遍历所有层，累加 rank-component 辅助 loss（返回未加权总和）。

        训练 loop 示例：
            loss = loss + injector.aux_loss_weight * injector.compute_aux_loss()
        """
        if not self._style_layers:
            return torch.tensor(0.0)

        device = self._style_layers[0].up.weight.device
        total = torch.zeros(1, device=device)
        for layer in self._style_layers:
            total = total + layer.aux_loss(self.ortho_reg, self.mag_reg, self.mag_amplify)
        return total.squeeze(0)

    # ---------------------------------------------------------------- detach

    def detach(self) -> bool:
        """撤销注入：把 StyleKLoRALinear 替换回原始 Linear。"""
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
                    if isinstance(layer, StyleKLoRALinear):
                        setattr(parent, attr, layer.original)
        except Exception as exc:
            logger.warning(f"StyleK-LoRA detach 部分失败: {exc}")
            return False
        self._style_layers.clear()
        self._layer_keys.clear()
        self._injected_model = None
        return True

    # ---------------------------------------------------------------- params

    def get_params(self) -> list[nn.Parameter]:
        params = []
        for layer in self._style_layers:
            params.extend([layer.down.weight, layer.up.weight])
        return params

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        return [{"params": self.get_params(), "weight_decay": weight_decay}]

    def named_trainable_params(self) -> list[Tuple[str, nn.Parameter]]:
        """供 OrthoGrad 使用；名称中含 'lora_down' / 'lora_up'。"""
        result = []
        for key, layer in self._layer_keys.items():
            result.append((f"{key}.lora_down", layer.down.weight))
            result.append((f"{key}.lora_up", layer.up.weight))
        return result

    # ---------------------------------------------------------------- state I/O

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        for key, layer in self._layer_keys.items():
            sd[f"{key}.lora_down.weight"] = layer.down.weight.data.clone()
            sd[f"{key}.lora_up.weight"] = layer.up.weight.data.clone()
        return sd

    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True):
        missing, unexpected = [], []
        expected_down = {f"{k}.lora_down.weight" for k in self._layer_keys}
        expected_up = {f"{k}.lora_up.weight" for k in self._layer_keys}
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
            if k not in expected_down and k not in expected_up:
                unexpected.append(k)
        return type("Result", (), {"missing_keys": missing, "unexpected_keys": unexpected})()

    def save(self, path: str | Path) -> None:
        sd = self.state_dict()
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": "stylek",
            "ss_network_args": json.dumps({
                "algo": "stylek",
                "rank": self.rank,
                "min_rank": self.min_rank,
                "alpha": self.alpha,
                "alpha_rank_scale": self.alpha_rank_scale,
                "sig_type": self.sig_type,
                "ortho_reg": self.ortho_reg,
                "mag_reg": self.mag_reg,
                "mag_amplify": self.mag_amplify,
            }),
        }
        save_file(sd, str(path), metadata=meta)
        logger.info(f"StyleK-LoRA 保存到: {path}")

    def load(self, path: str | Path) -> None:
        logger.info(f"加载 StyleK-LoRA 权重: {path}")
        sd: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        result = self.load_state_dict(sd, strict=False)
        missing = len(getattr(result, "missing_keys", []))
        unexpected = len(getattr(result, "unexpected_keys", []))
        logger.info(f"加载 {len(sd)} 个张量，missing={missing}, unexpected={unexpected}")
