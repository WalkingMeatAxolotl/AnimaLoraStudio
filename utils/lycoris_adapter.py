"""LycorisNetwork 的 Anima-friendly 封装。

替换 anima_train.py 中的 LoRAInjector / LoRALayer / LoKrLayer / LoRALinear。

API 与原 LoRAInjector 等价（drop-in），并保留 w1 排除 weight_decay 的优化。

T-LoRA timestep rank mask 调度（_install_tlora_masks / _set_tlora_mask 的
``(1-t)^α`` 公式与 batch 均值聚合）取自 sorryhyun/anima_lora（MIT，
Copyright (c) 2026 Seunghyun Ji，见 THIRD_PARTY_NOTICES.md）；mask 注入机制
（patch lycoris make_weight）为本仓库实现。算法思想出自 T-LoRA 论文
（ControlGenAI/T-LoRA）。
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file
from safetensors import safe_open

from utils.lycoris_patch import apply_lokr_device_patch

logger = logging.getLogger(__name__)

# lycoris-lora 3.4.0 LokrModule.get_weight rank_dropout device bug 一次性修复。
# 模块级调用：任何路径走到 lycoris_adapter（CLI 训练 / Studio worker / 测试）
# 都会先 patch 一次。返回值供测试断言；正常 import 路径下结果落到 logger。
_LOKR_PATCH_STATUS = apply_lokr_device_patch()


class LycorisAdapter:
    """对 LycorisNetwork 的等价封装，对外接口对齐原 LoRAInjector。

    对比原 LoRAInjector：
    - inject()/get_params()/get_param_groups()/state_dict()/save()/load() 等价
    - 多支持 algo: lora/lokr/loha + DoRA/dropout/rs_lora 等 LyCORIS 原生参数
    - 保留 w1 排除 weight_decay 的优化
    - 保存键名前缀 lora_unet_*，与现 ComfyUI workflow 完全兼容
    """

    def __init__(
        self,
        *,
        preset: dict | None = None,
        algo: str = "lokr",
        rank: int = 32,
        alpha: float = 16.0,
        factor: int = 8,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        weight_decompose: bool = False,
        rs_lora: bool = False,
        lora_reg_dims: Optional[dict[str, int]] = None,
        tlora_min_rank: int = 8,
        tlora_alpha_rank_scale: float = 1.0,
    ):
        # 族知识（target/exclude/前缀）由调用方注入，见 families/<fam>/preset.py
        self._preset = preset
        self.algo = algo
        self.rank = rank
        self.alpha = alpha
        self.factor = factor
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.weight_decompose = weight_decompose
        self.rs_lora = rs_lora
        # 分层 rank：正则表达式 → rank 的字典，用 re.fullmatch 对 lora_name 匹配
        self.lora_reg_dims: Optional[dict[str, int]] = lora_reg_dims or None
        self.use_timestep_mask = (algo == "tlora")
        self.tlora_min_rank = max(1, min(int(tlora_min_rank), int(rank)))
        self.tlora_alpha_rank_scale = max(0.0, float(tlora_alpha_rank_scale))
        self._tlora_modules: list[nn.Module] = []
        self._tlora_mask: Optional[torch.Tensor] = None
        self._tlora_arange: Optional[torch.Tensor] = None

        # use_lokr 是原 LoRAInjector 的字段，anima_train.py 多处用它做分支判断；
        # 保留此字段以避免改动太多调用点。
        self.use_lokr = (algo == "lokr")
        self.network = None  # lazy init in inject()
        # commit 20：detach() 撤销 hook 用 —— inject 时记录 model 引用 +
        # model.train 原始函数，detach 时还原。
        self._injected_model: Optional[nn.Module] = None
        self._orig_train: Optional[Any] = None

    # --------------------------------------------------------------- inject
    def inject(self, model: nn.Module) -> dict[str, nn.Module]:
        """注入 lycoris 适配器到模型。"""
        from lycoris import LycorisNetwork

        if self._preset is None:
            raise ValueError(
                "LycorisAdapter 需要显式传入 preset（族知识不再内置，"
                "见 runtime/training/families/<fam>/preset.py）"
            )
        LycorisNetwork.apply_preset(self._preset)

        # algo 名映射：anima_train 用 'lora'/'tlora'，lycoris 用 'locon'（with conv 关闭即等价 lora）
        net_module = self.algo
        if net_module in {"lora", "tlora"}:
            net_module = "locon"

        # extra kwargs: 仅在该算法支持时传入对应字段
        extra: dict[str, Any] = {}
        if self.algo == "lokr":
            extra["factor"] = self.factor
        if self.weight_decompose:
            extra["weight_decompose"] = True
        if self.rs_lora:
            extra["rs_lora"] = True

        # algo='lora' (LoCon) 默认走 bypass_mode：lycoris LoConModule 默认 forward 会
        # rebuild ΔW=up@down (out,in) 再多跑一次 F.linear，等于每层 ~2× FLOPs。
        # bypass_mode=True 走 bypass_forward_diff = org_forward(x) + lora_up(lora_down(x))，
        # 是 LoRA 论文 + sd-scripts + PEFT 的标准 forward；对外行为完全等价但 ~2× 快。
        # DoRA(weight_decompose) 路径数学上必须 rebuild —— lycoris bypass forward 不走 wd
        # 分支，会让 DoRA 静默失效；这里 guard。参考 lycoris docs/Network-Args.md "Bypass Mode"。
        if self.algo == "lora" and not self.weight_decompose:
            extra["bypass_mode"] = True

        self.network = LycorisNetwork(
            model,
            multiplier=1.0,
            lora_dim=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            rank_dropout=self.rank_dropout,
            module_dropout=self.module_dropout,
            network_module=net_module,
            **extra,
        )
        self.network.apply_to()

        if self.lora_reg_dims:
            _apply_reg_dims_(self.network, self.lora_reg_dims)
        if self.use_timestep_mask:
            self._install_tlora_masks()

        # lycoris 默认在 CPU 创建模块；model 多半已在 CUDA — 必须显式同步
        # device/dtype，否则首次 forward 报 "tensors on cuda:0 and cpu"。从模型
        # 首个 parameter 推断。推理 parity 路径会在 load_state_dict 前再把 network
        # 转成 fp32，以贴近 ComfyUI LoRA patch 的中间精度。
        try:
            ref = next(model.parameters())
            self.network.to(device=ref.device, dtype=ref.dtype)
        except StopIteration:
            pass

        # LycorisNetwork 是独立 nn.Module，不在 model 子树里；model.eval()/.train() 不会
        # 级联到 lycoris 模块（self.training 永远 True）。这导致 sample 时仍进 rank_dropout
        # 分支，触发 lycoris 上游 bug：torch.rand(...) 没传 device，CPU mask 与 CUDA weight
        # 相乘报 device mismatch（lokr.py:380）。
        # 修复：劫持 model.train()，让 network 跟随；并立刻同步当前模式。
        # commit 20：保存 _orig_train + _injected_model 让 detach() 能还原。
        _orig_train = model.train
        _network = self.network

        def _train_with_lycoris(mode: bool = True):
            _network.train(mode)
            return _orig_train(mode)

        model.train = _train_with_lycoris  # type: ignore[method-assign]
        self.network.train(model.training)
        self._orig_train = _orig_train
        self._injected_model = model

        n = len(self.network.loras)
        forward_path = "bypass (low-rank)" if extra.get("bypass_mode") else "rebuild (ΔW)"
        logger.info(f"注入 {self.algo.upper()} 到 {n} 层（lycoris-lora, forward={forward_path}）")
        if self.use_lokr:
            full_matrix = [lora for lora in self.network.loras if getattr(lora, "use_w2", False)]
            if full_matrix:
                logger.info(
                    "LoKr dim/rank=%s 触发 LyCORIS full dimension：%s/%s 层的第二块不再分解，alpha 将被忽略。",
                    self.rank,
                    len(full_matrix),
                    n,
                )
        return {lora.lora_name: lora for lora in self.network.loras}

    def _install_tlora_masks(self) -> None:
        if self.network is None:
            return
        self._tlora_modules = [
            lora for lora in self.network.loras
            if hasattr(lora, "lora_down") and hasattr(lora, "lora_up") and callable(getattr(lora, "make_weight", None))
        ]
        for lora in self._tlora_modules:
            if getattr(lora, "_anima_tlora_patched", False):
                continue
            original_make_weight = lora.make_weight

            def _make_weight_with_tlora(device=None, *, _lora=lora, _original_make_weight=original_make_weight):
                wa = _lora.lora_up.weight.to(device)
                wb = _lora.lora_down.weight.to(device)
                mask = getattr(_lora, "_anima_tlora_mask", None)
                if mask is not None:
                    mask = mask.to(device=device, dtype=wa.dtype)
                    view_up = (1, -1) + (1,) * max(0, wa.dim() - 2)
                    view_down = (-1, 1) + (1,) * max(0, wb.dim() - 2)
                    wa = wa * mask.view(*view_up)
                    wb = wb * mask.view(*view_down)
                if getattr(_lora, "tucker", False):
                    t = _lora.lora_mid.weight
                    wa_t = wa.view(wa.size(0), -1).transpose(0, 1)
                    wb_t = wb.view(wb.size(0), -1)
                    from lycoris.modules.locon import rebuild_tucker
                    weight = rebuild_tucker(t, wa_t, wb_t)
                else:
                    weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)
                weight = weight.view(_lora.shape)
                if _lora.training and _lora.rank_dropout:
                    drop = (torch.rand(weight.size(0), device=device) > _lora.rank_dropout).to(weight.dtype)
                    drop = drop.view(-1, *[1] * len(weight.shape[1:]))
                    if _lora.rank_dropout_scale:
                        drop /= drop.mean()
                    weight *= drop
                return weight * _lora.scalar.to(device)

            lora._anima_original_make_weight = original_make_weight
            lora.make_weight = _make_weight_with_tlora
            lora._anima_tlora_patched = True
        logger.info(
            "T-LoRA timestep rank mask enabled on %s/%s LoRA layers (min_rank=%s, alpha_rank_scale=%s)",
            len(self._tlora_modules),
            len(self.network.loras),
            self.tlora_min_rank,
            self.tlora_alpha_rank_scale,
        )

    def _set_tlora_mask(self, sigma_t: torch.Tensor) -> None:
        if not self._tlora_modules:
            return
        device = sigma_t.device
        rank = self.rank
        if self._tlora_mask is None or self._tlora_mask.device != device:
            self._tlora_mask = torch.ones(rank, device=device)
            self._tlora_arange = torch.arange(rank, device=device)
            for lora in self._tlora_modules:
                lora._anima_tlora_mask = self._tlora_mask
        # 与 ControlGenAI/T-LoRA 官方 (arxiv 2507.05964) SDXL get_mask_by_timestep
        # 对齐: r = ((max_t - t)/max_t)^alpha * (rank - min_rank) + min_rank
        # PR 的 sigma_t ∈ [0,1]、t=1=noisy (training_loop.py:120 锁死),
        # 故 (max_t - t)/max_t == (1 - t)。alpha=1.0 退化为 FLUX 路径的线性 schedule。
        # 论文 motivation: "higher diffusion timesteps are more prone to overfitting" —
        # 高噪声 timestep 限制 rank, 低噪声 timestep 给满 rank。
        t = sigma_t.float().mean().clamp(min=0.0, max=1.0)
        frac = (1.0 - t).pow(self.tlora_alpha_rank_scale)
        active_rank = frac * (rank - self.tlora_min_rank) + self.tlora_min_rank
        active_rank = active_rank.clamp(min=float(self.tlora_min_rank), max=float(rank))
        self._tlora_mask.copy_((self._tlora_arange < active_rank).to(self._tlora_mask.dtype))

    def clear_timestep_mask(self) -> None:
        if self._tlora_mask is not None:
            self._tlora_mask.fill_(1)

    # --------------------------------------------------------------- detach
    def detach(self) -> bool:
        """撤销 inject：还原 model.train 钩子 + 调 LycorisNetwork.restore（如有）。

        让 daemon 切换 LoRA 时不必重 load 整个 transformer。返回值：
          - True：成功；旧 hook 已撤销，可安全 inject 新 LoRA
          - False：lycoris 当前版本没暴露 restore 接口，hook 残留；调用方
                  应 fallback 到模型整体 reload（粗暴但安全）

        多次调用幂等（self.network=None 后直接 noop）。
        """
        if self.network is None:
            return True

        # 先尝试 lycoris 自带的 restore；不同版本接口名不同，挨个试
        ok = True
        for restore_attr in ("restore", "restore_apply", "remove_apply"):
            fn = getattr(self.network, restore_attr, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception as e:
                    logger.warning(f"LycorisNetwork.{restore_attr}() 失败: {e}")
                    ok = False
                    break
        else:
            # 三个接口都不存在 → 当前 lycoris 版本不支持热卸载
            logger.warning("LycorisNetwork 无 restore/restore_apply/remove_apply 接口；hook 残留")
            ok = False

        # 还原 model.train 劫持（无论 restore 是否成功都该还原 monkey patch）
        if self._injected_model is not None and self._orig_train is not None:
            try:
                self._injected_model.train = self._orig_train  # type: ignore[method-assign]
            except Exception as e:
                logger.warning(f"还原 model.train 失败: {e}")
                ok = False

        # 释放引用让 GC 清掉 LycorisNetwork（含 closure 内 _network 引用）
        self.network = None
        self._injected_model = None
        self._orig_train = None
        return ok

    # --------------------------------------------------------------- params
    def get_params(self) -> list[nn.Parameter]:
        """所有可训练参数（与原 LoRAInjector.get_params 等价）"""
        if self.network is None:
            return []
        return [p for p in self.network.parameters() if p.requires_grad]

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        """LoKr 模式下 w1 排除 weight_decay（与原 LoRAInjector 等价）。

        其他算法下不分组，所有参数共用 weight_decay。
        """
        if self.network is None:
            return [{"params": [], "weight_decay": weight_decay}]

        if not self.use_lokr or weight_decay == 0:
            return [{"params": self.get_params(), "weight_decay": weight_decay}]

        no_decay = []  # lokr_w1（满矩阵分支）/ lokr_w1_a/b（如果开 decompose_both）
        decay = []
        for lora in self.network.loras:
            for n, p in lora.named_parameters():
                if not p.requires_grad:
                    continue
                # 'lokr_w1' / 'lokr_w1_a' / 'lokr_w1_b' 都视为 w1 系
                if "lokr_w1" in n:
                    no_decay.append(p)
                else:
                    decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # --------------------------------------------------------------- state I/O
    def state_dict(self) -> dict[str, torch.Tensor]:
        """LoRA 权重 state_dict（带 lora_unet_* 前缀，ComfyUI 兼容）。

        lycoris 已经按 LORA_PREFIX (preset 中 'lora_prefix=lora_unet') 输出正确前缀。
        """
        if self.network is None:
            return {}
        return self.network.state_dict()

    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True) -> Any:
        if self.network is None:
            raise RuntimeError("AnimaLycorisAdapter.inject() 必须先调用")
        return self.network.load_state_dict(sd, strict=strict)

    # --------------------------------------------------------------- safetensors
    def save(self, path: str | Path) -> None:
        """保存为 safetensors（带 ss_* metadata，ComfyUI/sd-scripts 兼容）"""
        sd = self.state_dict()
        # 每层 .alpha 按当前 (scale, lora_dim) 重算，让下游标准公式 alpha/rank 还原
        # 训练 scale。lycoris init 写入的 .alpha 已经是 scale*lora_dim；但
        # _apply_reg_dims_ 改 lora_dim 后没动 self.scale / self.alpha buffer，
        # 分层 rank 的层 .alpha 与 per-layer rank 失配 → ComfyUI 按 alpha/rank
        # 算出的 scale 偏离训练实际值几十倍，导致出图噪点。
        # 这里按 lora 当前真实 (scale, lora_dim) 重写：未分层层 no-op；分层层修正。
        _rewrite_per_layer_alpha_(self.network, sd)
        # lora_reg_dims 必须写入：训练时按 pattern 改了部分层的 rank，推理重建
        # 网络要用同一份字典才能让 lokr_w2_a/b 形状对齐 checkpoint。
        ss_args: dict[str, Any] = {
            "algo": self.algo,
            "factor": self.factor,
            "preset": "anima_full",
            "dropout": self.dropout,
            "rank_dropout": self.rank_dropout,
            "module_dropout": self.module_dropout,
            "weight_decompose": self.weight_decompose,
            "rs_lora": self.rs_lora,
        }
        if self.lora_reg_dims:
            ss_args["lora_reg_dims"] = self.lora_reg_dims
        # 多模型 D13：族标记（phases/models 注入 family.lora_metadata()）；
        # 无标记的存量产物读取侧 grandfather 为 anima
        ss_args.update(getattr(self, "metadata_extra", None) or {})
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": "lycoris.kohya",
            "ss_network_args": json.dumps(ss_args),
        }
        save_file(sd, str(path), metadata=meta)
        logger.info(f"LoRA 保存到: {path}")

    def load(self, path: str | Path) -> None:
        """从 safetensors 加载已有 LoRA 权重（用于继续训练）"""
        logger.info(f"加载已有 LoRA 权重: {path}")
        sd: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)

        # 旧自实现格式（lora_unet_*.lokr_w2_a/b 低秩 vs lokr_w2 全矩阵）的 fallback：
        # lycoris 期望的是它自己写出来的格式（同样 lora_unet_* 前缀，但内部
        # 可能因 dim 太大走 full_matrix 模式产 lokr_w2 而非 lokr_w2_a/b）。
        # 直接用 strict=False 让 lycoris 容忍键缺失，并打印缺失数。
        result = self.load_state_dict(sd, strict=False)
        missing = len(getattr(result, "missing_keys", [])) if hasattr(result, "missing_keys") else 0
        unexpected = len(getattr(result, "unexpected_keys", [])) if hasattr(result, "unexpected_keys") else 0
        logger.info(
            f"加载 {len(sd)} 个权重张量，"
            f"missing={missing}, unexpected={unexpected}"
        )

    # ─── ADR 0003 PR-C：AdapterProtocol 可选 hook 的 no-op 实现 ───
    # 给 LyCORIS adapter 满足 runtime_checkable Protocol；论文级变体
    # （T-LoRA / OFT / Ortho-Hydra）可在自己的 build wrapper 里 override。

    def on_step_begin(self, ctx) -> None:
        """每 micro-batch 前向之前调用；T-LoRA 按 sigma_t 更新 rank mask。"""
        if self.use_timestep_mask:
            self._set_tlora_mask(ctx.sigma_t)
        return None

    def regularization_loss(self, ctx):
        """LoKr/LoRA/LoHa 无额外正则项。"""
        return None

    def excludes_weight_decay(self, param_name: str) -> bool:
        """w1 系参数排除 weight_decay；其他不排除。

        注：现有 get_param_groups 已经在内部按 "lokr_w1" 子串分组并把这些
        param 的 weight_decay 设为 0，本方法只是 Protocol 接口暴露，给外
        部 caller（如 phase log 决策）调。
        """
        return self.use_lokr and "lokr_w1" in param_name


def _rewrite_per_layer_alpha_(network: Optional[nn.Module], sd: dict[str, torch.Tensor]) -> None:
    """对 sd 里每层 .alpha tensor 重算为 lora.scale × lora.lora_dim。

    保持下游标准 LoRA 公式 alpha/rank 还原训练 scale。对未触发 lora_reg_dims
    的层是 no-op（lycoris init 写入的 .alpha 本来就 = scale × lora_dim）；对
    分层 rank 的层是修正。
    """
    if network is None:
        return
    loras = getattr(network, "loras", None) or []
    for lora in loras:
        name = getattr(lora, "lora_name", None)
        if not name:
            continue
        alpha_key = f"{name}.alpha"
        if alpha_key not in sd:
            continue
        try:
            scale = float(getattr(lora, "scale"))
            dim = int(getattr(lora, "lora_dim"))
        except (AttributeError, TypeError, ValueError):
            continue
        old = sd[alpha_key]
        sd[alpha_key] = torch.tensor(scale * dim, dtype=old.dtype, device=old.device)


def _apply_reg_dims_(network: nn.Module, lora_reg_dims: dict[str, int]) -> None:
    """分层 rank：对 lora_name 正则全匹配的模块重新分配 rank。

    支持 LoRA/LoCoN（lora_A / lora_B 参数式，或 lora_up / lora_down 子模块式 ——
    lycoris 不同代码路径命名不同）和 LoKr 非全矩阵分支（lokr_w2_a / lokr_w2_b）。
    LoKr 全矩阵分支（lokr_w2，use_w2=True）的 rank 不适用分层覆盖，跳过并 warn。

    re-init 策略与 lycoris 原始初始化一致：
      - A/w2_a：kaiming_uniform_(a=√5)（如同 nn.Linear weight init）
      - B/w2_b：zeros（确保初始 ΔW=0）
    """
    patterns = list(lora_reg_dims.items())
    changed = 0
    skipped = 0

    loras = getattr(network, "loras", None) or []
    for lora_mod in loras:
        name: str = getattr(lora_mod, "lora_name", "") or ""
        new_dim: Optional[int] = None
        for pat, dim in patterns:
            if re.fullmatch(pat, name):
                new_dim = int(dim)
                break
        if new_dim is None:
            continue

        old_dim = getattr(lora_mod, "lora_dim", None)
        if old_dim == new_dim:
            continue

        # ── LoRA / LoCoN ─────────────────────────────────────────────────────
        if hasattr(lora_mod, "lora_A") and hasattr(lora_mod, "lora_B"):
            in_f = lora_mod.lora_A.shape[1]
            out_f = lora_mod.lora_B.shape[0]
            dev, dt = lora_mod.lora_A.device, lora_mod.lora_A.dtype
            new_A = torch.empty(new_dim, in_f, device=dev, dtype=dt)
            nn.init.kaiming_uniform_(new_A, a=math.sqrt(5))
            lora_mod.lora_A = nn.Parameter(new_A)
            lora_mod.lora_B = nn.Parameter(torch.zeros(out_f, new_dim, device=dev, dtype=dt))
            lora_mod.lora_dim = new_dim
            changed += 1

        # ── LoRA/LoCoN 子模块式（本 lycoris 版本 LoConModule：lora_up/lora_down 是
        #    nn.Linear）。仅处理 Linear；conv（lora_mid/Conv2d）不适用分层覆盖 ──
        elif (
            isinstance(getattr(lora_mod, "lora_down", None), nn.Linear)
            and isinstance(getattr(lora_mod, "lora_up", None), nn.Linear)
        ):
            in_f = lora_mod.lora_down.in_features
            out_f = lora_mod.lora_up.out_features
            p = lora_mod.lora_down.weight
            dev, dt = p.device, p.dtype
            new_down = nn.Linear(in_f, new_dim, bias=False, device=dev, dtype=dt)
            nn.init.kaiming_uniform_(new_down.weight, a=math.sqrt(5))
            new_up = nn.Linear(new_dim, out_f, bias=False, device=dev, dtype=dt)
            nn.init.zeros_(new_up.weight)
            lora_mod.lora_down = new_down
            lora_mod.lora_up = new_up
            lora_mod.lora_dim = new_dim
            changed += 1

        # ── LoKr 低秩分支（use_w2=False）：w2_a [d_out//f, dim], w2_b [dim, d_in//f] ──
        elif hasattr(lora_mod, "lokr_w2_a") and hasattr(lora_mod, "lokr_w2_b"):
            d0 = lora_mod.lokr_w2_a.shape[0]   # d_out // factor
            d1 = lora_mod.lokr_w2_b.shape[1]   # d_in  // factor
            dev, dt = lora_mod.lokr_w2_a.device, lora_mod.lokr_w2_a.dtype
            new_w2a = torch.empty(d0, new_dim, device=dev, dtype=dt)
            nn.init.kaiming_uniform_(new_w2a, a=math.sqrt(5))
            lora_mod.lokr_w2_a = nn.Parameter(new_w2a)
            lora_mod.lokr_w2_b = nn.Parameter(torch.zeros(new_dim, d1, device=dev, dtype=dt))
            lora_mod.lora_dim = new_dim
            changed += 1

        # ── LoKr 全矩阵分支（use_w2=True）：rank 概念不适用，跳过 ──────────
        elif hasattr(lora_mod, "lokr_w2"):
            logger.warning(
                "[lora_reg_dims] %s: full-matrix LoKr branch (use_w2=True) — "
                "rank override not applicable, skipped",
                name,
            )
            skipped += 1

        else:
            skipped += 1

    if changed:
        logger.info("[lora_reg_dims] applied rank overrides to %d modules", changed)
    if skipped:
        logger.info("[lora_reg_dims] skipped %d modules (full-matrix or unsupported)", skipped)


# 兼容别名（更名于多模型 PR-2b；引用点逐步切换后删除）
AnimaLycorisAdapter = LycorisAdapter
