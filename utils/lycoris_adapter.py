"""LycorisNetwork 的 Anima-friendly 封装。

替换 anima_train.py 中的 LoRAInjector / LoRALayer / LoKrLayer / LoRALinear。

API 与原 LoRAInjector 等价（drop-in），并保留 w1 排除 weight_decay 的优化。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file
from safetensors import safe_open

from utils.lokr_preset import apply as apply_anima_preset
from utils.lycoris_patch import apply_lokr_device_patch

logger = logging.getLogger(__name__)

# lycoris-lora 3.4.0 LokrModule.get_weight rank_dropout device bug 一次性修复。
# 模块级调用：任何路径走到 lycoris_adapter（CLI 训练 / Studio worker / 测试）
# 都会先 patch 一次。返回值供测试断言；正常 import 路径下结果落到 logger。
_LOKR_PATCH_STATUS = apply_lokr_device_patch()


class AnimaLycorisAdapter:
    """对 LycorisNetwork 的等价封装，对外接口对齐原 LoRAInjector。

    对比原 LoRAInjector：
    - inject()/get_params()/get_param_groups()/state_dict()/save()/load() 等价
    - 多支持 algo: lora/lokr/loha + DoRA/dropout/rs_lora 等 LyCORIS 原生参数
    - 保留 w1 排除 weight_decay 的优化
    - 保存键名前缀 lora_unet_*，与现 ComfyUI workflow 完全兼容
    """

    def __init__(
        self,
        algo: str = "lokr",
        rank: int = 32,
        alpha: float = 16.0,
        factor: int = 8,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        weight_decompose: bool = False,
        rs_lora: bool = False,
    ):
        self.algo = algo
        self.rank = rank
        self.alpha = alpha
        self.factor = factor
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.weight_decompose = weight_decompose
        self.rs_lora = rs_lora

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

        apply_anima_preset(LycorisNetwork)

        # algo 名映射：anima_train 用 'lora'，lycoris 用 'locon'（with conv 关闭即等价 lora）
        net_module = self.algo
        if net_module == "lora":
            net_module = "locon"

        # extra kwargs: 仅在该算法支持时传入对应字段
        extra: dict[str, Any] = {}
        if self.algo == "lokr":
            extra["factor"] = self.factor
        if self.weight_decompose:
            extra["weight_decompose"] = True
        if self.rs_lora:
            extra["rs_lora"] = True

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

        # lycoris 默认在 CPU 创建模块；model 多半已在 CUDA — 必须显式同步 device/dtype，
        # 否则首次 forward 报 "tensors on cuda:0 and cpu"。从模型首个 parameter 推断。
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
        logger.info(f"注入 {self.algo.upper()} 到 {n} 层（lycoris-lora）")
        return {lora.lora_name: lora for lora in self.network.loras}

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
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_network_module": "lycoris.kohya",
            "ss_network_args": json.dumps({
                "algo": self.algo,
                "factor": self.factor,
                "preset": "anima_full",
                "dropout": self.dropout,
                "rank_dropout": self.rank_dropout,
                "module_dropout": self.module_dropout,
                "weight_decompose": self.weight_decompose,
                "rs_lora": self.rs_lora,
            }),
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
