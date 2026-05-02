"""Anima 模型的 lycoris-lora preset 配置。

LycorisNetwork.apply_preset(ANIMA_PRESET) 后，注入到 Anima DiT 时：
- 命中 self/cross attention 的 q/k/v/output_proj
- 命中 MLP 的 layer1/layer2
- 排除 llm_adapter（与 anima_train.py:986 当前规则一致 — 训这个会破坏文本理解）
- 不动 norm 层
- 保存键名前缀 lora_unet_*（与现 ComfyUI 加载流程兼容）
"""
from __future__ import annotations

from typing import Any

ANIMA_PRESET: dict[str, Any] = {
    "enable_conv": False,                    # Anima DiT 主干 + TE + LLM Adapter 全是 nn.Linear
    "target_module": [],                     # 不按 module class 匹配
    "target_name": [
        "*q_proj", "*k_proj", "*v_proj", "*output_proj",
        "*mlp.layer1", "*mlp.layer2",
    ],
    "exclude_name": ["llm_adapter*"],
    "use_fnmatch": True,                     # 启用 fnmatch（接受 * 通配）
    "lora_prefix": "lora_unet",              # 保留 ComfyUI 现有加载流程兼容
    "module_algo_map": {},                   # 留作后续 per-module algorithm 覆盖
    "name_algo_map": {},                     # 同上，按层名覆盖
}


def apply(network_cls) -> None:
    """对 LycorisNetwork 类应用 ANIMA_PRESET。

    需在 `LycorisNetwork(model, ...)` 实例化前调用。
    """
    network_cls.apply_preset(ANIMA_PRESET)
