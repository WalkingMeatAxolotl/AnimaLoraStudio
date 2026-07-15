"""Anima 族的 LoRA/LoKr target preset（多模型 PR-2b，自 utils/lokr_preset.py 迁入）。

「算法族无关、target 选择族相关」（docs/design/multi-model/01 §7）：LyCORIS 的
lokr/loha/lora 数学对任何 Linear 堆都成立；打哪些层、排除什么、保存键名前缀
才是族知识，归 families/。

LycorisNetwork.apply_preset(ANIMA_PRESET) 后，注入到 Anima DiT 时：
- 命中 self/cross attention 的 q/k/v/output_proj
- 命中 MLP 的 layer1/layer2
- 排除 llm_adapter（训这个会破坏文本理解）
- 不动 norm 层
- 保存键名前缀 lora_unet_*（ComfyUI 生态约定，与 spec.lora.prefix 一致）
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
