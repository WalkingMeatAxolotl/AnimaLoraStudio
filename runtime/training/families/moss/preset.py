"""MOSS-Image 的 LoRA/LoKr target preset（MOSS 设计稿 §5.7）。

- 命中 42 个主干 block 的 ``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}``（252 个 Linear）
- 不动 ``text_embedder``（文本 refiner）、各 embedder、共享 modulation 与输出头
- 保存键名 ``lora_unet_layers_{i}_proj_{name}``：ComfyUI ``comfy/lora.py`` 的 MOSS 分支把
  分开的 q/k/v 映射到融合 ``proj.qkv`` 的行切片，非融合底模走通用映射，所以同一个
  LoRA 在 bf16 / int8、融合 / 非融合底模上都能加载
"""
from __future__ import annotations

from typing import Any

MOSS_PRESET: dict[str, Any] = {
    "enable_conv": False,
    "target_module": [],
    "target_name": ["layers.*.proj.*"],
    "exclude_name": ["text_embedder*"],
    "use_fnmatch": True,
    "lora_prefix": "lora_unet",
    "module_algo_map": {},
    "name_algo_map": {},
}
