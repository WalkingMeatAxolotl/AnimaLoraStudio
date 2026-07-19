"""Krea2 LoRA/LoKr target preset.

The all-Linear target and ``lora_unet`` prefix follow kohya-ss/musubi-tuner
(Apache-2.0), fixed commit 8934cfb:
Copyright 2026 Kohya S. and musubi-tuner contributors.
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/networks/lora_krea2.py

The musubi ``target=None`` contract means every ``nn.Linear`` in the Krea2 DiT.
For this repository's LyCORIS/Ortho preset walkers, ``target_name=["*"]`` with
convolution disabled is the equivalent representation. It covers all 264
Linear modules, including attention gates and the text-fusion stack.
"""

from __future__ import annotations

from typing import Any


KREA2_PRESET: dict[str, Any] = {
    "enable_conv": False,
    "target_module": [],
    "target_name": ["*"],
    "exclude_name": [],
    "use_fnmatch": True,
    "lora_prefix": "lora_unet",
    "module_algo_map": {},
    "name_algo_map": {},
}
