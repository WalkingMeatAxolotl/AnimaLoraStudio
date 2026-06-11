"""OrthoLoRA adapter builder."""
from __future__ import annotations

from training.adapters.protocol import AdapterProtocol


def build(args) -> AdapterProtocol:
    from utils.ortho_adapter import OrthoLoRAAdapter

    return OrthoLoRAAdapter(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
        rank_dropout=float(getattr(args, "lora_rank_dropout", 0.0) or 0.0),
        module_dropout=float(getattr(args, "lora_module_dropout", 0.0) or 0.0),
        use_timestep_mask=False,
    )
