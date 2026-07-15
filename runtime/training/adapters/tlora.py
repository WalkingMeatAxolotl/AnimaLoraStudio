"""T-LoRA adapter builder."""

from __future__ import annotations

from training.adapters.protocol import AdapterProtocol


def build(args) -> AdapterProtocol:
    if bool(getattr(args, "tlora_use_ortho", False)):
        from training.families.anima.preset import ANIMA_PRESET
        from utils.ortho_adapter import OrthoLoRAAdapter

        return OrthoLoRAAdapter(
            preset=ANIMA_PRESET,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
            rank_dropout=float(getattr(args, "lora_rank_dropout", 0.0) or 0.0),
            module_dropout=float(getattr(args, "lora_module_dropout", 0.0) or 0.0),
            use_timestep_mask=True,
            tlora_min_rank=int(getattr(args, "tlora_min_rank", 1)),
            tlora_alpha_rank_scale=float(getattr(args, "tlora_alpha_rank_scale", 1.0)),
        )

    from training.families.anima.preset import ANIMA_PRESET
    from utils.lycoris_adapter import LycorisAdapter

    return LycorisAdapter(
        preset=ANIMA_PRESET,
        algo="tlora",
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        factor=args.lokr_factor,
        dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
        rank_dropout=float(getattr(args, "lora_rank_dropout", 0.0) or 0.0),
        module_dropout=float(getattr(args, "lora_module_dropout", 0.0) or 0.0),
        weight_decompose=bool(getattr(args, "lora_dora", False)),
        rs_lora=bool(getattr(args, "lora_rs", False)),
        lora_reg_dims=getattr(args, "lora_reg_dims", None) or None,
        tlora_min_rank=int(getattr(args, "tlora_min_rank", 1)),
        tlora_alpha_rank_scale=float(getattr(args, "tlora_alpha_rank_scale", 1.0)),
    )
