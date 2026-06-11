"""Comfy-style runtime defaults for test generation.

Comfy-style generation follows the ComfyUI implementation where possible:
bf16 UNET/text weights, fp32 VAE, fast T5 tokenization, and Comfy-style Qwen
conditioning. Exact image/latent parity with the pinned ComfyUI KSampler oracle
is guaranteed only for the xformers attention backend used by that oracle
environment. Other backends can run the same Comfy-style path, but they are not
exact KSampler parity guarantees.
"""

from typing import Any


EXACT_KSAMPLER_PARITY_BACKEND = "xformers"
COMFY_PARITY_ATTENTION_BACKEND = EXACT_KSAMPLER_PARITY_BACKEND
COMFY_PARITY_MIXED_PRECISION = "bf16"
COMFY_PARITY_VAE_PRECISION = "fp32"
COMFY_PARITY_TEXT_ENCODER_BACKEND = "comfy_qwen3"


def is_exact_ksampler_parity_backend(attention_backend: str | None) -> bool:
    return str(attention_backend or "").lower().strip() == EXACT_KSAMPLER_PARITY_BACKEND


def force_comfy_parity_runtime_config(
    data: Any,
    *,
    force_exact_ksampler_backend: bool = True,
) -> Any:
    """Return a config dict for the Comfy-style generation runtime.

    When ``force_exact_ksampler_backend`` is true, the attention backend is
    locked to the xformers backend used by the pinned ComfyUI KSampler oracle.
    When false, the caller's chosen backend is preserved; that path is
    Comfy-style but does not guarantee exact KSampler parity unless the backend
    is already xformers.
    """
    if not isinstance(data, dict):
        return data
    out = dict(data)
    out.pop("xformers", None)
    out.pop("flash_attn", None)
    if force_exact_ksampler_backend:
        out["attention_backend"] = EXACT_KSAMPLER_PARITY_BACKEND
    else:
        out["attention_backend"] = str(out.get("attention_backend") or "none")
    out["mixed_precision"] = COMFY_PARITY_MIXED_PRECISION
    out["vae_precision"] = COMFY_PARITY_VAE_PRECISION
    out["text_encoder_backend"] = COMFY_PARITY_TEXT_ENCODER_BACKEND
    out["t5_tokenizer_backend"] = "fast"
    return out
