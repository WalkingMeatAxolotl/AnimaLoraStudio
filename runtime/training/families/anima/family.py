"""AnimaFamily —— Anima 族的 ModelFamily 实现（多模型 PR-2b）。

行为实现分居同目录模块：loader.py（DiT/TE 加载）、forward.py（检查点前向）、
preset.py（LoRA target）；采样与文本编码当前仍在 training.sampling /
training.text_encoding（S3 迁入本目录，方法已收口在此）。
"""

from __future__ import annotations

from typing import Any, Iterable

from training.families.anima import ANIMA_SPEC
from training.families.anima.preset import ANIMA_PRESET


class AnimaFamily:
    spec = ANIMA_SPEC

    # ── 加载 ─────────────────────────────────────────────────────────────
    def load_dit(self, path, device, dtype, *,
                 attention_backend: str = "flash_attn", repo_root=None):
        from training.families.anima.loader import load_anima_model
        from training.model_loading import enable_xformers

        model = load_anima_model(
            path, device, dtype, repo_root,
            flash_attn=(attention_backend == "flash_attn"),
        )
        if attention_backend == "xformers":
            enable_xformers(model)
        return model

    def load_vae(self, path, device, dtype, *, tiling: str = "auto"):
        # 跨族共享实现（D6）；方法留在 family 上，第 3 族 VAE 不同款时零迁移
        from training.models import load_vae

        return load_vae(path, device, dtype, None, tiling=tiling)

    def load_text(self, text_encoder_path, device, dtype, *,
                  t5_tokenizer_path: str = "", comfy_qwen: bool = False,
                  purpose: str = "train"):
        from training.families.anima.loader import load_text_encoders

        return load_text_encoders(
            text_encoder_path, t5_tokenizer_path, device, dtype,
            comfy_qwen=(comfy_qwen or purpose == "generate"),
        )

    # ── 文本条件 ─────────────────────────────────────────────────────────
    def prepare_text_cache(self, captions: Iterable[str],
                           extra_prompts: Iterable[str]) -> None:
        return None  # Anima 每步在线编码（spec.text.strategy="online"），无缓存

    def encode_text_for_batch(self, text, dit, captions, device, dtype, *,
                              kv_trim: bool = True):
        raise NotImplementedError("S3 接线：loop.py 文本编码块下沉后启用")

    # ── 训练前向 / 采样 ──────────────────────────────────────────────────
    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint: bool = False):
        raise NotImplementedError("S3 接线：pad_mask 构造随文本块一并下沉后启用")

    def sample_image(self, *args, **kwargs):
        from training.sampling import sample_image

        return sample_image(*args, **kwargs)

    # ── LoRA 产物 ────────────────────────────────────────────────────────
    def lora_preset(self) -> dict[str, Any]:
        return ANIMA_PRESET

    def lora_metadata(self) -> dict[str, str]:
        return {"model_family": self.spec.family_id}

    def convert_lora_state_dict(self, sd: dict) -> dict:
        return sd  # kohya 格式即产物格式（04 §7.1），恒等
