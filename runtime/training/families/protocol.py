"""ModelFamily 行为契约（多模型 PR-2b；冻结面 = docs/design/multi-model/03 §4.1 + 04 §3-C4）。

九方法：load_dit / load_vae / load_text / prepare_text_cache / encode_text_for_batch
/ forward_train / sample_image / lora_preset / lora_metadata，外加
convert_lora_state_dict（默认恒等）。演化纪律：追加参数一律 keyword-only 带默认值，
禁止 **kwargs 黑洞（03 §4.2）。

共享循环七不变量（03 §2.7）由实现方保证：latent 恒 5D 且 T==1、t∈(0,1)、
rectified flow 代数归循环、v_pred 同形、cond opaque、autocast 归循环、RNG 纪律。
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

from training.families.spec import ModelSpec


@runtime_checkable
class ModelFamily(Protocol):
    spec: ModelSpec

    # ── 加载（models_phase + 全部旁路调用方，04 D8'）─────────────────────
    def load_dit(self, path: str, device, dtype, *,
                 attention_backend: str = "flash_attn", repo_root=None) -> Any: ...

    def load_vae(self, path: str, device, dtype, *, tiling: str = "auto") -> Any: ...

    def load_text(self, text_encoder_path: str, device, dtype, *,
                  t5_tokenizer_path: str = "", comfy_qwen: bool = False,
                  purpose: str = "train") -> Any: ...

    # ── 文本条件 ─────────────────────────────────────────────────────────
    def prepare_text_cache(self, captions: Iterable[str],
                           extra_prompts: Iterable[str]) -> None: ...

    def encode_text_for_batch(self, text, dit, captions: list[str],
                              device, dtype, *, kv_trim: bool = True) -> Any: ...

    # ── 训练前向 / 采样 ──────────────────────────────────────────────────
    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint: bool = False): ...

    def sample_image(self, *args, **kwargs): ...

    # ── LoRA 产物 ────────────────────────────────────────────────────────
    def lora_preset(self) -> dict[str, Any]: ...

    def lora_metadata(self) -> dict[str, str]: ...

    def convert_lora_state_dict(self, sd: dict) -> dict: ...
