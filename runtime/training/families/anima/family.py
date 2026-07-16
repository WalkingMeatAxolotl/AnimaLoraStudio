"""AnimaFamily —— Anima 族的 ModelFamily 实现（多模型 PR-2b）。

行为实现分居同目录模块：loader.py（DiT/TE 加载）、forward.py（检查点前向）、
preset.py（LoRA target）；采样与文本编码当前仍在 training.families.anima.sampling /
training.families.anima.text_encoding（S3 迁入本目录，方法已收口在此）。
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
        from training.vae import load_vae

        return load_vae(path, device, dtype, None, tiling=tiling)

    def load_text(self, text_encoder_path, device, dtype, *,
                  t5_tokenizer_path: str = "", comfy_qwen: bool = False,
                  t5_fast: bool = False, purpose: str = "train",
                  cache_enabled: bool = True):
        from training.families.anima.loader import load_text_encoders

        return load_text_encoders(
            text_encoder_path, t5_tokenizer_path, device, dtype,
            comfy_qwen=(comfy_qwen or purpose == "generate"),
            t5_fast=t5_fast,
        )

    # ── 文本条件 ─────────────────────────────────────────────────────────
    def prepare_text_cache(self, captions: Iterable[str],
                           extra_prompts: Iterable[str], *, cache_entries=(),
                           cache_root=None, text=None, device=None,
                           dtype=None) -> None:
        return None  # Anima 每步在线编码（spec.text.strategy="online"），无缓存

    def encode_text_for_batch(self, text, dit, captions, device, dtype, *,
                              comfy_encoding: bool = True, kv_trim: bool = False):
        """每步在线编码（自 loop.py 文本块下沉，行为零变化）。

        comfy_encoding=True（默认）：raw caption 进 Qwen；T5 整段字面 tokenize，
        与测试出图 / 训练预览 conditioning 同一链路。False = legacy A/B 路径。
        返回 cross —— 对循环 opaque（03 §2.7-4）。
        """
        import torch
        import torch.nn.functional as F

        from training.families.anima.text_encoding import (
            _build_qwen_text_from_prompt,
            encode_qwen,
            tokenize_t5_comfy_literal,
            tokenize_t5_weighted,
        )

        qwen_model, qwen_tok, t5_tok = text
        max_len = self.spec.text.max_seq_len
        with torch.no_grad():
            if comfy_encoding:
                qwen_texts = [str(c) for c in captions]
                qwen_emb, _ = encode_qwen(qwen_model, qwen_tok, qwen_texts, device)
                t5_ids, t5_attn, t5_w = tokenize_t5_comfy_literal(t5_tok, captions, max_length=max_len)
            else:
                qwen_texts = [_build_qwen_text_from_prompt(c) for c in captions]
                qwen_emb, _ = encode_qwen(qwen_model, qwen_tok, qwen_texts, device)
                t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tok, captions, max_length=max_len)
            t5_ids = t5_ids.to(device)
            t5_attn = t5_attn.to(device)
            t5_w = t5_w.to(device, dtype=torch.float32)
            # t5_w 在 preprocess_text_embeds 内乘到 LLMAdapter 输出上（ComfyUI 对齐）
            cross = dit.preprocess_text_embeds(qwen_emb, t5_ids, t5xxl_weights=t5_w)
            if cross.shape[1] < max_len:
                cross = F.pad(cross, (0, 0, 0, max_len - cross.shape[1]))
            if kv_trim:
                # KV trim：padding 截到最近有效 token bucket（64/128/256/512）
                _actual = int(t5_attn.sum(dim=-1).max().item())
                _bucket = max_len
                for _b in (64, 128, 256, max_len):
                    if _b >= _actual:
                        _bucket = _b
                        break
                cross = cross[:, :_bucket, :].contiguous()
        return cross

    # ── 训练前向 / 采样 ──────────────────────────────────────────────────
    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint: bool = False):
        """v_pred = f(noisy, t, cond)。pad_mask（concat_padding_mask 私有输入）
        与 t 形状按摩（(B,)→(B,1)）为族内部事务（03-③）。"""
        import torch

        from training.families.anima.forward import forward_with_optional_checkpoint

        pad_mask = torch.zeros(
            noisy.shape[0], 1, noisy.shape[-2], noisy.shape[-1],
            device=noisy.device, dtype=noisy.dtype,
        )
        return forward_with_optional_checkpoint(
            dit, noisy, t.view(-1, 1), cond, pad_mask,
            use_checkpoint=use_checkpoint,
        )

    def sample_image(self, model, vae, text, prompt, **kwargs):
        from training.families.anima.sampling import sample_image

        # distilled 是蒸馏推理族（Krea2 Turbo）的旋钮；Anima 无蒸馏变体，
        # 接受并忽略（调用方按统一协议传，无需按族分支）。
        kwargs.pop("distilled", None)
        return sample_image(model, vae, *text, prompt, **kwargs)

    # ── LoRA 产物 ────────────────────────────────────────────────────────
    def lora_preset(self) -> dict[str, Any]:
        return ANIMA_PRESET

    def lora_metadata(self) -> dict[str, str]:
        return {
            "model_family": self.spec.family_id,
            "preset": self.spec.lora.preset_name,
        }

    def convert_lora_state_dict(self, sd: dict) -> dict:
        return sd  # kohya 格式即产物格式（04 §7.1），恒等
