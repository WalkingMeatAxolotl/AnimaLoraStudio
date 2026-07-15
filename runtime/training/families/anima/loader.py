"""Anima 族加载器（多模型 PR-2b，自 training/models.py 函数级迁入）。

load_anima_model / load_text_encoders 是族知识（checkpoint 形状推断两档、
llm_adapter 缺失兜底、Qwen+T5 双 encoder）；VAEWrapper / load_vae 为跨族
共享资产留在 training.vae（D6）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from training.model_loading import (
    _load_safetensors_state_dict,
    _load_weights_best_effort,
)

logger = logging.getLogger(__name__)


def load_anima_model(transformer_path, device, dtype, repo_root, *, flash_attn: bool = True):
    """加载 Anima transformer 模型。

    `flash_attn=False` 显式禁用 flash_attn fast path（attention_backend=xformers/none
    时由 caller 传入），让 caller 完全决定 attention 实现 —— PR #17 那版默认
    fn(True) 强制开 flash_attn 不让用户关，与 cfg.attention_backend 解耦不彻底。
    """
    from safetensors import safe_open

    # repo_root 参数保留但已不使用（sister 契约签名「可加不可减不可改」）：模型
    # 代码随仓库发布，走正常 import —— 单一模块身份，exec-load 已退役（多模型
    # PR-2a），attention backend 开关不再需要跨模块别名广播。
    from modeling.anima import anima_modeling, cosmos_predict2_modeling

    Anima = anima_modeling.Anima

    # attention backend 全局开关：set_attention_backend() 一次性清掉未选中的 fast path
    flash_enabled = False
    for module in (cosmos_predict2_modeling, anima_modeling):
        set_backend = getattr(module, "set_attention_backend", None)
        if set_backend is not None:
            try:
                effective = str(set_backend("flash_attn" if flash_attn else "none"))
                flash_enabled = (effective == "flash_attn") or flash_enabled
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("attention backend 设置失败，继续走 SDPA fallback: %s", exc)
                continue
        fn = getattr(module, "set_flash_attn_enabled", None)
        if fn is None:
            continue
        try:
            flash_enabled = bool(fn(flash_attn)) or flash_enabled
        except Exception as exc:  # noqa: BLE001
            logger.warning("flash_attn 启用失败，继续走 SDPA fallback: %s", exc)
    if flash_enabled:
        logger.info("flash_attn 启用（训练 + sample 走 fast path）")
    else:
        logger.info("flash_attn 关闭（attention_backend=%s 或包未安装）",
                    "flash_attn" if flash_attn else "non-flash")

    # 从 checkpoint 推断配置
    with safe_open(transformer_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith("x_embedder.proj.1.weight"):
                w = f.get_tensor(k)
                break

    in_channels = (w.shape[1] // 4) - 1  # concat_padding_mask=True
    model_channels = w.shape[0]

    if model_channels == 2048:
        num_blocks, num_heads = 28, 16
    elif model_channels == 5120:
        num_blocks, num_heads = 36, 40
    else:
        raise RuntimeError(f"未知的 model_channels={model_channels}")

    config = dict(
        max_img_h=1024, max_img_w=1024, max_frames=128,
        in_channels=in_channels, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=model_channels,
        num_blocks=num_blocks, num_heads=num_heads,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=256,
        rope_h_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_w_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_t_extrapolation_ratio=1.0,
    )

    model = Anima(**config)

    # 加载权重
    sd = _load_safetensors_state_dict(Path(transformer_path))
    info = _load_weights_best_effort(model, sd, label="Transformer")

    # 如果 checkpoint 中完全没有 llm_adapter 权重，随机初始化会把 cross-attn 条件搞乱，直接禁用更安全
    has_llm_adapter = any("llm_adapter" in k for k in sd.keys())
    if not has_llm_adapter and hasattr(model, "llm_adapter"):
        try:
            model.llm_adapter = None
            logger.warning("检测到 checkpoint 不包含 llm_adapter 权重：已禁用 llm_adapter（回退为直接使用 Qwen embeddings）")
        except Exception:
            pass
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    logger.info(f"Anima 模型加载完成: {model_channels}ch, {num_blocks} blocks")
    return model


def load_text_encoders(
    qwen_path,
    t5_tokenizer_path,
    device,
    dtype,
    *,
    comfy_qwen: bool = False,
    t5_fast: bool = False,
):
    """加载文本编码器（Qwen + T5）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5TokenizerFast

    # Qwen
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    if comfy_qwen:
        from training.families.anima.comfy_qwen import load_comfy_qwen3_encoder

        qwen_model = load_comfy_qwen3_encoder(qwen_path, device=device, dtype=dtype)
    else:
        qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval().requires_grad_(False)

    # T5 tokenizer
    t5_cls = T5TokenizerFast if t5_fast else T5Tokenizer
    if t5_tokenizer_path and Path(t5_tokenizer_path).exists():
        t5_tokenizer = t5_cls.from_pretrained(t5_tokenizer_path)
    else:
        logger.warning(
            "T5 tokenizer 本地目录缺失（t5_tokenizer_path=%s），"
            "开始从 Hugging Face 下载 google/t5-v1_1-xxl",
            t5_tokenizer_path or "未配置",
        )
        try:
            t5_tokenizer = t5_cls.from_pretrained("google/t5-v1_1-xxl")
        except Exception as e:
            raise RuntimeError(
                f"T5 tokenizer 下载失败（google/t5-v1_1-xxl）：{type(e).__name__}: {e}\n"
                f"请检查网络后重试；或在 Studio 设置页下载 t5_tokenizer 模型，"
                f"并确认 t5_tokenizer_path（当前值：{t5_tokenizer_path or '未配置'}）指向该目录。"
            ) from e

    logger.info("文本编码器加载完成")
    return qwen_model, qwen_tokenizer, t5_tokenizer
