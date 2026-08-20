"""Anima 族加载器（多模型 PR-2b，自 training/models.py 函数级迁入）。

load_anima_model / load_text_encoders 是族知识（checkpoint 形状推断两档、
llm_adapter 缺失兜底、Qwen+T5 双 encoder）；VAEWrapper / load_vae 为跨族
共享资产留在 training.vae（D6）。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from studio.infrastructure.log_messages import msg
from training.model_loading import (
    _load_safetensors_state_dict,
    _load_weights_best_effort,
)

logger = logging.getLogger(__name__)

#: checkpoint 键里的 block 归属（键可能带 model./module. 等前缀，
#: _load_weights_best_effort 加载时才剥——这里按子串匹配，前缀无关）
_BLOCK_KEY_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


def swapped_param_ratio_from_header(checkpoint_path, blocks_to_swap: int) -> float:
    """换出层占全模型参数的比例，从 safetensors header 数 numel（不读 payload）。

    krea2 用固定 config 数 meta 模型参数；Anima 的层数由 checkpoint 决定
    （2B=28 层 / 14B=36 层），header 才是版本真相，且数参数天然 dtype 无关
    （显存折扣必须按比例乘文件实际大小，见 krea2 loader 同名函数的说明）。
    """
    from safetensors import safe_open

    if blocks_to_swap <= 0:
        return 0.0
    per_block: dict[int, int] = {}
    total = 0
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            numel = 1
            for dim in f.get_slice(key).get_shape():
                numel *= dim
            total += numel
            m = _BLOCK_KEY_RE.search(key)
            if m:
                idx = int(m.group(1))
                per_block[idx] = per_block.get(idx, 0) + numel
    if not per_block or total <= 0:
        return 0.0
    num_blocks = max(per_block) + 1
    first = max(num_blocks - blocks_to_swap, 0)
    swapped = sum(n for i, n in per_block.items() if i >= first)
    return swapped / total


def place_model_for_block_swap(model, device, dtype, blocks_to_swap: int) -> int:
    """换出层不上卡的模型放置：CPU 内 cast 到 dtype，只把非换出部分搬上 GPU。

    §9.4 纪律（docs/design/block-swap.md）：**不能全量上卡再搬下来**——那样
    GPU 瞬时峰值仍等于完整模型，小卡目标不成立。换出层留在 CPU（dtype 已
    cast），pinned 化由 ``PinnedBlockSwap._build`` 就地接管（已在 CPU 的张量
    只 pin、不重复拷贝）。

    返回实际换出层数（clamp 到总层数——blocks_to_swap 是全局设置，用户可能
    按 36 层版调的值喂给 28 层版，超界按全量换出处理）。
    """
    import torch

    from training.sysmem import check_pinned_budget

    total = len(model.blocks)
    num_swap = min(int(blocks_to_swap), total)
    first = total - num_swap
    swapped_prefixes = tuple(f"blocks.{i}." for i in range(first, total))

    # pinned 预算护栏先行（B6：fail-fast，此刻尚无任何 GPU / pinned 分配）
    elem = torch.empty(0, dtype=dtype).element_size()
    need = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith(swapped_prefixes)
    ) * elem
    check_pinned_budget(need, blocks=num_swap)

    model.to(dtype=dtype)  # CPU 内 cast（fp32 构建 → 目标 dtype）
    target = torch.device(device)
    for name, param in model.named_parameters():
        if not name.startswith(swapped_prefixes):
            param.data = param.data.to(target)
    for name, buf in model.named_buffers():
        if not name.startswith(swapped_prefixes):
            buf.data = buf.data.to(target)
    # 公开标记：采样期 VAE decode 的整模型 offload 必须跳过本模型（一刀切
    # .to() 恢复时会把 CPU 主副本搬上卡，swap 白做且瞬时占用=完整模型），
    # families/anima/sampling.py 按它分流
    model.blocks_to_swap = num_swap
    logger.debug(
        "block_swap: placement swapped=%d/%d pinned=%.2f GB rest_on_gpu=true",
        num_swap, total, need / 1024 ** 3,
    )
    return num_swap


def load_anima_model(transformer_path, device, dtype, repo_root, *,
                     flash_attn: bool = True, blocks_to_swap: int = 0,
                     attention_backend: str = ""):
    """加载 Anima transformer 模型。

    `flash_attn=False` 显式禁用 flash_attn fast path（attention_backend=xformers/none
    时由 caller 传入），让 caller 完全决定 attention 实现 —— PR #17 那版默认
    fn(True) 强制开 flash_attn 不让用户关，与 cfg.attention_backend 解耦不彻底。

    `attention_backend` 只用于日志：告诉用户「flash_attn 没开是因为你选了哪个
    backend」，与「装没装 flash_attn 包」是两个不同的原因。
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
                logger.warning(
                    "Attention backend could not be set: %s — attention falls back "
                    "to PyTorch SDPA", exc,
                )
                continue
        fn = getattr(module, "set_flash_attn_enabled", None)
        if fn is None:
            continue
        try:
            flash_enabled = bool(fn(flash_attn)) or flash_enabled
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "flash_attn could not be enabled: %s — attention falls back to "
                "PyTorch SDPA", exc,
            )
    if flash_enabled:
        logger.info(msg("train.flash_attn_on"))
    elif not flash_attn:
        # 「设置里关掉」与「包没装」是两个确定原因，各给一条确定文案。
        logger.info(msg(
            "train.flash_attn_off_by_setting",
            backend=attention_backend or "none",
        ))
    else:
        logger.info(msg("train.flash_attn_off_missing"))

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
            logger.warning(
                "The model file has no text-adapter weights: the adapter is "
                "disabled and the text encoder output is fed to the model directly"
            )
        except Exception:
            pass
    if blocks_to_swap > 0:
        place_model_for_block_swap(model, device, dtype, blocks_to_swap)
    else:
        model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    logger.info(msg(
        "train.anima_model_loaded", channels=model_channels, blocks=num_blocks,
    ))
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
            "T5 tokenizer not found locally (t5_tokenizer_path=%s): downloading "
            "google/t5-v1_1-xxl from Hugging Face — this needs a working internet "
            "connection",
            t5_tokenizer_path or "unset",
        )
        try:
            t5_tokenizer = t5_cls.from_pretrained("google/t5-v1_1-xxl")
        except Exception as e:
            raise RuntimeError(
                f"T5 tokenizer 下载失败（google/t5-v1_1-xxl）：{type(e).__name__}: {e}\n"
                f"请检查网络后重试；或在 Studio 设置页下载 t5_tokenizer 模型，"
                f"并确认 t5_tokenizer_path（当前值：{t5_tokenizer_path or '未配置'}）指向该目录。"
            ) from e

    logger.info(msg("train.text_encoder_loaded"))
    return qwen_model, qwen_tokenizer, t5_tokenizer
