"""Anima Transformer / VAE / 文本编码器加载（公开 API，sister script 直接 import）。

抽自原 runtime/anima_train.py L614-775（ADR 0003 PR-A）。

公开（被 anima_daemon / anima_generate / anima_reg_ai 通过 anima_train.X 调用）：
- load_anima_model — Anima Transformer + flash_attn 开关 + checkpoint 推断配置
- load_vae — WAN VAE + 归一化 wrapper
- load_text_encoders — Qwen + T5 tokenizer

内部：
- ensure_models_namespace — 把模型代码目录加进 sys.path
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import torch

from training.model_loading import (
    _load_safetensors_state_dict,
    _load_weights_best_effort,
    load_module_from_path,
)


logger = logging.getLogger(__name__)


class VAEWrapper:
    """WAN VAE + 归一化参数 + 整图 decode OOM 自动回退到 tiled decode。

    Issue #200：8GB 类小显存跑 1024×1024 reg 生成时 transformer/Qwen/T5 常驻
    GPU 后剩 ~1GB，整图 decode 工作内存吃满 → OOM。本类在 `decode()` 入口先
    `try` 整图，CUDA OOM 才走 cosine-blend 切块 decode（每 tile 64 latent /
    512 pixel，单 tile 工作峰值约 75MB）。

    大显存路径永远在 `try` 一次就过、零成本；fallback 路径每张图慢 ~30%。

    现有训练代码（dataset 编 latent / phases/dataset.py 训练时重建预览）继续直
    接调 `wrapper.model.encode/decode(z, wrapper.scale)`，保持现有行为不动 ——
    encode 路径与训练侧低分辨率重建不会 OOM。
    """

    # tile 几何：tile=512px / overlap=128px / 4-stage VAE 8× upsample
    _TILE_LATENT = 64
    _STRIDE_LATENT = 48
    _UPSAMPLE = 8

    def __init__(self, model, mean, std):
        self.model = model
        self.mean = mean
        self.std = std
        self.scale = [mean, 1.0 / std]

    def decode(self, z):
        """latent → pixel；整图 OOM 时自动切块重试。

        z: ``[b, 16, t, H, W]`` latent
        return: ``[b, 3, t, H*8, W*8]``（与底层 `WanVAE_.decode` 一致，未 clamp）
        """
        try:
            return self.model.decode(z, self.scale)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning(
                "VAE 整图 decode OOM，回退到 tiled decode "
                "(tile=%dpx, overlap=%dpx)",
                self._TILE_LATENT * self._UPSAMPLE,
                (self._TILE_LATENT - self._STRIDE_LATENT) * self._UPSAMPLE,
            )
            return self._tiled_decode(z)

    @torch.no_grad()
    def _tiled_decode(self, z):
        """在 H/W 维度切块独立 decode，cosine blend 拼回。

        - tile=64 latent，stride=48，overlap=16 latent (128 px)；最后一个 tile
          起点 clamp 到 (size - tile) 防超出
        - 小图（H 或 W < tile）走 ``eff_h/eff_w = min(tile, size)``，等价单 tile 全图
        - blend 用 raised cosine 边缘 ramp，accumulator 走 fp32 防 bf16 精度损失
        - mask 最小值 clamp 1e-6 防角落像素 wsum 除 0
        """
        b, _c, t, H, W = z.shape
        tile = self._TILE_LATENT
        stride = self._STRIDE_LATENT
        up = self._UPSAMPLE

        eff_h = min(tile, H)
        eff_w = min(tile, W)
        hs = _tile_starts(H, eff_h, stride)
        ws = _tile_starts(W, eff_w, stride)

        tile_h_px = eff_h * up
        tile_w_px = eff_w * up
        overlap_px = (tile - stride) * up
        H_px, W_px = H * up, W * up

        acc = torch.zeros(b, 3, t, H_px, W_px, dtype=torch.float32, device=z.device)
        wsum = torch.zeros(1, 1, 1, H_px, W_px, dtype=torch.float32, device=z.device)

        mask = _cosine_blend_mask(tile_h_px, tile_w_px, fade=overlap_px, device=z.device)

        for hi in hs:
            for wi in ws:
                z_tile = z[:, :, :, hi:hi + eff_h, wi:wi + eff_w]
                img_tile = self.model.decode(z_tile, self.scale).float()
                hp, wp = hi * up, wi * up
                acc[:, :, :, hp:hp + tile_h_px, wp:wp + tile_w_px] += img_tile * mask
                wsum[:, :, :, hp:hp + tile_h_px, wp:wp + tile_w_px] += mask

        return (acc / wsum).to(z.dtype)


def _tile_starts(size: int, tile: int, stride: int) -> list[int]:
    """固定 stride 的 tile 起点列表；尾部未覆盖时追加 ``size - tile``。"""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def _cosine_blend_mask(h: int, w: int, *, fade: int, device) -> torch.Tensor:
    """2D raised-cosine mask；边缘 `fade` 像素内 0→1 ramp，中心 1。

    最终 mask 整体 clamp_min(1e-6) 防 wsum 角落除 0 —— 2D 角落是 1D × 1D，
    1D 上 clamp 1e-6 在 2D 角落变成 1e-12 太接近 fp32 denormal，统一在 2D 后 clamp。
    """
    def ramp_1d(n: int) -> torch.Tensor:
        m = torch.ones(n, dtype=torch.float32, device=device)
        if fade > 0:
            t = torch.linspace(math.pi, 2 * math.pi, fade, device=device)
            r = torch.cos(t) * 0.5 + 0.5
            m[:fade] = r
            m[-fade:] = r.flip(0)
        return m
    mh = ramp_1d(h)
    mw = ramp_1d(w)
    mask_2d = (mh[:, None] * mw[None, :]).clamp_min(1e-6)
    return mask_2d[None, None, None, :, :]


def ensure_models_namespace(repo_root):
    """确保 models 命名空间可用。"""
    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))


def load_anima_model(transformer_path, device, dtype, repo_root, *, flash_attn: bool = True):
    """加载 Anima transformer 模型。

    `flash_attn=False` 显式禁用 flash_attn fast path（attention_backend=xformers/none
    时由 caller 传入），让 caller 完全决定 attention 实现 —— PR #17 那版默认
    fn(True) 强制开 flash_attn 不让用户关，与 cfg.attention_backend 解耦不彻底。
    """
    from safetensors import safe_open

    ensure_models_namespace(repo_root)

    # 加载模型类
    cosmos_modeling = load_module_from_path(
        "cosmos_predict2_modeling",
        repo_root / "cosmos_predict2_modeling.py",
    )
    anima_modeling = load_module_from_path(
        "anima_modeling",
        repo_root / "anima_modeling.py",
    )
    Anima = anima_modeling.Anima

    # flash_attn 全局开关：set_flash_attn_enabled 内部检查 _FLASH_ATTN_AVAILABLE，
    # 未装时返回 False 不抛错（idempotent）。用 caller 传入的 flash_attn 而不是
    # 强制 True，让 attention_backend=none/xformers 时显式关掉。
    fn = getattr(cosmos_modeling, "set_flash_attn_enabled", None)
    if fn is not None:
        try:
            if fn(flash_attn):
                logger.info("flash_attn 启用（训练 + sample 走 fast path）")
            else:
                logger.info("flash_attn 关闭（attention_backend=%s 或包未安装）",
                            "flash_attn" if flash_attn else "non-flash")
        except Exception as exc:  # noqa: BLE001
            logger.warning("flash_attn 启用失败，继续走 SDPA fallback: %s", exc)

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


def load_vae(vae_path, device, dtype, repo_root):
    """加载 VAE。"""
    wan_vae = load_module_from_path("wan_vae", repo_root / "wan" / "vae2_1.py")
    WanVAE = wan_vae.WanVAE_

    cfg = dict(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True], dropout=0.0,
    )

    model = WanVAE(**cfg).eval().requires_grad_(False)

    sd = _load_safetensors_state_dict(Path(vae_path))
    _load_weights_best_effort(model, sd, label="VAE")
    model = model.to(device=device, dtype=dtype)

    # VAE 归一化参数
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=dtype, device=device)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=dtype, device=device)

    logger.info("VAE 加载完成")
    return VAEWrapper(model, mean, std)


def load_text_encoders(qwen_path, t5_tokenizer_path, device, dtype):
    """加载文本编码器（Qwen + T5）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    # Qwen
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval().requires_grad_(False)

    # T5 tokenizer
    if t5_tokenizer_path and Path(t5_tokenizer_path).exists():
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_path)
    else:
        t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

    logger.info("文本编码器加载完成")
    return qwen_model, qwen_tokenizer, t5_tokenizer
