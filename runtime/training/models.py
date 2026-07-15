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
)


logger = logging.getLogger(__name__)


class VAEWrapper:
    """WAN VAE + 归一化参数 + 整图 decode OOM 自动回退到 tiled decode。

    Issue #200：8GB 类小显存跑 1024×1024 reg 生成时 transformer/Qwen/T5 常驻
    GPU 后剩 ~1GB，整图 decode 工作内存吃满 → OOM。本类在 `decode()` 入口先
    `try` 整图，CUDA OOM 才走 cosine-blend 切块 decode（每 tile 64 latent /
    512 pixel，单 tile 工作峰值约 75MB）。

    后续（VRAM 崖修复）：`decode()` / `encode()` 都按 `tiling`（auto/on/off）决策，
    auto 在「已用 + 预计峰值」越过总显存阈值时主动分块——不只是等 OOM。大显存卡上
    整图 op 接近占满显存会触发 WDDM 显存换页、单次 op 从 <1s 退化到上百秒，且不抛
    干净 OOM，reactive 兜底救不了，故需 proactive。

    调用方应走 `wrapper.encode(pixels)` / `wrapper.decode(z)`（带分块决策），不要直接
    调 `wrapper.model.encode/decode(...)`（绕过分块）。
    """

    # tile 几何：tile=512px / overlap=128px / 4-stage VAE 8× upsample
    _TILE_LATENT = 64
    _STRIDE_LATENT = 48
    _UPSAMPLE = 8

    # 整图 decode 峰值显存 ≈ _DECODE_PEAK_BYTES_PER_OUT_PX × (elem/4) × 输出像素 × B × T。
    # 标定自 tools/spike/vae_stress.py（RTX 5090 / WAN VAE dim=96）：fp32 1024²≈10.4G、
    # 1536²≈22.6G；bf16 减半。取 11000（实测 ~9.8k）留 ~12% 余量。
    _DECODE_PEAK_BYTES_PER_OUT_PX = 11000
    # 整图 encode 峰值 ≈ _ENCODE_PEAK_BYTES_PER_IN_PX × (elem/4) × 输入像素 × B × T。
    # 标定自 tools/spike/vae_stress.py（fp32）：1024²≈5.5G、1536²≈11.6G、2048²≈20.3G。
    _ENCODE_PEAK_BYTES_PER_IN_PX = 5500
    # auto：当「当前已用 + 预计峰值」超过总显存此比例就分块。崖在 ~50% 而非满显存——
    # fp32 1536²(峰值 22.6G / 总 31.8G = 71%) 即便「装得下」也会因 WDDM 显存换页退化到
    # ~190s；fp32 1024²(10.4G / 33%) 正常。0.5 让快路径(fp32≤1024 / bf16≤1536)走整图，
    # 把会撞崖的(大图、fp32、或叠加常驻模型)切到分块。
    _TILE_VRAM_FRACTION = 0.5

    def __init__(self, model, mean, std, tiling: str = "auto"):
        self.model = model
        self.mean = mean
        self.std = std
        self.scale = [mean, 1.0 / std]
        # VAE 权重精度（mean/std 与 model 同 dtype）。encode/decode 入口按此 cast 输入，
        # 这样 fp16 训练 + fp32 VAE 时调用方传 fp16 latent / pixel 也不会 dtype mismatch。
        self.dtype = mean.dtype
        # 分块模式：
        #   auto（默认）= 按 free VRAM 估算，整图峰值逼近可用显存时主动分块；
        #   on          = 始终分块（省显存，慢约 30%）；
        #   off          = 整图，仅真 OOM 时回退分块（旧行为）。
        # auto 解决大显存卡整图 decode 接近占满时触发「系统内存回退」→ 单次 decode
        # 从 <1s 退化到上百秒的卡死（reactive OOM 兜底救不了，因为没抛干净 OOM）。
        self.tiling = str(tiling or "auto").lower().strip()
        # 分块决策 / OOM 回退日志去重：latent 缓存逐 bucket 调 encode（200 张不同尺寸
        # 的图 → 200 次调用），每次都记会刷屏。同一 wrapper 实例下每种事件只记一次。
        self._logged_once: set[str] = set()

    def _log_once(self, key: str, log_fn, msg: str, *args) -> None:
        """同一 wrapper 实例下，相同 ``key`` 的日志只发一次（其余静默）。

        缓存阶段 transformer/Qwen 已驻留 GPU（models_phase 早于 dataset_phase），
        free 显存低 → auto 判定对几乎每张图都分块。分块本身是对的（整图 op 会把
        占用推过 WDDM 显存换页崖），只是逐图重复记日志没有信息量，故去重。
        """
        if key in self._logged_once:
            return
        self._logged_once.add(key)
        log_fn(msg, *args)

    def _est_decode_peak_bytes(self, z) -> int:
        b, _c, t, H, W = z.shape
        out_px = (H * self._UPSAMPLE) * (W * self._UPSAMPLE)
        elem = z.element_size()  # 2=bf16/fp16, 4=fp32
        return int(self._DECODE_PEAK_BYTES_PER_OUT_PX * (elem / 4.0) * out_px * b * max(1, t))

    def _est_encode_peak_bytes(self, pixels) -> int:
        b, _c, t, H, W = pixels.shape  # H/W 为像素分辨率
        in_px = H * W
        elem = pixels.element_size()
        return int(self._ENCODE_PEAK_BYTES_PER_IN_PX * (elem / 4.0) * in_px * b * max(1, t))

    @classmethod
    def _should_auto_tile(cls, used_bytes: int, est_peak_bytes: int, total_bytes: int) -> bool:
        """auto 判定：当前已用 + 预计 decode 峰值 是否越过总显存的分块阈值。"""
        return (used_bytes + est_peak_bytes) > total_bytes * cls._TILE_VRAM_FRACTION

    def should_offload_for_whole_decode(self, z) -> bool:
        """采样 decode 前是否值得把非活跃模块（DiT/Qwen）挪到 CPU 腾显存。

        仅当「显存紧张（当前会分块）**且** 峰值仍在崖下」时 True：此时腾出常驻模块
        就能整图 decode、保住 parity / 无 tile 缝。峰值已越崖（如 fp32 1536²）时 False
        —— 腾显存也救不了整图（崖按总显存比例算、与 free 无关），交给分块更省系统内存。
        """
        if not (torch.is_tensor(z) and z.is_cuda and torch.cuda.is_available()):
            return False
        free, total = torch.cuda.mem_get_info()
        est = self._est_decode_peak_bytes(z)
        return self._should_auto_tile(total - free, est, total) and est < total * self._TILE_VRAM_FRACTION

    def decode(self, z):
        """latent → pixel；按 self.tiling 决定整图 / 分块。

        z: ``[b, 16, t, H, W]`` latent
        return: ``[b, 3, t, H*8, W*8]``（与底层 `WanVAE_.decode` 一致，未 clamp）
        """
        z = z.to(self.dtype)  # 对齐 VAE 权重精度（fp16 训练 / fp32 VAE；dtype 一致时为 no-op）
        if self.tiling == "on":
            return self._tiled_decode(z)

        if self.tiling == "auto" and z.is_cuda and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            est = self._est_decode_peak_bytes(z)
            if self._should_auto_tile(used, est, total):
                # debug 级：缓存阶段模型驻留 → free 低 → 几乎张张分块属正常，
                # 默认不显示，免得「已用 X > 总显存」被误读成显存不足。
                self._log_once(
                    "auto_decode", logger.debug,
                    "VAE decode 主动分块（tiling=auto）：已用 %.1fG + 预计峰值 %.1fG > 总显存 %.1fG×%.2f",
                    used / 1024 ** 3, est / 1024 ** 3, total / 1024 ** 3, self._TILE_VRAM_FRACTION,
                )
                return self._tiled_decode(z)

        # off，或 auto 判定显存够：整图，仍保留 OOM 兜底（小显存卡的安全网）。
        try:
            return self.model.decode(z, self.scale)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log_once(
                "oom_decode", logger.warning,
                "VAE 整图 decode OOM，回退到 tiled decode "
                "(tile=%dpx, overlap=%dpx)（同类后续不再重复）",
                self._TILE_LATENT * self._UPSAMPLE,
                (self._TILE_LATENT - self._STRIDE_LATENT) * self._UPSAMPLE,
            )
            return self._tiled_decode(z)

    def encode(self, pixels):
        """pixel → latent；按 self.tiling 决定整图 / 分块。

        pixels: ``[b, 3, t, H, W]``（H/W 为像素分辨率，须为 8 的倍数）
        return: ``[b, 16, t, H/8, W/8]`` latent（与 `WanVAE_.encode` 一致）
        """
        pixels = pixels.to(self.dtype)  # 对齐 VAE 权重精度（同 decode；dtype 一致时为 no-op）
        if self.tiling == "on":
            return self._tiled_encode(pixels)

        if self.tiling == "auto" and pixels.is_cuda and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            est = self._est_encode_peak_bytes(pixels)
            if self._should_auto_tile(used, est, total):
                # debug 级：缓存阶段模型驻留 → free 低 → 几乎张张分块属正常，
                # 默认不显示，免得「已用 X > 总显存」被误读成显存不足。
                self._log_once(
                    "auto_encode", logger.debug,
                    "VAE encode 主动分块（tiling=auto）：已用 %.1fG + 预计峰值 %.1fG > 总显存 %.1fG×%.2f",
                    used / 1024 ** 3, est / 1024 ** 3, total / 1024 ** 3, self._TILE_VRAM_FRACTION,
                )
                return self._tiled_encode(pixels)

        try:
            return self.model.encode(pixels, self.scale)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log_once(
                "oom_encode", logger.warning,
                "VAE 整图 encode OOM，回退到 tiled encode "
                "(tile=%dpx, overlap=%dpx)（同类后续不再重复）",
                self._TILE_LATENT * self._UPSAMPLE,
                (self._TILE_LATENT - self._STRIDE_LATENT) * self._UPSAMPLE,
            )
            return self._tiled_encode(pixels)

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

    @torch.no_grad()
    def _tiled_encode(self, pixels, tile_px=None, overlap_px=None):
        """在 H/W 维度切块独立 encode，latent 空间 cosine blend 拼回。

        - 默认 tile=512px / stride=384px / overlap=128px（= decode 的 64/48/16 latent ×8）
        - ``tile_px`` / ``overlap_px``（像素）非 None 时覆盖默认，供缓存分块按 config 传入；
          二者须为 VAE 下采样（``up``）的整倍数（latent 块边界落整格）
        - 像素起点恒为 8 的倍数 → latent 位置为整数（_TILE/_STRIDE×8 与 H/W 均整除 8）
        - blend 在 latent 空间，accumulator 走 fp32
        - encode 非线性，拼接为近似（overlap+cosine 抹平 tile 边界），用于 latent 缓存足够
        """
        b, _c, t, H, W = pixels.shape
        up = self._UPSAMPLE
        if tile_px is None:
            tile_px = self._TILE_LATENT * up
            stride_px = self._STRIDE_LATENT * up
        else:
            tile_px = int(tile_px)
            ov_px = int(overlap_px) if overlap_px is not None else (self._TILE_LATENT - self._STRIDE_LATENT) * up
            for _name, _v in (("tile_px", tile_px), ("overlap_px", ov_px)):
                if _v % up != 0:
                    raise ValueError(
                        f"_tiled_encode 要求 {_name}={_v} 是 VAE 下采样 {up} 的整倍数"
                        "（latent 块边界需落整格）。"
                    )
            if not (0 <= ov_px < tile_px):
                raise ValueError(f"overlap_px={ov_px} 必须满足 0 <= overlap < tile_px={tile_px}")
            stride_px = tile_px - ov_px

        eff_h = min(tile_px, H)
        eff_w = min(tile_px, W)
        hs = _tile_starts(H, eff_h, stride_px)
        ws = _tile_starts(W, eff_w, stride_px)

        lat_h, lat_w = H // up, W // up
        tile_lh, tile_lw = eff_h // up, eff_w // up
        overlap_lat = (tile_px - stride_px) // up

        # encode 输出通道 = z_dim(16)。从第一个 tile 拿真实通道数更稳，但 WAN VAE 固定 16。
        acc = torch.zeros(b, 16, t, lat_h, lat_w, dtype=torch.float32, device=pixels.device)
        wsum = torch.zeros(1, 1, 1, lat_h, lat_w, dtype=torch.float32, device=pixels.device)

        mask = _cosine_blend_mask(tile_lh, tile_lw, fade=overlap_lat, device=pixels.device)

        for hi in hs:
            for wi in ws:
                px_tile = pixels[:, :, :, hi:hi + eff_h, wi:wi + eff_w]
                z_tile = self.model.encode(px_tile, self.scale).float()
                lh, lw = hi // up, wi // up
                acc[:, :, :, lh:lh + tile_lh, lw:lw + tile_lw] += z_tile * mask
                wsum[:, :, :, lh:lh + tile_lh, lw:lw + tile_lw] += mask

        return (acc / wsum).to(pixels.dtype)


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


def load_vae(vae_path, device, dtype, repo_root, *, tiling: str = "auto"):
    """加载 VAE。``tiling`` 透传给 VAEWrapper（auto/on/off）。"""
    # 正常 import（exec-load 退役，多模型 PR-2a）；repo_root 参数保留但不再使用
    from modeling.wan.vae2_1 import WanVAE_ as WanVAE

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
    return VAEWrapper(model, mean, std, tiling=tiling)
