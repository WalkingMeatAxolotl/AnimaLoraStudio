"""VAEWrapper OOM-fallback tiled decode 单测（issue #200）。

测：
- `_tile_starts` 边界 case（含 1024 latent / 128 latent 实际 issue 规格）
- `_cosine_blend_mask` 对称性 + clamp_min 避免角落除 0
- `VAEWrapper.decode` 走 try-full 路径不触发 tile
- `VAEWrapper.decode` 在 model.decode 抛 OOM 时走 tile fallback 并产出形状一致的结果
- `_tiled_decode` 拼接：mock 一个线性 decode（z 上 nearest-upsample 8× 取前 3 通道），
  tile 重建结果与整图重建数值一致（验证 cosine blend + accumulator 正确性）
"""
from __future__ import annotations

import logging

import pytest
import torch

from training.models import (
    VAEWrapper,
    _cosine_blend_mask,
    _tile_starts,
)


# ---------------------------------------------------------------------------
# _tile_starts
# ---------------------------------------------------------------------------


def test_tile_starts_size_le_tile_returns_zero_only() -> None:
    assert _tile_starts(64, 64, 48) == [0]
    assert _tile_starts(32, 64, 48) == [0]


def test_tile_starts_appends_boundary_when_stride_doesnt_cover() -> None:
    """128 latent (1024 px reg) 的实际规格：tile=64 stride=48 → [0, 48, 64]。"""
    assert _tile_starts(128, 64, 48) == [0, 48, 64]


def test_tile_starts_exact_stride_no_duplicate_tail() -> None:
    """stride 恰好整除时尾部不重复追加。"""
    # size=160 tile=64 stride=48 → range(0, 97, 48) = [0, 48, 96]，96+64=160 命中尾部
    assert _tile_starts(160, 64, 48) == [0, 48, 96]


# ---------------------------------------------------------------------------
# _cosine_blend_mask
# ---------------------------------------------------------------------------


def test_cosine_blend_mask_shape_and_dtype() -> None:
    m = _cosine_blend_mask(512, 512, fade=128, device="cpu")
    assert m.shape == (1, 1, 1, 512, 512)
    assert m.dtype == torch.float32


def test_cosine_blend_mask_center_one_edges_clamped() -> None:
    """中心 = 1，边缘 ≈ 1e-6（防 wsum 除 0；fp32 clamp 实测略低于 1e-6）。"""
    m = _cosine_blend_mask(512, 512, fade=128, device="cpu")[0, 0, 0]
    # 中心
    assert m[256, 256].item() == pytest.approx(1.0, abs=1e-5)
    # 四角：clamp 在 fp32 精度下 ≈ 1e-6（容差 1e-7）
    assert m[0, 0].item() == pytest.approx(1e-6, abs=1e-7)
    assert m[-1, -1].item() == pytest.approx(1e-6, abs=1e-7)
    assert m[0, -1].item() == pytest.approx(1e-6, abs=1e-7)


def test_cosine_blend_mask_symmetric() -> None:
    m = _cosine_blend_mask(256, 256, fade=64, device="cpu")[0, 0, 0]
    # 上下对称
    assert torch.allclose(m, m.flip(0), atol=1e-6)
    # 左右对称
    assert torch.allclose(m, m.flip(1), atol=1e-6)


def test_cosine_blend_mask_zero_fade_all_ones() -> None:
    m = _cosine_blend_mask(64, 64, fade=0, device="cpu")
    assert torch.all(m == 1.0)


# ---------------------------------------------------------------------------
# VAEWrapper.decode 走 try-full 路径
# ---------------------------------------------------------------------------


class _RecordingModel:
    """记录 decode 调用次数 + 形状的 mock；行为是 nearest upsample 8× + 取前 3 通道。"""

    def __init__(self, oom_on_full: bool = False, oom_on_full_enc: bool = False):
        self.calls: list[tuple[int, ...]] = []
        self.enc_calls: list[tuple[int, ...]] = []
        self.oom_on_full = oom_on_full
        self.oom_on_full_enc = oom_on_full_enc

    def encode(self, pixels: torch.Tensor, scale) -> torch.Tensor:
        """线性 mock：8× avg_pool 下采样 + 3ch→16ch 补零（与 decode 的 nearest 互逆近似）。

        avg_pool stride=8、tile 起点恒为 8 倍数 → 每个输出 latent 像素对应固定 8×8 输入块，
        分块与整图逐块一致，blend 拼接可高精度复原。
        """
        self.enc_calls.append(tuple(pixels.shape))
        b, _c, t, H, W = pixels.shape
        if self.oom_on_full_enc and H >= 1024 and len(self.enc_calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM on full encode")
        z = torch.nn.functional.avg_pool3d(pixels, kernel_size=(1, 8, 8))  # [b,3,t,H/8,W/8]
        if z.shape[1] < 16:
            z = torch.nn.functional.pad(z, (0, 0, 0, 0, 0, 0, 0, 16 - z.shape[1]))  # 3ch→16ch
        return z

    def decode(self, z: torch.Tensor, scale) -> torch.Tensor:
        self.calls.append(tuple(z.shape))
        b, _c, t, h, w = z.shape
        # 模拟 WanVAE_ 的 8× spatial upsample + 16ch → 3ch
        # 故意用 z 的前 3 通道 nearest upsample，方便 tile / full 对比
        upsampled = torch.nn.functional.interpolate(
            z[:, :3].reshape(b * t, 3, h, w),
            scale_factor=8,
            mode="nearest",
        ).reshape(b, 3, t, h * 8, w * 8)
        if self.oom_on_full and h >= 128 and len(self.calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM on full decode")
        return upsampled


def _make_wrapper(model) -> VAEWrapper:
    mean = torch.zeros(16)
    std = torch.ones(16)
    return VAEWrapper(model, mean, std)


def test_decode_full_path_calls_model_once() -> None:
    """大显存路径：try full 成功 → 不进 tile，model.decode 只调一次。"""
    model = _RecordingModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    z = torch.randn(1, 16, 1, 32, 32)  # 256×256 输出
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 256, 256)
    assert len(model.calls) == 1


# ---------------------------------------------------------------------------
# VAEWrapper.decode OOM fallback
# ---------------------------------------------------------------------------


def test_decode_oom_fallback_invokes_tiled_decode() -> None:
    """整图 OOM → catch + empty_cache + tile 多次调 model.decode。

    1024 reg 规格：z = [1,16,1,128,128]；tile=64 stride=48 →
    tile 起点 hs=ws=[0,48,64] → 3×3 = 9 个 tile decode 调用，
    加上第一次失败的整图调用 = 10。
    """
    model = _RecordingModel(oom_on_full=True)
    wrapper = _make_wrapper(model)
    z = torch.zeros(1, 16, 1, 128, 128)  # 1024×1024 reg
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 1024, 1024)
    # 1 次失败 full + 9 个 tile
    assert len(model.calls) == 1 + 9
    # 第一次是 full
    assert model.calls[0] == (1, 16, 1, 128, 128)
    # 后续都是 tile 大小
    for shape in model.calls[1:]:
        assert shape == (1, 16, 1, 64, 64)


# ---------------------------------------------------------------------------
# Tile 拼接数值正确性
# ---------------------------------------------------------------------------


def test_tiled_decode_reconstructs_full_for_linear_decoder() -> None:
    """对一个线性 decode（nearest upsample 8×），tile + cosine blend 拼接
    结果与整图 decode 数值一致（accumulator 归一化正确）。

    用 latent[:, :3] 作 "image"，nearest 8× 上采。每个像素的实际值与位置无
    关，blend mask 加权累加后归一化必须恢复原值。
    """
    model = _RecordingModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    torch.manual_seed(0)
    z = torch.randn(1, 16, 1, 128, 128)

    full = wrapper.decode(z)
    tiled = wrapper._tiled_decode(z)

    # 数值应几乎一致；mask clamp_min 1e-6 + fp32 累加只引入极小数值误差
    assert tiled.shape == full.shape
    assert torch.allclose(tiled, full, atol=1e-4)


def test_tiled_decode_handles_small_input_without_tiling() -> None:
    """size ≤ tile 时 tile_starts 返回 [0]，等同整图 decode。"""
    model = _RecordingModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    z = torch.randn(1, 16, 1, 32, 32)  # < tile=64
    out = wrapper._tiled_decode(z)
    assert out.shape == (1, 3, 1, 256, 256)


# ---------------------------------------------------------------------------
# tiling 模式（auto / on / off）+ 峰值估算 + auto 判定阈值
# ---------------------------------------------------------------------------


def test_tiling_on_always_tiles_no_full_call() -> None:
    """tiling='on'：直接走 _tiled_decode，没有整图 decode 调用。

    128 latent → tile 起点 [0,48,64] → 9 个 64×64 tile，且首调不是 128 整图。
    """
    model = _RecordingModel(oom_on_full=False)
    wrapper = VAEWrapper(model, torch.zeros(16), torch.ones(16), tiling="on")
    z = torch.zeros(1, 16, 1, 128, 128)
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 1024, 1024)
    assert len(model.calls) == 9
    assert all(s == (1, 16, 1, 64, 64) for s in model.calls)


def test_tiling_off_uses_whole_image_then_oom_net() -> None:
    """tiling='off'：整图优先；仍保留真 OOM 兜底分块（小显存安全网）。"""
    model = _RecordingModel(oom_on_full=True)
    wrapper = VAEWrapper(model, torch.zeros(16), torch.ones(16), tiling="off")
    z = torch.zeros(1, 16, 1, 128, 128)
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 1024, 1024)
    # 1 次失败整图 + 9 个 tile（OOM 兜底仍在）
    assert model.calls[0] == (1, 16, 1, 128, 128)
    assert len(model.calls) == 1 + 9


def test_est_decode_peak_scales_with_pixels_and_dtype() -> None:
    """峰值估算 ∝ 输出像素 × 元素大小。fp32 1024² ≈ 11.5G，bf16 减半。"""
    model = _RecordingModel()
    wrapper = _make_wrapper(model)
    z32 = torch.zeros(1, 16, 1, 128, 128, dtype=torch.float32)   # 1024² 输出
    est_fp32 = wrapper._est_decode_peak_bytes(z32)
    assert est_fp32 == pytest.approx(11000 * 1024 * 1024, rel=1e-6)
    z16 = torch.zeros(1, 16, 1, 128, 128, dtype=torch.bfloat16)
    assert wrapper._est_decode_peak_bytes(z16) == pytest.approx(est_fp32 / 2, rel=1e-6)


def test_should_auto_tile_threshold_matches_measured_cliff() -> None:
    """auto 阈值复刻实测崖（RTX 5090 31.8G）：fp32 1536² 分块、其余快路径整图。"""
    GB = 1024 ** 3
    total = int(31.8 * GB)
    model = _RecordingModel()
    w = _make_wrapper(model)

    def est(res, dtype):
        h = res // 8
        return w._est_decode_peak_bytes(torch.zeros(1, 16, 1, h, h, dtype=dtype))

    light = int(2.2 * GB)  # 仅 VAE 常驻
    # fp32 1024（实测 0.34s 整图安全）→ 不分块
    assert not VAEWrapper._should_auto_tile(light, est(1024, torch.float32), total)
    # fp32 1536（实测整图 196s）→ 分块
    assert VAEWrapper._should_auto_tile(light, est(1536, torch.float32), total)
    # bf16 1536（实测 0.4s 整图安全）→ 不分块
    assert not VAEWrapper._should_auto_tile(light, est(1536, torch.bfloat16), total)
    # bf16 1536 但叠加 ~12G 常驻模型（训练 sample 场景）→ 分块
    heavy = int(13 * GB)
    assert VAEWrapper._should_auto_tile(heavy, est(1536, torch.bfloat16), total)


# ---------------------------------------------------------------------------
# encode 分块（latent 缓存路径）
# ---------------------------------------------------------------------------


def test_encode_full_path_calls_model_once() -> None:
    """CPU（非 cuda）走整图 encode：model.encode 只调一次。"""
    model = _RecordingModel()
    wrapper = _make_wrapper(model)
    px = torch.randn(1, 3, 1, 256, 256)
    z = wrapper.encode(px)
    assert z.shape == (1, 16, 1, 32, 32)
    assert len(model.enc_calls) == 1


def test_encode_tiling_on_tiles_in_pixel_space() -> None:
    """tiling='on'：1024px → 像素 tile=512/stride=384 起点 [0,384,512] → 9 个 512² tile。"""
    model = _RecordingModel()
    wrapper = VAEWrapper(model, torch.zeros(16), torch.ones(16), tiling="on")
    px = torch.zeros(1, 3, 1, 1024, 1024)
    z = wrapper.encode(px)
    assert z.shape == (1, 16, 1, 128, 128)
    assert len(model.enc_calls) == 9
    assert all(s == (1, 3, 1, 512, 512) for s in model.enc_calls)


def test_encode_oom_fallback_invokes_tiled_encode() -> None:
    """整图 encode OOM → catch + tile 兜底（off 模式也保留安全网）。"""
    model = _RecordingModel(oom_on_full_enc=True)
    wrapper = VAEWrapper(model, torch.zeros(16), torch.ones(16), tiling="off")
    px = torch.zeros(1, 3, 1, 1024, 1024)
    z = wrapper.encode(px)
    assert z.shape == (1, 16, 1, 128, 128)
    assert model.enc_calls[0] == (1, 3, 1, 1024, 1024)  # 先试整图
    assert len(model.enc_calls) == 1 + 9                 # 失败 + 9 tile


def test_tiled_encode_reconstructs_full_for_linear_encoder() -> None:
    """线性 encode（avg_pool）下，tile + cosine blend 拼接 ≈ 整图 encode。"""
    model = _RecordingModel()
    wrapper = _make_wrapper(model)
    torch.manual_seed(0)
    px = torch.randn(1, 3, 1, 1024, 1024)
    full = wrapper.model.encode(px, wrapper.scale)
    tiled = wrapper._tiled_encode(px)
    assert tiled.shape == full.shape
    assert torch.allclose(tiled, full, atol=1e-4)


def test_should_offload_for_whole_decode_false_on_cpu() -> None:
    """CPU 张量（无 cuda）下不 offload：守卫返回 False，不触碰 mem_get_info。"""
    wrapper = _make_wrapper(_RecordingModel())
    z = torch.zeros(1, 16, 1, 128, 128)  # CPU
    assert wrapper.should_offload_for_whole_decode(z) is False


def test_est_encode_peak_scales_with_pixels_and_dtype() -> None:
    """encode 峰值估算 ∝ 输入像素 × 元素大小。fp32 1024² ≈ 5.5G，bf16 减半。"""
    wrapper = _make_wrapper(_RecordingModel())
    px32 = torch.zeros(1, 3, 1, 1024, 1024, dtype=torch.float32)
    est_fp32 = wrapper._est_encode_peak_bytes(px32)
    assert est_fp32 == pytest.approx(5500 * 1024 * 1024, rel=1e-6)
    px16 = torch.zeros(1, 3, 1, 1024, 1024, dtype=torch.bfloat16)
    assert wrapper._est_encode_peak_bytes(px16) == pytest.approx(est_fp32 / 2, rel=1e-6)


# ---------------------------------------------------------------------------
# 分块决策 / OOM 回退日志去重（缓存逐图调用不刷屏）
# ---------------------------------------------------------------------------


def test_log_once_same_key_logs_only_first(caplog) -> None:
    """同一 key 调多次只记一次：latent 缓存逐 bucket 调 encode 不刷屏。"""
    wrapper = _make_wrapper(_RecordingModel())
    logger = logging.getLogger("training.models")
    with caplog.at_level(logging.INFO, logger="training.models"):
        for i in range(200):
            wrapper._log_once("auto_encode", logger.info, "主动分块 #%d", i)
    hits = [r for r in caplog.records if "主动分块" in r.getMessage()]
    assert len(hits) == 1
    assert "#0" in hits[0].getMessage()  # 记的是首次那条


def test_log_once_distinct_keys_each_logged_once(caplog) -> None:
    """不同事件（encode/decode 决策、OOM 回退）各自独立去重，互不抑制。"""
    wrapper = _make_wrapper(_RecordingModel())
    logger = logging.getLogger("training.models")
    with caplog.at_level(logging.WARNING, logger="training.models"):
        wrapper._log_once("auto_encode", logger.warning, "encode 分块")
        wrapper._log_once("auto_decode", logger.warning, "decode 分块")
        wrapper._log_once("auto_encode", logger.warning, "encode 分块")  # 重复，抑制
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == ["encode 分块", "decode 分块"]
