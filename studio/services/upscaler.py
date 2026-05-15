"""图片放大服务（预处理阶段）。

把 spandrel + tiled inference 包成单文件 API：

- `load_model(path)`：缓存式加载 ESRGAN/RRDB 权重（spandrel 自动识别架构）
- `tiled_inference(model, img, *, scale, tile_size, tile_pad)`：分块前向 +
  overlap 拼接，保证 VRAM 上界跟 tile_size 线性相关
- `upscale_file(src, dst, *, model_path, ...)`：完整文件级 API，自动处理
  Pillow 解码、设备选择、产物写盘 + 元数据 sidecar

设计：
- 模型只在第一次调用 load_model 时实例化；二次调相同 path 直接复用缓存
  （进程级 dict，跟 wd14_tagger 同思路）
- 设备策略：device='auto' → 有 cuda 用 cuda，否则 cpu；显式 'cuda' 但无 GPU
  时自动降级到 cpu 并 log warning（避免子进程直接崩）
- tile_size 单位是 **输入像素**；4x 模型 tile=256 → 输出 1024×1024 一块。
  VRAM 峰值约 `tile**2 * scale**2 * 4byte * batch * 7倍中间张量`，256 时
  大约 2-3GB
- tile_pad 是为了消除拼接边界，默认 16 像素重叠（spandrel 推荐值）
- 单测用 stub model 走通整条流水线，不需要真权重
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# spandrel `ImageModelDescriptor` 的最小协议：我们只用到 `.model` 和 `.scale`。
# 缓存键 = 绝对路径字符串。注意：换模型权重文件而保留同名时缓存会脏 —
# 实际场景里下载完不会动权重文件，可接受。需要清缓存就调 `clear_cache()`。
_MODEL_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 设备与模型加载
# ---------------------------------------------------------------------------


def resolve_device(device: str = "auto") -> torch.device:
    """device='auto' → cuda 可用就用 cuda，否则 cpu。显式 'cuda' 无 GPU 时降级。"""
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("requested cuda but cuda not available; falling back to cpu")
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    """precision='auto' → cuda 上 fp16，cpu 上 fp32。

    fp16 对 ESRGAN/RRDB 这类 CNN 视觉模型几乎无肉眼差异，但 GPU 上速度
    通常 1.6-2× ；ComfyUI 默认也走 fp16。
    bf16 在 sm_80+ (Ampere 及以后) 可用，对数值范围更友好但 RTX 20 / Tesla 不支持。
    """
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    # auto
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_model(
    model_path: Path,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Any:
    """加载 spandrel ImageModelDescriptor，进程级缓存。

    描述符暴露：
        .model — torch.nn.Module，可直接前向
        .scale — 整数放大倍率（4x-AnimeSharp 是 4）
        .input_channels / .output_channels — 通常 3

    传 dtype 时把模型权重 cast 过去（fp16 在 GPU 上典型 1.6-2× 提速）。
    """
    if not model_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {model_path}")

    key = str(model_path.resolve())
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        if device is not None:
            cached.model.to(device)
        if dtype is not None:
            cached.model.to(dtype)
        return cached

    try:
        from spandrel import ModelLoader
    except ImportError as exc:
        raise RuntimeError(
            "spandrel 未安装（pip install spandrel）。预处理放大需要它。"
        ) from exc

    descriptor = ModelLoader().load_from_file(str(model_path))
    descriptor.model.eval()
    if device is not None:
        descriptor.model.to(device)
    if dtype is not None:
        descriptor.model.to(dtype)
    _MODEL_CACHE[key] = descriptor
    return descriptor


def clear_cache() -> None:
    """清空模型缓存（测试 / 切换 device 时用）。"""
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------


def resize_to_area(img: Image.Image, target_area: int) -> Image.Image:
    """保 aspect ratio 缩放到 `~target_area` 像素的 LANCZOS resize。

    new_W * new_H ≈ target_area；保留 W/H 比。不 snap 到 64 倍数 — Kohya 内部
    还要按桶再缩 / 裁，提前 snap 只会减档限制。返回新的 PIL Image。
    """
    w, h = img.size
    if w <= 0 or h <= 0 or target_area <= 0:
        return img
    ratio = math.sqrt(target_area / (w * h))
    new_w = max(1, round(w * ratio))
    new_h = max(1, round(h * ratio))
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.LANCZOS)


# 已够大「跳过模型」的下限：面积 >= target_area × SKIP_RATIO 时直接 LANCZOS 缩。
# 0.95 = 容忍 5% 像素缺口走纯 LANCZOS（轻微上采样）— 视觉差别看不出，省掉
# 一次模型推理（贵）。再低就该走模型保细节了。
SKIP_MODEL_RATIO = 0.95


def _img_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → float32 BCHW [0,1]。RGBA 转 RGB（alpha 直接丢，4x-AnimeSharp 不
    处理透明通道；上游若需要保留 alpha 应改先 composite 到白底）。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    import numpy as np

    arr = np.array(img, dtype=np.float32) / 255.0  # HWC
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1CHW
    return t


def _tensor_to_img(t: torch.Tensor) -> Image.Image:
    """1CHW 或 CHW float[0,1] → PIL RGB。fp16/bf16 自动 cast 回 fp32 再转 numpy。"""
    import numpy as np

    if t.dim() == 4:
        t = t.squeeze(0)
    # numpy 不直接支持 bf16；fp16 能直接转但乘 255 时精度不够 — 统一回 fp32
    arr = t.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()  # HWC
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def tiled_inference(
    model: Callable[[torch.Tensor], torch.Tensor],
    img: torch.Tensor,
    *,
    scale: int,
    tile_size: int = 256,
    tile_pad: int = 16,
) -> torch.Tensor:
    """分块前向：每块 `tile_size+2*tile_pad` 输入，输出去掉 pad 部分后拼回。

    img: 1CHW float[0,1] tensor（已在目标 device）
    返回 1CHW float tensor，HW 都 ×scale。

    无 tile（tile_size <= 0）走单次整图前向（小图 / 显存够时省 IO 开销）。
    """
    if tile_size <= 0:
        with torch.inference_mode():
            return model(img)

    assert img.dim() == 4 and img.shape[0] == 1, f"need 1CHW tensor, got {img.shape}"
    _, c, h, w = img.shape
    out_h, out_w = h * scale, w * scale
    output = torch.zeros((1, c, out_h, out_w), dtype=img.dtype, device=img.device)

    # tile 网格按 tile_size 步进；边界 tile 不足时截断
    n_y = (h + tile_size - 1) // tile_size
    n_x = (w + tile_size - 1) // tile_size

    with torch.inference_mode():
        for ty in range(n_y):
            for tx in range(n_x):
                y0 = ty * tile_size
                x0 = tx * tile_size
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)

                # pad 区域（用于消除拼接边界）
                py0 = max(y0 - tile_pad, 0)
                px0 = max(x0 - tile_pad, 0)
                py1 = min(y1 + tile_pad, h)
                px1 = min(x1 + tile_pad, w)

                tile = img[:, :, py0:py1, px0:px1]
                up = model(tile)

                # 在放大后坐标系里，把 pad 部分裁掉，只保留 tile 实际范围
                cut_top = (y0 - py0) * scale
                cut_left = (x0 - px0) * scale
                cut_bot = cut_top + (y1 - y0) * scale
                cut_right = cut_left + (x1 - x0) * scale
                core = up[:, :, cut_top:cut_bot, cut_left:cut_right]

                output[
                    :,
                    :,
                    y0 * scale : y1 * scale,
                    x0 * scale : x1 * scale,
                ] = core

    return output


# ---------------------------------------------------------------------------
# 文件级 API
# ---------------------------------------------------------------------------


def upscale_file(
    src: Path,
    dst: Path,
    *,
    model_path: Path,
    label: str = "4x-AnimeSharp",
    tile_size: int = 256,
    tile_pad: int = 16,
    device: str = "auto",
    precision: str = "auto",
    target_area: Optional[int] = None,
    on_log: Callable[[str], None] = lambda _l: None,
    write_sidecar: bool = True,
    prewarm_thumb_sizes: Optional[list[int]] = None,
) -> dict[str, Any]:
    """读 src → 智能放大 → 写 dst（PNG）。返回元数据 dict。

    target_area 控制行为（LoRA 训练面向的"够用即可"）：
      - target_area=None：纯 4× 模型放大（兼容老路径）
      - target_area=N，src 面积 ≥ N×SKIP_MODEL_RATIO：跳过模型，直接 LANCZOS 缩到 ~N
      - target_area=N，src 面积 < N×SKIP_MODEL_RATIO：模型 4× → LANCZOS 缩到 ~N

    跳过模型那一路是"大图直接缩"，几百毫秒；走模型那一路才是几十秒的开销。
    大部分训练集图片像素都够 1024²/1536²，预处理实际只对少数小图调模型。

    元数据 sidecar（写到 `{dst}.preprocess.json` 当 write_sidecar=True）：
        {source, model, scale, action, target_area, tile_size, tile_pad,
         device, dtype, src_size, dst_size, elapsed_seconds, mtime}
    action: 'resize' | 'upscale' | 'upscale+resize'

    `device='cuda'` 但 GPU 不可用时静默降级 cpu。
    """
    t_start = time.monotonic()

    # 1) 先读图判断走哪条路（避免无谓的模型加载）
    with Image.open(src) as raw:
        raw.load()
        if raw.mode != "RGB":
            raw = raw.convert("RGB")
        src_img = raw.copy()  # 离开 with 块后还要用
    src_size = src_img.size  # (W, H)
    src_area = src_img.width * src_img.height
    skip_model = (
        target_area is not None
        and src_area >= int(target_area * SKIP_MODEL_RATIO)
    )

    if skip_model:
        # 已够大 — 直接 LANCZOS 缩到目标面积，绕过模型
        out_img = resize_to_area(src_img, int(target_area))  # type: ignore[arg-type]
        action = "resize"
        scale = 1  # 这条路径没经过模型，记 1 表示未放大
        dev = resolve_device(device)
        dtype = resolve_dtype(precision, dev)
    else:
        # 走模型放大
        dev = resolve_device(device)
        dtype = resolve_dtype(precision, dev)
        descriptor = load_model(model_path, device=dev, dtype=dtype)
        scale = int(descriptor.scale)
        tensor = _img_to_tensor(src_img).to(dev, dtype=dtype)
        out_tensor = tiled_inference(
            descriptor.model,
            tensor,
            scale=scale,
            tile_size=tile_size,
            tile_pad=tile_pad,
        )
        out_img = _tensor_to_img(out_tensor)
        if target_area is not None:
            out_img = resize_to_area(out_img, int(target_area))
            action = "upscale+resize"
        else:
            action = "upscale"

    dst.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(dst, format="PNG", optimize=False)

    # 趁内存里还有 PIL Image，把缩略图预生成进缓存。
    # 不预热的话用户首次浏览 grid 时会逐张解码 PNG（一张 1-3s，200 张要等几分钟），
    # 而 worker 这里已经付过解码代价了，多花零点几秒生成 thumb 摊到批处理里几乎无感。
    if prewarm_thumb_sizes:
        try:
            from .. import thumb_cache
            thumb_cache.prewarm_from_image(dst, out_img, prewarm_thumb_sizes)
        except Exception as exc:  # noqa: BLE001 — 缩略图预热失败不影响放大本体
            on_log(f"   ⚠ thumb prewarm failed: {exc}")

    elapsed = time.monotonic() - t_start

    meta = {
        "source": src.name,
        "model": label,
        "action": action,
        "scale": scale,
        "target_area": target_area,
        "tile_size": tile_size,
        "tile_pad": tile_pad,
        "device": str(dev),
        "dtype": str(dtype).replace("torch.", ""),
        "src_size": list(src_size),
        "dst_size": list(out_img.size),
        "elapsed_seconds": round(elapsed, 3),
        "mtime": time.time(),
    }
    if write_sidecar:
        sidecar = dst.with_suffix(dst.suffix + ".preprocess.json")
        sidecar.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    on_log(
        f"   ✓ [{action}] {src.name} → {dst.name}  "
        f"{src_size[0]}×{src_size[1]} → {out_img.size[0]}×{out_img.size[1]}  "
        f"({elapsed:.1f}s)"
    )
    return meta
