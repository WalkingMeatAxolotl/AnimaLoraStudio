"""图片缩略图缓存（PP3 polish）。

为什么要单独做：之前 `/api/.../thumb` 直接 serve 原图，前端只是用 CSS 缩。
当 download/ 有几百张几 MB 的 PNG 时，浏览器持续 decode 大图，滚动 / 悬停
切预览都卡。这里把缩略图先生成到 `studio_data/thumb_cache/{sha1}.jpg`
（hash = src 路径 + mtime + size），后续直接返回缓存。

设计：
- 缓存键含源文件 mtime，源被替换会自动 invalidate（hash 变）
- 多线程安全：先写 .tmp 再 rename
- size=0 表示「不缩」，直接返回源路径
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from ...paths import THUMB_CACHE_DIR

logger = logging.getLogger(__name__)

# 进程内锁：避免两个并发请求同时生成同一缩略图（写半截）。
_LOCKS_LOCK = threading.Lock()
_KEY_LOCKS: dict[str, threading.Lock] = {}

# Pillow 9.1+ 把 LANCZOS 挪到 Image.Resampling 下；旧版本仍可用 Image.LANCZOS。
_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]


def _key_lock(key: str) -> threading.Lock:
    with _LOCKS_LOCK:
        lk = _KEY_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _KEY_LOCKS[key] = lk
        return lk


def _key_for(src: Path, size: int) -> Optional[str]:
    """缓存键：sha1(abs_path|mtime_ns|size)。

    stat 失败时返回 None —— 不再退化用 mtime=0，否则 Windows 偶发 stat 失败
    （杀软扫描 / 文件锁）会让所有受影响图片共享同一个 cache 键，串图。
    上层拿 None 应该跳过缓存直接生成临时缩略图（或退回原图）。
    """
    try:
        mtime = src.stat().st_mtime_ns
    except OSError as exc:
        logger.warning("thumb cache: stat failed for %s: %s", src, exc)
        return None
    payload = f"{src.resolve()}|{mtime}|{size}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def get_or_make_thumb(src: Path, size: int) -> Path:
    """返回可直接 FileResponse 的缩略图路径。

    size <= 0 → 原图直出（不缩）。生成失败 → 回退到原图。
    """
    if size <= 0:
        return src
    if not src.exists():
        return src
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _key_for(src, size)
    if key is None:
        # stat 失败：不缓存，不串图；直接返回原图（Cache-Control: no-cache 让浏览器
        # 下次会重试，stat 恢复后能拿到正常缩略图）
        return src
    out = THUMB_CACHE_DIR / f"{key}.jpg"
    if out.exists():
        return out

    lock = _key_lock(key)
    with lock:
        if out.exists():
            return out
        tmp = out.with_suffix(out.suffix + ".tmp")
        try:
            # 必须在文件 still-open 期间完成所有像素操作：
            # ImageOps.exif_transpose 对没有 orientation 的图直接返回原 lazy
            # image，而 Image.open 是 lazy 的；一旦 with 块退出文件句柄被关，
            # 后续 thumbnail/save 触发 lazy load 会失败，进而被 except 吞掉
            # 返回源图（几 MB），让前端依旧加载大图、滚动卡顿。
            with Image.open(src) as raw:
                img = ImageOps.exif_transpose(raw) or raw
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((size, size), _RESAMPLE)
                img.save(tmp, "JPEG", quality=80, optimize=True)
            os.replace(tmp, out)
        except Exception as exc:
            logger.warning(
                "thumb generation failed for %s (size=%d): %s", src, size, exc
            )
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return src
    return out


def prewarm_from_image(
    src: Path, image: Image.Image, sizes: list[int]
) -> list[Path]:
    """用已在内存的 PIL Image 直接写多档缩略图到缓存，省掉首次浏览时的解码。

    主要给 upscaler 用：放大后的 PIL Image 还在内存里，与其等用户首次访问时
    再读 PNG + decode + resize（一张几 MB 的 4× PNG 在 CPU 上 1-3s），不如
    在 worker 阶段一次性把 256 / 768 都生成好。

    `src` 决定缓存键 —— 必须是放大产物文件的真实路径（同 get_or_make_thumb
    的 hash 计算）。`image` 应该是 RGB；其它模式会自动转换。
    """
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    base = image
    if base.mode != "RGB":
        base = base.convert("RGB")
    for size in sizes:
        if size <= 0:
            continue
        key = _key_for(src, size)
        if key is None:
            continue
        out = THUMB_CACHE_DIR / f"{key}.jpg"
        if out.exists():
            written.append(out)
            continue
        lock = _key_lock(key)
        with lock:
            if out.exists():
                written.append(out)
                continue
            tmp = out.with_suffix(out.suffix + ".tmp")
            try:
                thumb = base.copy()
                thumb.thumbnail((size, size), _RESAMPLE)
                thumb.save(tmp, "JPEG", quality=80, optimize=True)
                os.replace(tmp, out)
                written.append(out)
            except Exception as exc:
                logger.warning(
                    "thumb prewarm failed for %s (size=%d): %s", src, size, exc
                )
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    return written


def clear_cache() -> int:
    """删除缓存目录下所有 .jpg；返回删除数量。"""
    if not THUMB_CACHE_DIR.exists():
        return 0
    n = 0
    for p in THUMB_CACHE_DIR.glob("*.jpg"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n
