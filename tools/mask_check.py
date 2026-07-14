"""训练 mask 数据链路快速检验（B1 数据面 → B2 变换数学 dry-run）。

把涂抹页保存的 mask（train/{folder}/{stem}.mask 同目录 sidecar，内容灰度
PNG）走一遍训练器将要执行的几何管线，验证「遮罩 → 训练」链路的数据面是通的：

  1. mask 存在性 + 尺寸校验（训练器 fail-safe 语义：不匹配 = 该图按无 mask）
  2. bucket 分桶 + resize-cover + center-crop（镜像 ImageDataset.get_with_flip
     的几何计算；mask 用 NEAREST 防灰度插值污染）
  3. area 下采样到 latent /8 分辨率（PIL BOX = 块均值，与 B2 决策一致）
  4. 输出每图 latent 权重图形状 + 均值 / 覆盖率统计

注意：训练器 masked loss（B2）尚未实现，本脚本验证的是数据链路与变换数学；
loss 层面的有效性在 B2 落地后跑 A/B（设计文档 §9 决策 6）。

用法:
  ./venv/Scripts/python.exe tools/mask_check.py <train_dir> [--reso 1024] [--save-vis DIR]

  train_dir  = projects/{id}-{slug}/versions/{label}/train
  --reso     bucket base 分辨率（默认 1024，对应训练 config resolution）
  --save-vis 保存人眼检查图到目录：{stem}.overlay.png（原图+红色 mask 叠加）
             / {stem}.latent.png（latent 权重图 NEAREST 放大回 bucket 尺寸）

退出码：0 = 全部通过；1 = 存在尺寸不匹配 / 读取失败（会被训练器降级为无 mask）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台 cp932 下 CJK 输出会崩，强制 utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from runtime.training.dataset import BucketManager  # noqa: E402

VAE_DOWNSAMPLE = 8
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MASK_SUFFIX = ".mask"


def mask_path_for(train_dir: Path, folder: str, filename: str) -> Path:
    return train_dir / folder / f"{Path(filename).stem}{MASK_SUFFIX}"


def bucket_geometry(
    w: int, h: int, mgr: BucketManager,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """镜像 ImageDataset.get_with_flip：bucket → resize-cover → center-crop。

    返回 (bucket_size, resize_size, crop_lt)。
    """
    tw, th = mgr.get_bucket(w, h)
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return (tw, th), (nw, nh), (left, top)


def transform_mask(
    mask: Image.Image, resize_size: tuple[int, int],
    crop_lt: tuple[int, int], bucket: tuple[int, int],
) -> Image.Image:
    """mask 走与图片相同的几何变换（NEAREST）。"""
    nw, nh = resize_size
    left, top = crop_lt
    tw, th = bucket
    m = mask.resize((nw, nh), Image.NEAREST)
    return m.crop((left, top, left + tw, top + th))


def downsample_to_latent(mask: Image.Image) -> Image.Image:
    """area（BOX = 块均值）下采样到 /8 latent 分辨率（B2 §9 决策 4）。"""
    lw = max(1, mask.width // VAE_DOWNSAMPLE)
    lh = max(1, mask.height // VAE_DOWNSAMPLE)
    return mask.resize((lw, lh), Image.BOX)


def mean_gray(img: Image.Image) -> float:
    hist = img.histogram()
    total = sum(hist)
    if total == 0:
        return 0.0
    return sum(v * n for v, n in enumerate(hist)) / total / 255.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("train_dir", type=Path)
    ap.add_argument("--reso", type=int, default=1024)
    ap.add_argument("--save-vis", type=Path, default=None)
    args = ap.parse_args()

    train_dir: Path = args.train_dir
    if not train_dir.is_dir():
        print(f"[error] train_dir 不存在: {train_dir}")
        return 1

    mgr = BucketManager(args.reso)
    vis_dir: Path | None = args.save_vis
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_masked = 0
    bad: list[str] = []

    for sub in sorted(train_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            n_images += 1
            rel = f"{sub.name}/{f.name}"
            try:
                with Image.open(f) as im:
                    w, h = im.size
            except (OSError, ValueError) as exc:
                bad.append(f"{rel}: 图片不可读 ({exc})")
                continue

            mp = mask_path_for(train_dir, sub.name, f.name)
            if not mp.is_file():
                print(f"  {rel:<40} {w}x{h}  mask=无（全图正常学习）")
                continue

            try:
                with Image.open(mp) as raw:
                    raw.load()
                    mask = raw.convert("L") if raw.mode != "L" else raw.copy()
            except (OSError, ValueError) as exc:
                bad.append(f"{rel}: mask 不可读 ({exc})")
                continue

            if mask.size != (w, h):
                bad.append(
                    f"{rel}: mask 尺寸 {mask.size[0]}x{mask.size[1]} != 图片 {w}x{h}"
                    "（训练器将降级为无 mask）"
                )
                continue

            n_masked += 1
            coverage = 1.0 - mean_gray(mask)
            bucket, resize_size, crop_lt = bucket_geometry(w, h, mgr)
            m_bucket = transform_mask(mask, resize_size, crop_lt, bucket)
            m_latent = downsample_to_latent(m_bucket)
            latent_weight = mean_gray(m_latent)
            print(
                f"  {rel:<40} {w}x{h}  mask=有  覆盖 {coverage * 100:5.1f}%  "
                f"bucket {bucket[0]}x{bucket[1]} → latent {m_latent.width}x{m_latent.height}  "
                f"latent 权重均值 {latent_weight:.3f}"
            )

            if vis_dir:
                stem = Path(f.name).stem
                with Image.open(f) as im:
                    base = im.convert("RGB")
                red = Image.new("RGB", base.size, (255, 45, 45))
                # alpha = 不学度（255 - 灰度）再乘显示强度
                alpha = mask.point(lambda v: int((255 - v) * 0.45))
                overlay = base.copy()
                overlay.paste(red, (0, 0), alpha)
                overlay.save(vis_dir / f"{stem}.overlay.png")
                m_latent.resize(bucket, Image.NEAREST).save(
                    vis_dir / f"{stem}.latent.png"
                )

    print()
    print(f"[summary] 图片 {n_images} 张，其中 {n_masked} 张有有效 mask")
    if bad:
        print(f"[warn] {len(bad)} 项异常（训练器会按无 mask 降级，不会 crash）：")
        for line in bad:
            print(f"  ⚠ {line}")
        return 1
    print("[ok] mask 数据链路全部通过：尺寸对齐 → bucket 变换 → latent /8 area 下采样")
    return 0


if __name__ == "__main__":
    sys.exit(main())
