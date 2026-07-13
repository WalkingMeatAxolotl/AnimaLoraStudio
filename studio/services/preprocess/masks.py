"""训练 mask sidecar（`train/masks/{folder}/{stem}.png`）读写 + 预处理变换跟随。

设计：docs/design/preprocess-inpaint-mask-design.md §2 / §7 / §9。

- 路径镜像 manifest entry key 且恒 `.png`：`1_data/X.jpg` 的 mask 是
  `masks/1_data/X.png`。stem 不含扩展名 —— crop / 涂抹把 X.jpg 产物统一成
  X.png 时 mask 路径不变，天然免疫产物改名。
- 灰度 L 语义：255=正常学习、0=不学、中间值=部分权重（训练器 /255 作 loss
  权重）。无 mask 文件 = 全 255；「清除 mask」= 删文件。
- `masks/` 位于 train/ 之下但**不是** concept folder：一级扫描点（studio
  `_train_images_listing` / bundle `_collect_train`）只看子目录直下文件，
  masks/ 直下只有目录所以天然不可见；**递归**扫描点（trainer
  `ImageDataset._scan` / `compute_bucket_histogram` /
  `preview_train_tag_distribution`）必须用 `TRAIN_RESERVED_DIRS` 排除，
  否则 mask 灰度图会被当训练图吞掉。
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from studio.domain.errors import ValidationError

MASKS_DIRNAME = "masks"


def mask_path_for(train_dir: Path, rel_name: str) -> Path:
    """`1_data/X.jpg` → `{train_dir}/masks/1_data/X.png`。

    写入端点前置 `_validate_rel_name`（严格两段）；删除 / 查询路径还会被
    manifest mutation 以**老式平铺 name**（无 folder 前缀，ADR 0004 兼容
    数据）调到 —— 平铺 name 映射到 `masks/{stem}.png`，文件不存在时上层
    no-op，不 crash。
    """
    if "/" in rel_name:
        folder, filename = rel_name.split("/", 1)
        return train_dir / MASKS_DIRNAME / folder / f"{Path(filename).stem}.png"
    return train_dir / MASKS_DIRNAME / f"{Path(rel_name).stem}.png"


def write_mask(
    train_dir: Path,
    rel_name: str,
    data: bytes,
    *,
    expected_size: tuple[int, int],
) -> dict[str, Any]:
    """写入 mask（灰度 PNG，tmp + atomic replace）。

    尺寸必须等于对应训练图当前尺寸 —— mask 是逐像素对齐的数据面，
    不符说明前端导出对象错位，直接拒绝。
    """
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # PIL 解码失败抛的类型不稳定，统一翻 400
        raise ValidationError(
            "Uploaded mask is not a valid image file",
            code="preprocess.mask_image_invalid",
            details={"name": rel_name}, http_status=400,
        ) from exc
    if img.size != tuple(expected_size):
        raise ValidationError(
            "Mask size does not match the source image",
            code="preprocess.mask_size_mismatch",
            details={
                "name": rel_name,
                "expected": [expected_size[0], expected_size[1]],
                "got": [img.size[0], img.size[1]],
            },
            http_status=400,
        )
    if img.mode != "L":
        img = img.convert("L")

    out = mask_path_for(train_dir, rel_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    img.save(tmp, format="PNG", optimize=False)
    os.replace(tmp, out)
    st = out.stat()
    return {"name": rel_name, "mtime": st.st_mtime, "size": st.st_size}


def delete_mask(train_dir: Path, rel_name: str) -> bool:
    """删除 mask 文件。返回是否真的删了（不存在返回 False，不报错）。"""
    p = mask_path_for(train_dir, rel_name)
    if not p.is_file():
        return False
    try:
        p.unlink()
    except OSError:
        return False
    return True


def delete_masks_for(train_dir: Path, rel_names: Iterable[str]) -> int:
    """批量删（restore / 去重 / 移除训练图的 sidecar 跟随清理）。"""
    n = 0
    for rel in rel_names:
        if delete_mask(train_dir, rel):
            n += 1
    return n


def crop_mask_like(
    train_dir: Path,
    src_rel: str,
    boxes: list[tuple[int, int, int, int]],
    out_rels: list[str],
) -> None:
    """crop 跟随：用与图片相同的像素 box 裁 mask，fan-out 到 out_rels 的
    mask 路径；源 mask 不在输出集合时删除。

    源图无 mask → no-op。mask 尺寸与源图不符（外部改图漏网）→ 删源 mask
    （几何已错位，保留只会污染训练；训练器尺寸校验是最后防线，这里主动清）。
    """
    from PIL import Image

    src_mask = mask_path_for(train_dir, src_rel)
    if not src_mask.is_file():
        return
    with Image.open(src_mask) as raw:
        raw.load()
        mask_img = raw.convert("L") if raw.mode != "L" else raw.copy()

    out_paths: list[Path] = []
    for box, out_rel in zip(boxes, out_rels):
        piece = mask_img.crop(box)
        out = mask_path_for(train_dir, out_rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        piece.save(tmp, format="PNG", optimize=False)
        os.replace(tmp, out)
        out_paths.append(out)

    if src_mask not in out_paths:
        try:
            src_mask.unlink()
        except OSError:
            pass


def resize_mask_like(
    train_dir: Path, rel_name: str, size: tuple[int, int],
) -> None:
    """upscale 跟随：mask NEAREST resize 到新尺寸（不走 RealESRGAN，
    防灰度值被超分模型污染）。无 mask → no-op。"""
    from PIL import Image

    p = mask_path_for(train_dir, rel_name)
    if not p.is_file():
        return
    with Image.open(p) as raw:
        raw.load()
        img = raw.convert("L") if raw.mode != "L" else raw.copy()
    if img.size == tuple(size):
        return
    img = img.resize(size, Image.NEAREST)
    tmp = p.with_suffix(p.suffix + ".tmp")
    img.save(tmp, format="PNG", optimize=False)
    os.replace(tmp, p)


def mask_stat(train_dir: Path, rel_name: str) -> Optional[dict[str, Any]]:
    """mask 存在时返回 {mtime, size}，否则 None（workspace 列表 has_mask 用）。"""
    p = mask_path_for(train_dir, rel_name)
    try:
        st = p.stat()
    except OSError:
        return None
    return {"mtime": st.st_mtime, "size": st.st_size}
