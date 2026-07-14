"""训练 mask sidecar（`train/{folder}/{stem}.mask`）读写 + 预处理变换跟随。

设计：docs/design/preprocess-inpaint-mask-design.md §2 / §7 / §9。

- mask 与训练图**同目录同 stem**，后缀恒 `.mask`（内容是灰度 PNG 字节）。
  与 .txt / .json caption sidecar 同构：后缀不在 IMAGE_EXTS，所有图片扫描点
  （一级 / 递归）天然不会把它当训练图 —— 不需要任何豁免规则。
- stem 不含扩展名 —— crop / 涂抹把 X.jpg 产物统一成 X.png 时 mask 路径不变，
  天然免疫产物改名；同 stem sidecar 家族（.txt/.json/.mask）删除 / 导出 /
  变换跟随共用一套心智模型。
- 灰度 L 语义：255=正常学习、0=不学、中间值=部分权重（训练器 /255 作 loss
  权重）。无 mask 文件 = 全 255；「清除 mask」= 删文件。
- **Legacy 布局迁移**：首发版（2026-07 PR #394）mask 落在保留目录
  `train/masks/{folder}/{stem}.png`。本模块所有公开入口先调
  `migrate_legacy_masks`（幂等；masks/ 不存在时一次 exists 即短路）把老文件
  搬到新位置。`TRAIN_RESERVED_DIRS` / trainer `RESERVED_SUBDIRS` 的 masks/
  豁免**保留**——防止尚未触发迁移的老数据被递归扫描点当训练图。
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from studio.domain.errors import ValidationError

MASK_SUFFIX = ".mask"
# 老布局的保留目录名（只用于迁移 + 各扫描点的 legacy 豁免，不再写入）。
LEGACY_MASKS_DIRNAME = "masks"


def mask_path_for(train_dir: Path, rel_name: str) -> Path:
    """`1_data/X.jpg` → `{train_dir}/1_data/X.mask`。

    写入端点前置 `_validate_rel_name`（严格两段）；删除 / 查询路径还会被
    manifest mutation 以**老式平铺 name**（无 folder 前缀，ADR 0004 兼容
    数据）调到 —— 平铺 name 映射到 `{train_dir}/{stem}.mask`，文件不存在时
    上层 no-op，不 crash。
    """
    if "/" in rel_name:
        folder, filename = rel_name.split("/", 1)
        return train_dir / folder / f"{Path(filename).stem}{MASK_SUFFIX}"
    return train_dir / f"{Path(rel_name).stem}{MASK_SUFFIX}"


def migrate_legacy_masks(train_dir: Path) -> int:
    """老布局 `train/masks/**.png` → 同目录 `.mask` sidecar。返回搬迁数。

    幂等：legacy 目录不存在时一次 exists 检查即返回。逐文件 `os.replace`：
    - 新位置已存在（迁移后又在新位置写过）→ 新的是权威，直接删老文件。
    - 目标 concept 目录已不存在（folder 被删 / 改名后的孤儿 mask）→ 跳过
      不搬也不删（留在 legacy 目录里无害——各扫描点仍豁免 masks/）。
    搬空的子目录顺手 rmdir（失败忽略）。
    """
    legacy_root = train_dir / LEGACY_MASKS_DIRNAME
    if not legacy_root.is_dir():
        return 0

    moved = 0

    def _migrate_file(src: Path, target: Path) -> None:
        nonlocal moved
        try:
            if target.exists():
                src.unlink()
            elif target.parent.is_dir():
                os.replace(src, target)
                moved += 1
        except OSError:
            pass

    for child in sorted(legacy_root.iterdir()):
        if child.is_file() and child.suffix.lower() == ".png":
            # 平铺老 mask（masks/{stem}.png，ADR 0004 兼容数据）→ train 根
            _migrate_file(child, train_dir / f"{child.stem}{MASK_SUFFIX}")
        elif child.is_dir():
            for f in sorted(child.iterdir()):
                if f.is_file() and f.suffix.lower() == ".png":
                    _migrate_file(
                        f, train_dir / child.name / f"{f.stem}{MASK_SUFFIX}",
                    )
            try:
                child.rmdir()
            except OSError:
                pass
    try:
        legacy_root.rmdir()
    except OSError:
        pass
    return moved


def write_mask(
    train_dir: Path,
    rel_name: str,
    data: bytes,
    *,
    expected_size: tuple[int, int],
) -> dict[str, Any]:
    """写入 mask（灰度 PNG 字节，tmp + atomic replace）。

    尺寸必须等于对应训练图当前尺寸 —— mask 是逐像素对齐的数据面，
    不符说明前端导出对象错位，直接拒绝。
    """
    from PIL import Image

    migrate_legacy_masks(train_dir)
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
    migrate_legacy_masks(train_dir)
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

    migrate_legacy_masks(train_dir)
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

    migrate_legacy_masks(train_dir)
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


def mask_file(train_dir: Path, rel_name: str) -> Optional[Path]:
    """mask 文件路径（不存在返回 None）。GET 端点用。"""
    migrate_legacy_masks(train_dir)
    p = mask_path_for(train_dir, rel_name)
    return p if p.is_file() else None


def mask_stat(train_dir: Path, rel_name: str) -> Optional[dict[str, Any]]:
    """mask 存在时返回 {mtime, size}，否则 None（workspace 列表 has_mask 用）。"""
    migrate_legacy_masks(train_dir)
    p = mask_path_for(train_dir, rel_name)
    try:
        st = p.stat()
    except OSError:
        return None
    return {"mtime": st.st_mtime, "size": st.st_size}
