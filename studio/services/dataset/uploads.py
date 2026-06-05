"""本地上传：单图或 zip 压缩包（自动解压拍平），落盘到 download/。

用户通过浏览器 file picker / 拖拽上传文件 → server 端解析 → 写入项目的
`download/` 目录，与 booru 下载共享同一份「全量备份」。

约束：
- 接受 IMAGE_EXTS 里所有格式的单图（png / jpg / jpeg / webp / bmp / gif）；
  zip 包内同样按这套白名单提取。
- zip 内的子目录结构会被拍平（取 basename）。
- zip 损坏时整包跳过并报告，不影响其它文件。

`convert_to_png` 模式（与 booru 下载共用的 `gelbooru.convert_to_png` 设置）：
- 所有图片经 PIL 解码后统一重编码为 .png，文件名 stem 不变后缀改 .png；
- 同 stem 冲突（含 `1.jpg` + `1.png` 同上传一次的场景）改加 `_1`/`_2` 后缀
  落盘，避免 caption `1.txt` 被两张不同图共用；
- `remove_alpha_channel=True` 时按白底压平 alpha，与 booru 下载一致；
- PIL 解码失败按 skipped 上报「图片损坏」。

`convert_to_png=False`（默认）保持历史行为：原扩展名拷贝、目标已存在则跳过。
"""
from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable

from PIL import Image

from ..booru.api import flatten_alpha, has_alpha
from .scan import IMAGE_EXTS

# PP10 起复用全链路白名单：上传 / 下载 / curation / 训练共用一份。
ALLOWED_IMAGE_EXTS = IMAGE_EXTS
ZIP_EXT = ".zip"


@dataclass
class UploadResult:
    """单次上传调用的汇总结果。"""

    added: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, list]:
        return {"added": self.added, "skipped": self.skipped}

    def merge(self, other: "UploadResult") -> None:
        self.added.extend(other.added)
        self.skipped.extend(other.skipped)


def _safe_basename(name: str) -> str:
    """剥掉 zip 内嵌套子目录 / Windows 反斜杠，只留 basename。"""
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def _is_image_ext(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_IMAGE_EXTS


def _unique_target(dest_dir: Path, name: str) -> Path:
    """目标已存在时加 `_1` / `_2` ... 后缀直到不冲突。

    用于 convert_to_png 模式：用户的本意是「这张图也进来」，文件名冲突时
    用后缀保住第二张，而不是丢弃。caption 文件按落盘后的实际 stem 配对，
    所以后缀化的 `1_1.png` 会拿到独立的 `1_1.txt`，不再与 `1.png` 共用。
    """
    target = dest_dir / name
    if not target.exists():
        return target
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        cand = dest_dir / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def _write_image_entry(
    src_name: str,
    src_stream: BinaryIO,
    dest_dir: Path,
    *,
    convert_to_png: bool,
    remove_alpha_channel: bool,
    report_name: str,
    result: UploadResult,
) -> None:
    """落盘单张图。"""
    if not convert_to_png:
        target = dest_dir / src_name
        if target.exists():
            result.skipped.append(
                {"name": report_name, "reason": "已存在，跳过"}
            )
            return
        with target.open("wb") as fh:
            shutil.copyfileobj(src_stream, fh)
        result.added.append(src_name)
        return

    raw = src_stream.read()
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001 — PIL 抛多种类型，整体当损坏
        result.skipped.append(
            {"name": report_name, "reason": f"图片损坏: {exc}"}
        )
        return
    target = _unique_target(dest_dir, Path(src_name).stem + ".png")
    if remove_alpha_channel and has_alpha(img):
        img = flatten_alpha(img)
    out = (
        img.convert("RGBA")
        if has_alpha(img) and not remove_alpha_channel
        else img.convert("RGB")
    )
    out.save(target, "PNG", optimize=True)
    result.added.append(target.name)


def accept_one(
    src_name: str,
    src_stream: BinaryIO,
    dest_dir: Path,
    *,
    convert_to_png: bool = False,
    remove_alpha_channel: bool = False,
) -> UploadResult:
    """处理单个上传文件。

    - IMAGE_EXTS 内任一格式 → 落盘（按 convert_to_png 决定是否重编码 PNG）
    - zip → 解压所有图片格式（拍平、内层 entry 同样按 convert_to_png 处理）
    - 其他 / 没扩展名 → 拒绝
    """
    base = _safe_basename(src_name or "")
    suffix = Path(base).suffix.lower()
    result = UploadResult()
    if not base:
        result.skipped.append({"name": src_name, "reason": "文件名为空"})
        return result

    dest_dir.mkdir(parents=True, exist_ok=True)

    if suffix in ALLOWED_IMAGE_EXTS:
        _write_image_entry(
            base, src_stream, dest_dir,
            convert_to_png=convert_to_png,
            remove_alpha_channel=remove_alpha_channel,
            report_name=base,
            result=result,
        )
        return result

    if suffix == ZIP_EXT:
        try:
            with zipfile.ZipFile(src_stream) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner_base = _safe_basename(info.filename)
                    label = f"{base}:{info.filename}"
                    if not inner_base:
                        continue
                    if not _is_image_ext(inner_base):
                        result.skipped.append(
                            {"name": label, "reason": "格式不支持"}
                        )
                        continue
                    with zf.open(info) as entry:
                        _write_image_entry(
                            inner_base, entry, dest_dir,
                            convert_to_png=convert_to_png,
                            remove_alpha_channel=remove_alpha_channel,
                            report_name=label,
                            result=result,
                        )
        except zipfile.BadZipFile:
            result.skipped.append({"name": base, "reason": "zip 损坏"})
        return result

    allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTS)) + ", .zip"
    result.skipped.append(
        {"name": base, "reason": f"格式不支持（仅 {allowed}）"}
    )
    return result


def accept_many(
    files: Iterable[tuple[str, BinaryIO]],
    dest_dir: Path,
    *,
    convert_to_png: bool = False,
    remove_alpha_channel: bool = False,
) -> UploadResult:
    """批量处理；汇总 added / skipped。"""
    out = UploadResult()
    for name, stream in files:
        out.merge(
            accept_one(
                name, stream, dest_dir,
                convert_to_png=convert_to_png,
                remove_alpha_channel=remove_alpha_channel,
            )
        )
    return out
