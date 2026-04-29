"""本地上传：单图（jpg/png）或 zip 压缩包（自动解压拍平），落盘到 download/。

用户通过浏览器 file picker / 拖拽上传文件 → server 端解析 → 写入项目的
`download/` 目录，与 booru 下载共享同一份「全量备份」。

约束：
- 仅接受 .jpg / .jpeg / .png 单图；zip 包内同样仅提取 jpg/png。
- zip 内的子目录结构会被拍平（取 basename）。
- 文件名冲突保守跳过，不覆盖、不自动重命名 —— 让用户知道哪里跳过了。
- zip 损坏时整包跳过并报告，不影响其它文件。
"""
from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
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


def accept_one(
    src_name: str, src_stream: BinaryIO, dest_dir: Path
) -> UploadResult:
    """处理单个上传文件。

    - jpg/jpeg/png → 直接拷到 dest_dir
    - zip → 解压所有 jpg/png（拍平）
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
        target = dest_dir / base
        if target.exists():
            result.skipped.append({"name": base, "reason": "已存在，跳过"})
        else:
            with target.open("wb") as fh:
                shutil.copyfileobj(src_stream, fh)
            result.added.append(base)
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
                    target = dest_dir / inner_base
                    if target.exists():
                        result.skipped.append(
                            {"name": label, "reason": "已存在，跳过"}
                        )
                        continue
                    with zf.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    result.added.append(inner_base)
        except zipfile.BadZipFile:
            result.skipped.append({"name": base, "reason": "zip 损坏"})
        return result

    result.skipped.append(
        {"name": base, "reason": "格式不支持（仅 .jpg / .png / .zip）"}
    )
    return result


def accept_many(
    files: Iterable[tuple[str, BinaryIO]], dest_dir: Path
) -> UploadResult:
    """批量处理；汇总 added / skipped。"""
    out = UploadResult()
    for name, stream in files:
        out.merge(accept_one(name, stream, dest_dir))
    return out
