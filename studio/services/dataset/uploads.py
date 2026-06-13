"""本地上传：单图或 zip 压缩包（自动解压拍平），落盘到 download/。

用户通过浏览器 file picker / 拖拽上传文件 → server 端解析 → 写入项目的
`download/` 目录，与 booru 下载共享同一份「全量备份」。

约束：
- 接受 IMAGE_EXTS 里所有格式的单图（png / jpg / jpeg / webp / bmp / gif）；
  zip 包内同样按这套白名单提取。
- zip 内的子目录结构会被拍平（取 basename）。
- zip 损坏时整包跳过并报告，不影响其它文件。

`convert_to_png` 模式（与 booru 下载共用的 `gelbooru.convert_to_png` 设置）：
- 所有非 PNG 图片经 PIL 解码后统一重编码为 .png；
- **PNG 直通路径**：源已经是 PNG 且不要求去 alpha（或源本就无 alpha）时
  **直接字节拷贝**，跳过 decode + re-encode，省 10× 时间。这是最常见的 booru
  场景。
- 同 stem 冲突（含 `1.jpg` + `1.png` 同上传一次的场景）改加 `_1`/`_2` 后缀
  落盘，避免 caption `1.txt` 被两张不同图共用；
- `remove_alpha_channel=True` 时按白底压平 alpha；
- PIL 解码失败按 skipped 上报「图片损坏」。

`convert_to_png=False`（默认）保持历史行为：原扩展名拷贝、目标已存在则跳过。

性能历史（hotfix #273）：
- `optimize=True` 删除：原 4K 图 PNG 重编码 1.5-2s/张 → 200-300ms/张（5-10×）
- PNG 直通：已是 PNG 跳过整段 decode+encode，~20ms/张（vs 慢路径 1s+）
- 上层路由改为流式（不再 `await f.read()` 整包进 RAM），1GB zip 内存峰值从
  1GB 降到 ~1MB（SpooledTemporaryFile 默认 spool 阈值）

阶段日志：`accept_one` 处理 zip 时通过 `on_log` 回调推每 25 张 / 5s 一行进度
（仿 `services/booru/downloader.py` 的 on_progress 模式），方便用户看到进度
而不是"卡在 100%"。
"""
from __future__ import annotations

import logging
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Optional

from PIL import Image

from ..booru.api import flatten_alpha, has_alpha
from .scan import IMAGE_EXTS

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]

# PP10 起复用全链路白名单：上传 / 下载 / curation / 训练共用一份。
ALLOWED_IMAGE_EXTS = IMAGE_EXTS
ZIP_EXT = ".zip"

# 阶段日志节流参数（zip 内逐图处理时用）
_LOG_EVERY_N_IMAGES = 25
_LOG_EVERY_SECONDS = 5.0
_LOG_SLOW_IMAGE_THRESHOLD = 1.0  # 单张 > 1s 立刻 emit 一行（提示哪张拖后腿）


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


def _png_needs_reencode(raw: bytes, *, remove_alpha_channel: bool) -> Optional[bool]:
    """已是 PNG 源时判断要不要走慢路径（decode + re-encode）。

    返回：
      - False → 直接字节拷贝就行（最常见，省 10× 时间）
      - True  → 需要 flatten alpha，走慢路径
      - None  → PIL 解析失败，让慢路径接管报"图片损坏"

    判定逻辑：
      - `remove_alpha_channel=False` → 永远不需要重编码现成 PNG
      - `remove_alpha_channel=True`  → 只有源真有 alpha 才需要重编码
    """
    if not remove_alpha_channel:
        return False  # 用户不要求去 alpha，现成 PNG 直接字节拷贝
    try:
        with Image.open(BytesIO(raw)) as probe:
            # Image.open() 只读 IHDR 几十字节，**不**触发全图解码（lazy load）
            return has_alpha(probe)
    except Exception:
        return None  # 让慢路径走 .load() 抛同样的 exception → 统一 skipped 报告


def _write_image_entry(
    src_name: str,
    src_stream: BinaryIO,
    dest_dir: Path,
    *,
    convert_to_png: bool,
    remove_alpha_channel: bool,
    report_name: str,
    result: UploadResult,
) -> float:
    """落盘单张图。返回耗时秒数（给上层节流日志判定"慢图"用）。"""
    t0 = time.monotonic()
    if not convert_to_png:
        target = dest_dir / src_name
        if target.exists():
            result.skipped.append(
                {"name": report_name, "reason": "已存在，跳过"}
            )
            return time.monotonic() - t0
        with target.open("wb") as fh:
            shutil.copyfileobj(src_stream, fh)
        result.added.append(src_name)
        return time.monotonic() - t0

    raw = src_stream.read()

    # PNG 直通路径：源已是 PNG 且不需要 alpha 处理 → 直接字节拷贝。
    # booru 下来的图绝大多数都走这一支，省掉整段 decode + re-encode。
    src_ext = Path(src_name).suffix.lower()
    if src_ext == ".png":
        verdict = _png_needs_reencode(raw, remove_alpha_channel=remove_alpha_channel)
        if verdict is False:
            target = _unique_target(dest_dir, src_name)
            target.write_bytes(raw)
            result.added.append(target.name)
            return time.monotonic() - t0
        # verdict is True or None：fall through 到慢路径

    # 慢路径：真的需要重编码（JPG/WebP/BMP/GIF → PNG，或 PNG 需要 flatten alpha）
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001 — PIL 抛多种类型，整体当损坏
        result.skipped.append(
            {"name": report_name, "reason": f"图片损坏: {exc}"}
        )
        return time.monotonic() - t0

    target = _unique_target(dest_dir, Path(src_name).stem + ".png")

    # has_alpha 算一次，下面 convert 模式判定 + flatten 复用
    img_has_alpha = has_alpha(img)
    if remove_alpha_channel and img_has_alpha:
        img = flatten_alpha(img)
        target_mode = "RGB"
    elif img_has_alpha:
        target_mode = "RGBA"
    else:
        target_mode = "RGB"

    # 只在 mode 真的需要切换时才 convert —— 同 mode convert 仍会创建副本（24MB
    # for 4K image），不便宜
    if img.mode != target_mode:
        img = img.convert(target_mode)

    # 不带 optimize=True：原版每条扫描线试 5 种 filter + zlib level 9，4K 图
    # 1.5-2s/张；默认 compress_level=6 + 不试 filter ~200ms/张，体积差 5-15% 但
    # 对临时落盘可忽略（preprocess 还会再裁剪 / 放缩）
    img.save(target, "PNG")
    result.added.append(target.name)
    return time.monotonic() - t0


def accept_one(
    src_name: str,
    src_stream: BinaryIO,
    dest_dir: Path,
    *,
    convert_to_png: bool = False,
    remove_alpha_channel: bool = False,
    on_log: Optional[LogFn] = None,
) -> UploadResult:
    """处理单个上传文件。

    - IMAGE_EXTS 内任一格式 → 落盘（按 convert_to_png 决定是否重编码 PNG）
    - zip → 解压所有图片格式（拍平、内层 entry 同样按 convert_to_png 处理）
    - 其他 / 没扩展名 → 拒绝

    on_log：阶段日志回调，模仿 `services/booru/downloader.py` 的 on_progress
    模式。zip 解压时按 `[upload] <name>: N/total processed (Xs)` 节流推。
    """
    log = on_log or (lambda _s: None)
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
        t_zip_start = time.monotonic()
        try:
            with zipfile.ZipFile(src_stream) as zf:
                infos = [info for info in zf.infolist() if not info.is_dir()]
                image_count = sum(
                    1 for info in infos
                    if _is_image_ext(_safe_basename(info.filename))
                )
                log(
                    f"[upload] {base}: 打开 zip "
                    f"({len(infos)} entries, {image_count} 张图)"
                )
                processed = 0
                last_log_t = t_zip_start
                for info in infos:
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
                        per_image_secs = _write_image_entry(
                            inner_base, entry, dest_dir,
                            convert_to_png=convert_to_png,
                            remove_alpha_channel=remove_alpha_channel,
                            report_name=label,
                            result=result,
                        )
                    processed += 1
                    # 节流：每 N 张 / 每 K 秒 / 慢图 (>1s) 都触发一行
                    now = time.monotonic()
                    if (
                        processed % _LOG_EVERY_N_IMAGES == 0
                        or (now - last_log_t) >= _LOG_EVERY_SECONDS
                        or per_image_secs >= _LOG_SLOW_IMAGE_THRESHOLD
                    ):
                        elapsed = now - t_zip_start
                        log(
                            f"[upload] {base}: {processed}/{image_count} "
                            f"({elapsed:.1f}s)"
                            + (
                                f"  [slow {per_image_secs:.1f}s: {inner_base}]"
                                if per_image_secs >= _LOG_SLOW_IMAGE_THRESHOLD
                                else ""
                            )
                        )
                        last_log_t = now
                total_secs = time.monotonic() - t_zip_start
                log(
                    f"[upload] {base}: 完成 ({len(result.added)} added, "
                    f"{len(result.skipped)} skipped, {total_secs:.1f}s)"
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
    on_log: Optional[LogFn] = None,
) -> UploadResult:
    """批量处理；汇总 added / skipped。

    on_log 见 `accept_one` —— accept_many 自己也会在开始 / 结束各推一行，
    多文件场景中间逐 file 委托给 accept_one。
    """
    log = on_log or (lambda _s: None)
    files_list = list(files)
    out = UploadResult()
    t_all = time.monotonic()
    log(f"[upload] starting: {len(files_list)} file(s)")
    for name, stream in files_list:
        out.merge(
            accept_one(
                name, stream, dest_dir,
                convert_to_png=convert_to_png,
                remove_alpha_channel=remove_alpha_channel,
                on_log=on_log,
            )
        )
    elapsed = time.monotonic() - t_all
    log(
        f"[upload] all done in {elapsed:.1f}s "
        f"({len(out.added)} added, {len(out.skipped)} skipped)"
    )
    return out
