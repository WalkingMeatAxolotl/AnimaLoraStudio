#!/usr/bin/env python
"""下载 Anima 训练所需的全部模型 + tokenizer（CLI 薄壳）。

实际逻辑在 `studio.services.model_downloader`，CLI 和 Studio 设置页 UI 共用。

最终落地结构（默认 ./anima/，与 anima_train._guess_default_paths 一致）：
    anima/
      diffusion_models/anima-preview3-base.safetensors
      vae/qwen_image_vae.safetensors
      text_encoders/                # Qwen3 模型 + tokenizer
      t5_tokenizer/                 # T5 仅 tokenizer，不要权重

用法:
    python tools/download_models.py
    python tools/download_models.py --variant preview2
    python tools/download_models.py --no-mirror
    python tools/download_models.py --skip-main --skip-vae
    python tools/download_models.py --output /data/anima
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台 cp936/cp932 写中文 / emoji 会 UnicodeEncodeError，强制 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# 让 `python tools/download_models.py` 也能 import studio package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.services.model_downloader import (  # noqa: E402
    ANIMA_VARIANTS,
    LATEST_ANIMA,
    download_anima_main,
    download_anima_vae,
    download_qwen3,
    download_t5_tokenizer,
    models_root,
    setup_mirror,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="下载 Anima 训练所需的全部模型 + tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Anima 主模型版本（--variant）:
{chr(10).join(f"  {k:<14} {v}" for k, v in ANIMA_VARIANTS.items())}
  latest         (= {LATEST_ANIMA})

国内用户默认走 hf-mirror.com；如果镜像有问题加 --no-mirror。
""",
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="使用 HuggingFace 官方源（默认走 hf-mirror.com）",
    )
    parser.add_argument(
        "--output", default="",
        help="目标根目录（默认 ./anima/）",
    )
    parser.add_argument(
        "--variant", default="latest",
        choices=list(ANIMA_VARIANTS) + ["latest"],
        help=f"Anima 主模型版本（默认 latest = {LATEST_ANIMA}）",
    )
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-vae",  action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-t5",   action="store_true")
    args = parser.parse_args()

    setup_mirror(use_mirror=not args.no_mirror)
    print(
        "使用镜像: https://hf-mirror.com" if not args.no_mirror
        else "使用官方源: https://huggingface.co"
    )

    out_root = Path(args.output) if args.output else models_root()
    print(f"📁 目标根目录: {out_root.absolute()}")

    ok = True
    if not args.skip_main:
        ok &= download_anima_main(out_root, args.variant)
    if not args.skip_vae:
        ok &= download_anima_vae(out_root)
    if not args.skip_qwen:
        ok &= download_qwen3(out_root)
    if not args.skip_t5:
        ok &= download_t5_tokenizer(out_root)

    print()
    print("=" * 50)
    print("✅ 全部下载完成！" if ok else "⚠️  部分下载失败，详见上方日志")
    print("=" * 50)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
