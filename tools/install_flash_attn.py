#!/usr/bin/env python
"""flash_attn prebuild wheel 安装 CLI（Studio Settings UI 的命令行同步入口）。

使用：
    python tools/install_flash_attn.py            # 自动选最优 wheel 装
    python tools/install_flash_attn.py --url URL  # 手动指定 wheel URL
    python tools/install_flash_attn.py --dry-run  # 只列环境 + 候选，不真装
    python tools/install_flash_attn.py --force    # 已装也重装

退出码：0 成功 / 1 安装失败 / 2 环境不支持

实现共享 studio.services.flash_attention_setup —— 与 UI 走完全相同的 wheel 选择逻辑。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本在 venv 直接 `python tools/install_flash_attn.py` 跑得了 —— 注入仓库根
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="flash_attn prebuild wheel 安装",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", help="手动指定 wheel URL（跳过自动匹配）")
    parser.add_argument(
        "--dry-run", action="store_true", help="只列环境 + 候选，不真装"
    )
    parser.add_argument(
        "--force", action="store_true", help="即使已装也重装"
    )
    args = parser.parse_args(argv)

    from studio.services import flash_attention_setup as fa  # noqa: PLC0415

    env = fa.detect_env()
    print("[env] python:   ", env.get("python_tag"))
    print("[env] platform: ", env.get("platform"))
    print("[env] cuda:     ", env.get("cuda_tag"), f"({env.get('cuda_ver')})")
    print("[env] torch:    ", env.get("torch_tag"), f"({env.get('torch_ver')})")

    status = fa.current_status()
    if status["installed"]:
        print(f"[status] flash_attn=={status['version']} 已安装")
        if not args.force and not args.dry_run:
            print("       使用 --force 重装；否则不动")
            return 0
    else:
        print("[status] flash_attn 未安装")

    if not env.get("platform"):
        print(
            "[error] 不支持的平台（仅 linux_x86_64 / win_amd64 有 prebuild wheel）",
            file=sys.stderr,
        )
        return 2

    if args.dry_run or not args.url:
        candidates, fetch_error = fa.find_candidates(env)
        if fetch_error:
            print(f"[warn] 拉候选列表失败: {fetch_error}", file=sys.stderr)
            print(
                "       可手动传 --url，从 "
                "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases 选",
                file=sys.stderr,
            )
            if args.dry_run:
                return 0
            return 2
        if not candidates:
            print("[warn] 没找到匹配 wheel（先看上面的 env 是否完整）", file=sys.stderr)
            return 2

        print(f"\n[candidates] 共 {len(candidates)} 个 wheel（按 score 降序）：")
        for i, c in enumerate(candidates[:10]):
            mark = "✓" if c["usable"] else "✗"
            note_str = "; ".join(c["notes"]) if c["notes"] else ""
            print(f"  {mark} score={c['score']:>3}  {c['name']}")
            if note_str:
                print(f"      {note_str}")
        if len(candidates) > 10:
            print(f"  ... 另 {len(candidates) - 10} 个未列出")

    if args.dry_run:
        return 0

    try:
        result = fa.install(args.url)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"\n[ok] flash_attn=={result['version']} 已装")
    print(f"     {result['url']}")
    if result.get("restart_required"):
        print("[note] flash_attn 是 C extension，已运行的 Studio / 训练进程需要重启才生效")
    return 0


if __name__ == "__main__":
    sys.exit(main())
