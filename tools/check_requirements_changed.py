#!/usr/bin/env python
"""Bootstrap helper: 检测 requirements.txt 自上次 venv 同步以来有没有变。

studio.sh / studio.bat 启动期调用本脚本，根据 stdout 决定是否补装新增依赖。

模式：
- 默认（读模式）：比对 requirements.txt 内容 hash 与 marker 文件
  - 输出 `stale`：内容变了 / 没 marker（首次启动或老 venv）→ caller 应跑 pip install
  - 输出 `current`：hash 一致 → skip
  - 输出 `missing`：requirements.txt 不存在 → skip

- `--update-marker`：成功同步后写入新 hash，输出 `written`

为什么用 content hash 不用 mtime：
- `git checkout` / `git pull` 在某些 git 配置下保留 commit 时间戳，mtime 会
  误判为「stale」触发不必要的 pip install
- hash 只对真实内容变化敏感，bulletproof

stdlib only —— 在 venv 刚装好（pip 都还没装包）时也能跑。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def compute_req_hash(req_path: Path) -> str:
    return hashlib.sha256(req_path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--marker", required=True,
        help="marker 文件路径（如 venv/.studio-requirements.sha256）",
    )
    parser.add_argument(
        "--requirements", default="requirements.txt",
        help="要比对的 requirements 文件",
    )
    parser.add_argument(
        "--update-marker", action="store_true",
        help="写入当前 hash 到 marker（同步成功后调）",
    )
    args = parser.parse_args(argv)

    req = Path(args.requirements)
    if not req.exists():
        print("missing")
        return 0

    current = compute_req_hash(req)
    marker = Path(args.marker)

    if args.update_marker:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(current, encoding="utf-8")
        print("written")
        return 0

    if not marker.exists():
        # 老 venv 没 marker → 视为 stale；caller 跑一次 pip 后写 marker，下次正常
        print("stale")
        return 0

    try:
        stored = marker.read_text(encoding="utf-8").strip()
    except OSError:
        # marker 损坏 → 当 stale，下次同步时重写
        print("stale")
        return 0

    print("stale" if stored != current else "current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
