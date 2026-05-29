"""Worker 子进程入口通用模板（PR-8 commit 1 从 4 个 worker 抽出）。

`supervisor` 启动每个 worker 子进程都是相同模式：
    python -m studio.workers.<kind>_worker --job-id N
    → 读 project_jobs 行 → 跑业务 → 退出码反映成败

每个 worker 模块只需提供 `run(job_id: int) -> int` 主体，本模板包：
    - argparse 解析 `--job-id`
    - 调用 run + sys.exit

可选 helper `reconfigure_console_utf8()` — Windows 控制台默认 cp932/cp936，
写中文 / emoji 会 UnicodeEncodeError；调一次让 stdout/stderr 转 UTF-8 +
replace 模式。当前只有 tag_worker 需要（其它 worker 不写中文 caption）。
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable


def worker_main(run_fn: Callable[[int], int]) -> None:
    """4 个 worker 共用的 `if __name__ == "__main__"` 入口。

    worker 模块底部写：
        if __name__ == "__main__":
            from ._base import worker_main
            worker_main(run)

    PR-1 C4: 装 setup_logging (ADR-0009)。file=False — worker 走 stdout 进
    supervisor 重定向 jobs/<id>.log 单写（C round2 §1.2 决策；0.13.x ADR-0009
    §还的债"演进双写"后改 True）。process 名格式 "worker:<module>/<job_id>"，
    便于 jq 按 process 过滤；module 取 sys.argv[0] basename（download_worker.py
    → download_worker → 去掉 _worker 后缀 = download）。
    """
    import os
    from ..infrastructure.logging import setup_logging
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", type=int, required=True)
    args = p.parse_args()
    kind = _worker_kind_from_argv()
    # PROCESS_ENV 由 supervisor C6 后设置；环境无则按 argv 推断
    process = os.environ.get("ANIMA_PROCESS_NAME") or f"worker:{kind}/{args.job_id}"
    setup_logging(process, file=False, console=True)
    sys.exit(run_fn(args.job_id))


def _worker_kind_from_argv() -> str:
    """sys.argv[0] basename → kind（download_worker.py → download）。"""
    from pathlib import Path as _Path
    name = _Path(sys.argv[0]).stem
    if name.endswith("_worker"):
        name = name[: -len("_worker")]
    return name or "unknown"


def reconfigure_console_utf8() -> None:
    """Windows 控制台默认 cp932/cp936，写中文 / emoji 会 UnicodeEncodeError。
    强制 stdout/stderr 用 UTF-8 + 替换不可编码字符，让 progress 永远不抛。

    当前只 tag_worker 顶层调用（其它 worker 不写中文 caption）。supervisor
    `_popen` 已经给所有 worker 子进程注入 `PYTHONIOENCODING=utf-8 / PYTHONUTF8=1`
    env，但 Windows 上少数 console host 仍要 reconfigure 双保险。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass
