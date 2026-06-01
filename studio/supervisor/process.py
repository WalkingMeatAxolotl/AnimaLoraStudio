"""跨平台杀进程树工具（PR-4 从 supervisor.py 抽出）。

Windows 上 `proc.kill()` 只杀 immediate child，DataLoader workers /
accelerate 的 sub-subprocess 会留下来占着 GPU；用 `taskkill /T /F` 能
递归到整个进程树。POSIX 用 killpg。
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess

logger = logging.getLogger(__name__)


def _kill_process_tree(pid: int) -> None:
    """杀掉以 pid 为根的整棵进程树。

    Windows 上 `proc.kill()` 只杀 immediate child，DataLoader workers /
    accelerate 的 sub-subprocess 会留下来占着 GPU；用 `taskkill /T /F` 能
    递归到整个进程树。POSIX 用 killpg。
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                check=False, capture_output=True, timeout=10,
            )
        except Exception:
            logger.exception("taskkill /T /F failed for pid %d", pid)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            logger.exception("killpg failed for pid %d", pid)
