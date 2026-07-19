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


def _kill_process_tree_psutil(pid: int) -> None:
    """taskkill 不可用/被策略拒绝时的 Windows fallback。

    psutil 是项目必需依赖。先杀最深子进程再杀根，尽量保留 taskkill 的 tree
    语义；NoSuchProcess 是并发退出的正常竞态。
    """
    import psutil

    try:
        root = psutil.Process(pid)
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    except Exception:
        logger.exception("psutil enumerate process tree failed for pid %d", pid)
        descendants = []
        try:
            root = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return

    for proc in reversed(descendants):
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        except Exception:
            logger.exception("psutil kill failed for child pid %d", proc.pid)
    try:
        root.kill()
    except psutil.NoSuchProcess:
        pass
    except Exception:
        logger.exception("psutil kill failed for root pid %d", pid)


def _kill_process_tree_windows(pid: int) -> None:
    """Windows taskkill 主路径；失败时保证落到 psutil tree fallback。"""
    try:
        result = subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            check=False, capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return
        detail = (result.stderr or result.stdout or b"").decode(
            errors="replace"
        ).strip()
        logger.warning(
            "taskkill failed for pid %d (rc=%d): %s; falling back to psutil",
            pid, result.returncode, detail or "no output",
        )
    except Exception:
        logger.exception("taskkill /T /F failed for pid %d", pid)
    _kill_process_tree_psutil(pid)


def _kill_process_tree(pid: int) -> None:
    """杀掉以 pid 为根的整棵进程树。

    Windows 上 `proc.kill()` 只杀 immediate child，DataLoader workers /
    accelerate 的 sub-subprocess 会留下来占着 GPU；用 `taskkill /T /F` 能
    递归到整个进程树。POSIX 用 killpg。
    """
    if os.name == "nt":
        _kill_process_tree_windows(pid)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            logger.exception("killpg failed for pid %d", pid)
