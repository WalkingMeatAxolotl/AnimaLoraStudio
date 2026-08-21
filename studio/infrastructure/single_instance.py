"""studio_data 单实例锁 —— 防第二个 server 进程对同一份数据执行破坏性启动清理。

背景（2026-07-28 定案的丢图 root cause）：uvicorn 是**先跑完 lifespan、之后才
bind 端口**。用户双开 `python -m studio` 时，注定 bind 失败的那个实例在死掉之前
已经把 lifespan 里的破坏性清理跑了一遍 —— `disk_cache.startup_clean` 会 rmtree
掉活 server 正在用的 cache session 目录，从那一刻起活 server 每张出图都丢。
db migration 等其余启动副作用同样不该在两个进程间并发。

所以独占的单位是 **studio_data 目录**（真正要保护的资源），不是端口：lifespan
一进来就抢 `studio_data/.server.lock`，抢不到立刻 raise，让第二个实例在执行任何
破坏性操作**之前**退出。

机制：OS 级文件区域锁（Windows `msvcrt.locking` / POSIX `fcntl.flock`），随进程
退出（含 SIGKILL / 断电）自动释放，没有 stale lock 问题 —— 这是不用「写 pid 进
lock 文件再判活」方案的原因。lock 文件本身 0 字节、不删除（Windows 上删除与
下次 open 有竞态，留着无害）。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import IO, Optional

logger = logging.getLogger(__name__)

LOCK_FILENAME = ".server.lock"


class SingleInstanceLock:
    """一个 lock 文件的独占句柄。acquire 失败 = 别的进程正持有。

    单次使用：acquire() → （进程生命周期）→ release()。release 后可以再
    acquire（测试用）；同一实例重复 acquire 而不 release 是使用错误。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Optional[IO[bytes]] = None

    def acquire(self) -> bool:
        """非阻塞抢锁。True=拿到；False=已被其他进程（或其他句柄）持有。"""
        if self._handle is not None:
            raise RuntimeError("lock already acquired by this instance")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # "a+b" 不截断已有文件；并发 open 同一路径是安全的，独占靠区域锁
        handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                # 锁首字节。文件是 0 字节也允许（Windows 区域锁可超出 EOF）
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        """释放并关闭句柄。未持有时 no-op（shutdown 路径容错）。"""
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            # close() 也会随句柄释放锁；这里失败不影响正确性
            logger.warning(
                "unlock failed for %s; the lock is reclaimed on the next start",
                self.path, exc_info=True,
            )
        finally:
            handle.close()
