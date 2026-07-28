"""studio_data 单实例锁 (`studio.infrastructure.single_instance`) 单测。

覆盖：
  - 抢锁 / 第二个句柄抢不到 / release 后可再抢
  - lock 文件父目录不存在时自动创建
  - release 幂等（未持有时 no-op）
  - lifespan 集成：锁被占时 startup 直接 raise，不执行任何破坏性清理
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.infrastructure.single_instance import LOCK_FILENAME, SingleInstanceLock


def test_acquire_blocks_second_handle(tmp_path: Path) -> None:
    lock_path = tmp_path / LOCK_FILENAME
    first = SingleInstanceLock(lock_path)
    second = SingleInstanceLock(lock_path)
    assert first.acquire()
    # OS 区域锁按句柄独占：同进程第二个句柄同样抢不到（等价于第二个进程）
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_acquire_creates_parent_dir(tmp_path: Path) -> None:
    lock = SingleInstanceLock(tmp_path / "not_yet" / "deeper" / LOCK_FILENAME)
    assert lock.acquire()
    assert lock.path.exists()
    lock.release()


def test_release_without_acquire_is_noop(tmp_path: Path) -> None:
    SingleInstanceLock(tmp_path / LOCK_FILENAME).release()  # 不抛即过


def test_double_acquire_same_instance_raises(tmp_path: Path) -> None:
    lock = SingleInstanceLock(tmp_path / LOCK_FILENAME)
    assert lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already acquired"):
            lock.acquire()
    finally:
        lock.release()


def test_lifespan_refuses_when_lock_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """双开保护：锁被占时 lifespan 在执行任何破坏性副作用之前 raise。

    锁检查位于 ensure_dirs / db.init_db / disk_cache.init（含 startup_clean
    rmtree）之前 —— 用「studio_data 里除了 lock 文件外什么都没被创建」验证
    「没执行到后续启动步骤」。
    """
    from fastapi.testclient import TestClient
    from studio import server
    from studio.infrastructure import paths

    studio_data = tmp_path / "studio_data"
    monkeypatch.setattr(paths, "STUDIO_DATA", studio_data)

    holder = SingleInstanceLock(studio_data / LOCK_FILENAME)
    assert holder.acquire()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            with TestClient(server.app):
                pass
        # 只有 lock 文件本身，没有 .cache/ logs/ 等后续启动产物
        assert [p.name for p in studio_data.iterdir()] == [LOCK_FILENAME]
    finally:
        holder.release()
