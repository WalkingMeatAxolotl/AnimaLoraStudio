"""加密磁盘 cache (`studio.services.inference.disk_cache`) 单测。

覆盖：
  - put/get 加解密 roundtrip
  - put 多次 → 每张图都有独立 nonce（同样 PNG 内容生成不同密文）
  - drop_task / clear_all
  - startup_clean 扫清所有 session-*
  - LRU by count / by bytes / configure shrink
  - "扫盘抗性" 烟雾测试：写出的文件无 PNG magic bytes、高熵字节
  - 不同 session 的 key 解不开旧文件
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from studio.services.inference import disk_cache


# ---------------------------------------------------------------- 测试 helper


def _png_bytes(payload: bytes = b"hello-png-content") -> bytes:
    """模拟一张 PNG —— 关键是带 PNG magic bytes 让"扫盘抗性"测试有意义。"""
    return b"\x89PNG\r\n\x1a\n" + payload


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "generate_cache"


@pytest.fixture
def cache(cache_root: Path):
    """每个测试拿一个 fresh SessionCache。直接构造（不走 init() 单例）以隔离。"""
    sc = disk_cache.SessionCache(root=cache_root, max_count=100, max_bytes=10**9)
    sc.ensure_dir()
    yield sc
    # cleanup 防文件残留污染其它测试
    sc.clear_all()


# ---------------------------------------------------------------- 加密 roundtrip


def test_put_get_roundtrip(cache: disk_cache.SessionCache) -> None:
    png = _png_bytes(b"abc")
    cache.put(1, "a.png", png, {"mode": "single", "prompts": ["test"]})
    assert cache.get_image(1, "a.png") == png


def test_get_missing_returns_none(cache: disk_cache.SessionCache) -> None:
    assert cache.get_image(99, "nope.png") is None


def test_put_writes_different_ciphertext_each_time(cache: disk_cache.SessionCache) -> None:
    """同 task 同 filename 反复 put 同样 png → 文件 nonce 每次都新，密文不同。

    防 nonce 复用泄露 keystream（虽然这条威胁模型不在意 integrity，仍然该
    保持 crypto hygiene）。
    """
    png = _png_bytes(b"same")
    cache.put(1, "x.png", png, {})
    blob1 = next(iter(cache._index.values())).file_path.read_bytes()
    cache.put(1, "x.png", png, {})
    blob2 = next(iter(cache._index.values())).file_path.read_bytes()
    assert blob1 != blob2


def test_put_recreates_session_dir_if_deleted(cache: disk_cache.SessionCache) -> None:
    """session 目录被外部 rmtree（另一进程 startup_clean / 手删）后 put 自愈。

    2026-07-28 丢图 root cause 回归：没有自愈时目录一旦消失，每张
    cache_image 都 FileNotFoundError 丢弃，直到 clear_all / 重启。
    """
    cache.put(1, "a.png", _png_bytes(b"a"), {})
    shutil.rmtree(cache.session_dir)
    cache.put(1, "b.png", _png_bytes(b"b"), {})
    assert cache.get_image(1, "b.png") == _png_bytes(b"b")
    # 旧 entry 文件已随目录消失：get 返 None 并剔出 index（既有语义不变）
    assert cache.get_image(1, "a.png") is None
    assert cache.total_count() == 1


def test_overwrite_same_key_deletes_old_file(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(b"v1"), {})
    old_path = next(iter(cache._index.values())).file_path
    cache.put(1, "a.png", _png_bytes(b"v2"), {})
    assert not old_path.exists()
    assert cache.total_count() == 1
    assert cache.get_image(1, "a.png") == _png_bytes(b"v2")


# ---------------------------------------------------------------- 扫盘抗性


def test_on_disk_blob_has_no_png_magic(cache: disk_cache.SessionCache) -> None:
    """加密后的文件首字节不能匹配 PNG header `89 50 4E 47`。"""
    cache.put(1, "a.png", _png_bytes(b"secret-payload" * 100), {})
    file_path = next(iter(cache._index.values())).file_path
    blob = file_path.read_bytes()
    assert not blob.startswith(b"\x89PNG"), "leaked PNG magic bytes"
    assert not blob[16:].startswith(b"\x89PNG"), "leaked PNG magic bytes after nonce"


def test_on_disk_blob_is_high_entropy(cache: disk_cache.SessionCache) -> None:
    """密文应接近均匀分布。粗略卡方：256 个 byte 桶里 single dominant 桶
    占比应远低于纯 PNG（PNG header / zlib stream 会让前几字节固定）。"""
    cache.put(1, "a.png", _png_bytes(b"\x00" * 4096), {})
    file_path = next(iter(cache._index.values())).file_path
    blob = file_path.read_bytes()
    # PNG bytes b"\x00" 全零会让明文 entropy 极低；密文反过来应均匀。
    # 任何单个 byte 值占比 > 5% 视为可疑。
    counts = [0] * 256
    for b in blob:
        counts[b] += 1
    max_share = max(counts) / len(blob)
    assert max_share < 0.05, f"密文 byte 分布异常均匀失败: max share {max_share:.3f}"


def test_filename_has_no_extension_hint(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(), {})
    file_path = next(iter(cache._index.values())).file_path
    assert file_path.suffix == ".bin", f"unexpected extension {file_path.suffix}"
    # 不能含原 .png filename
    assert "a.png" not in file_path.name


# ---------------------------------------------------------------- drop / clear


def test_drop_task_removes_files_and_index(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(), {})
    cache.put(1, "b.png", _png_bytes(), {})
    cache.put(2, "c.png", _png_bytes(), {})
    files_before = list(cache.session_dir.iterdir())
    assert len(files_before) == 3
    n = cache.drop_task(1)
    assert n == 2
    files_after = list(cache.session_dir.iterdir())
    assert len(files_after) == 1
    assert cache.get_image(1, "a.png") is None
    assert cache.get_image(2, "c.png") is not None


def test_clear_all_empties_session_dir(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(), {})
    cache.put(1, "b.png", _png_bytes(), {})
    cache.clear_all()
    assert cache.total_count() == 0
    assert cache.total_bytes() == 0
    # session_dir 在 clear_all 后被重 mkdir（让后续 put 能继续）
    assert cache.session_dir.exists()
    assert list(cache.session_dir.iterdir()) == []


# ---------------------------------------------------------------- startup_clean


def test_startup_clean_removes_all_session_dirs(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    # 模拟上次进程残留
    (root / "session-aaaaaaaa").mkdir()
    (root / "session-bbbbbbbb").mkdir()
    (root / "session-aaaaaaaa" / "file1.bin").write_bytes(b"garbage")
    # 非 session-* 目录不动
    (root / "other_dir").mkdir()
    n = disk_cache.startup_clean(root)
    assert n == 2
    assert not (root / "session-aaaaaaaa").exists()
    assert not (root / "session-bbbbbbbb").exists()
    assert (root / "other_dir").exists()


def test_startup_clean_on_nonexistent_root(tmp_path: Path) -> None:
    """root 不存在不报错。"""
    assert disk_cache.startup_clean(tmp_path / "doesnotexist") == 0


# ---------------------------------------------------------------- LRU


def test_lru_evicts_by_count(cache_root: Path) -> None:
    sc = disk_cache.SessionCache(root=cache_root, max_count=3, max_bytes=10**9)
    sc.ensure_dir()
    for i in range(5):
        sc.put(1, f"img{i}.png", _png_bytes(bytes([i])), {})
    assert sc.total_count() == 3
    assert sc.get_image(1, "img0.png") is None
    assert sc.get_image(1, "img1.png") is None
    assert sc.get_image(1, "img4.png") is not None
    # 文件也应该被删
    assert len(list(sc.session_dir.iterdir())) == 3
    sc.clear_all()


def test_lru_evicts_by_bytes(cache_root: Path) -> None:
    # 一张图：[16 nonce] + [4 snap_len] + [snap_json + png] payload
    # snap={}=b"{}" 是 2 bytes；png = b"hello"（5 bytes） → 文件 27 bytes
    sc = disk_cache.SessionCache(root=cache_root, max_count=10**6, max_bytes=80)
    sc.ensure_dir()
    sc.put(1, "a.png", b"hello", {})
    sc.put(1, "b.png", b"hello", {})
    sc.put(1, "c.png", b"hello", {})
    # 第 4 张触发 LRU（80 / 27 ≈ 容 2.96 张）
    sc.put(1, "d.png", b"hello", {})
    assert sc.get_image(1, "a.png") is None
    assert sc.get_image(1, "d.png") == b"hello"
    sc.clear_all()


def test_lru_get_marks_recent(cache_root: Path) -> None:
    sc = disk_cache.SessionCache(root=cache_root, max_count=3, max_bytes=10**9)
    sc.ensure_dir()
    sc.put(1, "a.png", _png_bytes(b"a"), {})
    sc.put(1, "b.png", _png_bytes(b"b"), {})
    sc.put(1, "c.png", _png_bytes(b"c"), {})
    # 访问 a 让它变最新
    assert sc.get_image(1, "a.png") == _png_bytes(b"a")
    sc.put(1, "d.png", _png_bytes(b"d"), {})
    # 现在最旧应该是 b（不是 a）
    assert sc.get_image(1, "b.png") is None
    assert sc.get_image(1, "a.png") == _png_bytes(b"a")
    sc.clear_all()


def test_configure_shrink_evicts(cache_root: Path) -> None:
    sc = disk_cache.SessionCache(root=cache_root, max_count=10, max_bytes=10**9)
    sc.ensure_dir()
    for i in range(5):
        sc.put(1, f"i{i}.png", _png_bytes(bytes([i])), {})
    assert sc.total_count() == 5
    sc.configure(max_count=2)
    assert sc.total_count() == 2
    # 最新两条留下
    assert sc.get_image(1, "i4.png") is not None
    assert sc.get_image(1, "i0.png") is None
    sc.clear_all()


# ---------------------------------------------------------------- key 隔离


def test_different_session_cannot_decrypt(cache_root: Path) -> None:
    """旧 session 写的文件，新 session 的 key 解不开 —— 重启后残留文件就是
    乱字节。可能 raise（snapshot len 随机变成超大值）或返回乱字节，但绝
    不会返回原始 PNG。
    """
    sc1 = disk_cache.SessionCache(root=cache_root, max_count=10, max_bytes=10**9)
    sc1.ensure_dir()
    sc1.put(1, "secret.png", _png_bytes(b"sensitive"), {})
    file_path = next(iter(sc1._index.values())).file_path
    blob = file_path.read_bytes()
    sc2 = disk_cache.SessionCache(root=cache_root, max_count=10, max_bytes=10**9)
    # 用 sc2 的 key 尝试解 sc1 的 blob：要么出 garbage 要么 raise；不可能等于原文
    original = _png_bytes(b"sensitive")
    try:
        result = disk_cache._decrypt_and_strip(sc2.aes_key, blob)
    except (ValueError, Exception):
        pass  # 期望路径之一：snapshot_len 随机值超出范围
    else:
        assert result != original
    sc1.clear_all()


# ---------------------------------------------------------------- module-level


def test_module_init_clears_stale_session_dirs(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "session-stale1").mkdir()
    (root / "session-stale2").mkdir()
    # 重置 module singleton 防被前测试污染
    monkeypatch.setattr(disk_cache, "_session", None)
    sc = disk_cache.init(root)
    try:
        assert not (root / "session-stale1").exists()
        assert not (root / "session-stale2").exists()
        # 新 session 目录是 session-<sc.session_id>
        assert sc.session_dir.exists()
    finally:
        sc.clear_all()
        monkeypatch.setattr(disk_cache, "_session", None)


def test_module_get_session_raises_when_not_initialized(monkeypatch) -> None:
    monkeypatch.setattr(disk_cache, "_session", None)
    with pytest.raises(RuntimeError, match="not initialized"):
        disk_cache.get_session()


def test_module_clear_all_safe_when_uninitialized(monkeypatch) -> None:
    monkeypatch.setattr(disk_cache, "_session", None)
    # 不报错就行
    disk_cache.clear_all()
    assert disk_cache.total_count() == 0
    assert disk_cache.total_bytes() == 0
