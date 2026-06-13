"""测试出图的加密磁盘 cache（替代之前 cache.py 的纯内存 dict）。

为什么改磁盘：内存 cache 吃常驻 RAM（200 张 / 500MB 上限），用户长时间炼丹 +
batch 出图会逼到上限。挪磁盘后 RAM 压力下来；同时为 session 持久（refresh
/ 切路由不丢历史栏）让路。

为什么加密：用户在云训练机上跑测试出图时，部分国内云厂商扫盘按文件名 /
扩展名 / PNG magic bytes / 内容分类（CNN）抓敏感图。明文 PNG 落盘哪怕只
活几秒也可能被扫到；加密后磁盘文件是高熵随机字节，无扩展、无 magic bytes，
扫盘识不出。

威胁模型 = 防被动扫盘：不防本机攻击者 attach 进程 dump key、不防有人故意
走 app 代码读取图片；这两件事这条防线本来也挡不住，加 AEAD / 真正的非对称
crypto 就是 over-engineering。

机制（"session 指纹"）：
  - 启动时进程生成 session_id (uuid4) + aes_key (32 bytes random)，**只在进程
    内存**，不落盘
  - cache 目录 `studio_data/.cache/generate/session-<uuid>/`
  - 每张图一个文件 `<file_uuid>.bin`，无扩展，self-contained：
    `[16B nonce][SHAKE-128 keystream XOR(payload)]`
    其中 payload = `[4B snapshot_len][snapshot_json_utf8][png_bytes]`
  - 启动时扫 root 下所有 `session-*` 目录全 rmtree（包括上次 SIGKILL / 断电
    残留）—— 新 session_id 随机，保证不撞旧目录
  - shutdown 删 session 目录 + 进程退出 key 一起没

异常退出时残留文件无 key 解不开 = 一堆乱字节，扫盘工具识别不出是图。下次
启动 startup_clean 顺手清掉。

Crypto 选择：纯 stdlib `hashlib.shake_128` keystream + XOR。SHAKE-128 是 SHA-3
变体（NIST 标准），C 实现，1.5MB PNG 加解密毫秒级。无 AEAD 但威胁模型不
需要 integrity——扫盘看 magic bytes / entropy / CNN 分类的就够了。零额外
依赖避开了 `cryptography` 库的 DLL 兼容麻烦。

API 参考 cache.py（被本模块替代）：put + get_image + list_filenames +
drop_task + clear_all + configure + list_index（新）。

"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_COUNT = 200
DEFAULT_MAX_BYTES = 500 * 1024 * 1024
_SESSION_DIR_PREFIX = "session-"
_NONCE_LEN = 16
_SNAPSHOT_LEN_HEADER = 4  # big-endian uint32


@dataclass
class _Entry:
    """index 里一条 entry —— 文件路径 + 内存里的 snapshot 缓存 + 大小。"""
    file_path: Path
    snapshot: dict[str, Any]
    created_at: float
    size: int  # 文件加密后的字节数（含 nonce + payload）
    mode: str  # 'single' | 'xy'，前端历史栏分组用
    task_id: int
    filename: str
    # XY 模式时 daemon image_done 事件里带的 {xi, yi, xv, yv}；single 模式 None。
    # list_index 重建 xyMeta.samples 给前端 PreviewXYGrid 用。
    xy_info: Optional[dict[str, Any]] = None


@dataclass
class SessionCache:
    """session-scoped 加密磁盘 cache，线程安全。

    每个 server 进程持有一个实例（lifespan 启动时 init，shutdown 时 clear_all）。
    SIGKILL / 断电后残留目录在下次启动由 startup_clean() 兜底清掉。
    """
    root: Path
    max_count: int = DEFAULT_MAX_COUNT
    max_bytes: int = DEFAULT_MAX_BYTES

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    aes_key: bytes = field(default_factory=lambda: os.urandom(32))
    _lock: threading.RLock = field(default_factory=threading.RLock)
    # (task_id, filename) → _Entry；OrderedDict 维护 LRU
    _index: "collections.OrderedDict[tuple[int, str], _Entry]" = field(
        default_factory=collections.OrderedDict,
    )
    _bytes_total: int = 0

    @property
    def session_dir(self) -> Path:
        return self.root / f"{_SESSION_DIR_PREFIX}{self.session_id}"

    def ensure_dir(self) -> None:
        """mkdir session 目录。lifespan init 完调一次。"""
        self.session_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- put/get
    def put(
        self,
        task_id: int,
        filename: str,
        data: bytes,
        snapshot: dict[str, Any],
        *,
        mode: str = "single",
        xy_info: Optional[dict[str, Any]] = None,
    ) -> None:
        """daemon image_done 时调；同 (task_id, filename) 重复则覆盖（删旧文件 + 写新）。

        snapshot：前端构造的 GenerateParamsSnapshot dict，跟 PNG 绑死塞进加密
        payload header；list_index() 也用同一份返回给前端，避免双 source。

        xy_info：XY 模式时 daemon image_done 携带的 {xi, yi, xv, yv}，重建
        PreviewXYGrid 用；single 模式 None。
        """
        snap_bytes = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
        if len(snap_bytes) >= 2**32:
            raise ValueError("snapshot too large (>4GiB)")
        payload = len(snap_bytes).to_bytes(_SNAPSHOT_LEN_HEADER, "big") + snap_bytes + data
        nonce = os.urandom(_NONCE_LEN)
        ciphertext = _xor(payload, _keystream(self.aes_key, nonce, len(payload)))
        blob = nonce + ciphertext

        file_id = uuid.uuid4().hex
        file_path = self.session_dir / f"{file_id}.bin"
        # 原子写：写 tmp + rename。session 目录是独占的，不存在并发写同名情况
        tmp = file_path.with_suffix(".bin.tmp")
        tmp.write_bytes(blob)
        tmp.replace(file_path)

        with self._lock:
            key = (task_id, filename)
            old = self._index.pop(key, None)
            if old is not None:
                self._bytes_total -= old.size
                _safe_unlink(old.file_path)
            entry = _Entry(
                file_path=file_path,
                snapshot=snapshot,
                created_at=time.time(),
                size=len(blob),
                mode=mode,
                task_id=task_id,
                filename=filename,
                xy_info=xy_info,
            )
            self._index[key] = entry
            self._bytes_total += entry.size
            self._enforce_limits_locked()

    def get_image(self, task_id: int, filename: str) -> Optional[bytes]:
        """读图：拿 file_path → 读盘 → 解密 → strip snapshot header → return PNG bytes。

        命中 move_to_end（LRU 看作最近使用）。文件丢了（外部删 / 解密失败）
        → 返 None 且把 index entry 也剔掉，避免一直挂着死引用。
        """
        with self._lock:
            key = (task_id, filename)
            entry = self._index.get(key)
            if entry is None:
                return None
            self._index.move_to_end(key)
            file_path = entry.file_path
        # 解密 IO 不持锁，避免一张大图卡住整个 cache
        try:
            blob = file_path.read_bytes()
        except FileNotFoundError:
            with self._lock:
                self._index.pop(key, None)
                self._bytes_total -= entry.size
            logger.warning("disk cache file gone: %s", file_path)
            return None
        try:
            return _decrypt_and_strip(self.aes_key, blob)
        except Exception:
            logger.exception("decrypt failed for %s", file_path)
            return None

    # ---------------------------------------------------------------- 列表 / 删
    def list_filenames(self, task_id: int) -> list[str]:
        with self._lock:
            return sorted(fn for (tid, fn) in self._index if tid == task_id)

    def list_index(self) -> list[dict[str, Any]]:
        """前端历史栏拉 /api/generate/cache/index 用。

        按 task_id 聚合 —— 同 task 的多张图（XY 一格一张）合成一条 history
        entry。返回结构对齐前端 CacheEntry adapter：
            { id, taskId, mode, createdAt (ms), filenames[], params, samples? }
        其中 samples 仅 mode=xy 时存在，列 [{filename, xy:{xi,yi,xv,yv}}]
        给 PreviewXYGrid 重建网格用。createdAt 取该 task 最新 entry 的时间。
        按 createdAt desc 排（最新在前）。
        """
        with self._lock:
            entries = list(self._index.values())

        by_task: dict[int, list[_Entry]] = collections.defaultdict(list)
        for e in entries:
            by_task[e.task_id].append(e)

        out: list[dict[str, Any]] = []
        for task_id, group in by_task.items():
            group_sorted = sorted(group, key=lambda e: e.filename)
            first = group_sorted[0]
            latest_created = max(e.created_at for e in group)
            item: dict[str, Any] = {
                "id": f"cache:{task_id}",
                "taskId": task_id,
                "mode": first.mode,
                "createdAt": int(latest_created * 1000),
                "filenames": [e.filename for e in group_sorted],
                "params": first.snapshot,
            }
            if first.mode == "xy":
                item["samples"] = [
                    {"filename": e.filename, "xy": e.xy_info}
                    for e in group_sorted
                    if e.xy_info is not None
                ]
            out.append(item)

        out.sort(key=lambda x: x["createdAt"], reverse=True)
        return out

    def drop_task(self, task_id: int) -> int:
        with self._lock:
            keys = [k for k in self._index if k[0] == task_id]
            for k in keys:
                entry = self._index.pop(k)
                self._bytes_total -= entry.size
                _safe_unlink(entry.file_path)
            return len(keys)

    def total_count(self) -> int:
        with self._lock:
            return len(self._index)

    def total_bytes(self) -> int:
        with self._lock:
            return self._bytes_total

    def clear_all(self) -> None:
        """删整个 session 目录 + 清 index。lifespan shutdown 调；测试也用。"""
        with self._lock:
            self._index.clear()
            self._bytes_total = 0
        if self.session_dir.exists():
            try:
                shutil.rmtree(self.session_dir)
            except OSError:
                logger.exception("clear_all: rmtree failed for %s", self.session_dir)
        # 让后续 put() 能继续工作（重 mkdir）
        self.ensure_dir()

    def configure(
        self,
        *,
        max_count: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> None:
        """动态调上限。改完立刻 enforce。"""
        with self._lock:
            if max_count is not None:
                self.max_count = int(max_count)
            if max_bytes is not None:
                self.max_bytes = int(max_bytes)
            self._enforce_limits_locked()

    def _enforce_limits_locked(self) -> None:
        while self._index and (
            len(self._index) > self.max_count or self._bytes_total > self.max_bytes
        ):
            _, entry = self._index.popitem(last=False)  # 最旧
            self._bytes_total -= entry.size
            _safe_unlink(entry.file_path)


# ---------------------------------------------------------------------------
# 模块级 singleton —— lifespan init 创建；测试可显式 init 自己的实例
# ---------------------------------------------------------------------------

_session: Optional[SessionCache] = None
_session_lock = threading.Lock()


def init(
    root: Path,
    *,
    max_count: int = DEFAULT_MAX_COUNT,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> SessionCache:
    """初始化 module singleton。lifespan startup 调一次。

    会先 startup_clean(root) 清掉残留 session-* 目录，再创建新 session。
    """
    global _session
    with _session_lock:
        if _session is not None:
            # 已有 session 时再 init = 旧 session 不再可达，清掉再开新的
            _session.clear_all()
        startup_clean(root)
        sc = SessionCache(root=root, max_count=max_count, max_bytes=max_bytes)
        sc.ensure_dir()
        _session = sc
        logger.info("disk_cache initialized: session_id=%s dir=%s", sc.session_id, sc.session_dir)
        return sc


def get_session() -> SessionCache:
    """拿当前 singleton。未 init 报错（防止隐式 lazy init 隐藏调用顺序 bug）。"""
    if _session is None:
        raise RuntimeError("disk_cache not initialized; call init() first")
    return _session


def startup_clean(root: Path) -> int:
    """扫 root 下所有 `session-*` 目录全 rmtree。

    新 session_id 是随机 uuid，保证不会跟历史撞，所以"删所有 session-*"恒等于
    "删 stale"。返回删了多少个目录。
    """
    if not root.exists():
        return 0
    n = 0
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(_SESSION_DIR_PREFIX):
            try:
                shutil.rmtree(child)
                n += 1
            except OSError:
                logger.warning("startup_clean: rmtree failed for %s", child, exc_info=True)
    if n:
        logger.info("startup_clean: removed %d stale session dir(s) under %s", n, root)
    return n


# ---------------------------------------------------------------------------
# 模块级 shortcut —— 替代旧 cache.py 的同名函数，调用方零改动
# ---------------------------------------------------------------------------


def cache_image(
    task_id: int,
    filename: str,
    data: bytes,
    snapshot: Optional[dict[str, Any]] = None,
    *,
    mode: str = "single",
    xy_info: Optional[dict[str, Any]] = None,
) -> None:
    """daemon image_done 时调（替代旧 cache.cache_image）。snapshot 缺省 {}。"""
    get_session().put(
        task_id, filename, data, snapshot or {},
        mode=mode, xy_info=xy_info,
    )


def get_image(task_id: int, filename: str) -> Optional[bytes]:
    return get_session().get_image(task_id, filename)


def list_filenames(task_id: int) -> list[str]:
    return get_session().list_filenames(task_id)


def list_index() -> list[dict[str, Any]]:
    return get_session().list_index()


def drop_task(task_id: int) -> int:
    return get_session().drop_task(task_id)


def total_count() -> int:
    if _session is None:
        return 0
    return _session.total_count()


def total_bytes() -> int:
    if _session is None:
        return 0
    return _session.total_bytes()


def clear_all() -> None:
    """lifespan shutdown 调；singleton 未 init 也是 no-op。"""
    if _session is None:
        return
    _session.clear_all()


def configure(
    *,
    max_count: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> None:
    get_session().configure(max_count=max_count, max_bytes=max_bytes)


# ---------------------------------------------------------------------------
# Crypto helpers —— SHAKE-128 keystream + XOR
# ---------------------------------------------------------------------------


def _keystream(key: bytes, nonce: bytes, n: int) -> bytes:
    """SHAKE-128(key || nonce).digest(n) —— 任意长度 keystream。

    SHAKE-128 是 SHA-3 标准的可扩展输出函数（NIST FIPS 202）。给定
    (key, nonce) 组合输出永远确定，作为 XOR 流加密的 keystream 用。
    """
    h = hashlib.shake_128()
    h.update(key)
    h.update(nonce)
    return h.digest(n)


def _xor(a: bytes, b: bytes) -> bytes:
    """等长 XOR。1.5MB 走 int.from_bytes 比 zip 循环快 ~50 倍。"""
    if len(a) != len(b):
        raise ValueError("xor lengths differ")
    if not a:
        return b""
    return (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).to_bytes(len(a), "big")


def _decrypt_and_strip(key: bytes, blob: bytes) -> bytes:
    """反向：blob → 解密 payload → strip snapshot header → return PNG bytes only。"""
    if len(blob) < _NONCE_LEN + _SNAPSHOT_LEN_HEADER:
        raise ValueError("blob too short")
    nonce = blob[:_NONCE_LEN]
    ciphertext = blob[_NONCE_LEN:]
    payload = _xor(ciphertext, _keystream(key, nonce, len(ciphertext)))
    snap_len = int.from_bytes(payload[:_SNAPSHOT_LEN_HEADER], "big")
    if snap_len > len(payload) - _SNAPSHOT_LEN_HEADER:
        raise ValueError("snapshot len out of range")
    return payload[_SNAPSHOT_LEN_HEADER + snap_len:]


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("unlink failed for %s", p, exc_info=True)
