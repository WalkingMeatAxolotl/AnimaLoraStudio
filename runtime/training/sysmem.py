"""系统内存工具（Windows 为主）：working set trim + 可用内存查询。

背景（真机卡死案例）：safetensors 用 mmap 流式读权重文件，读过的文件页
会驻留进程 working set——13GB DiT + 5GB TE 加载完后有 ~18GB「可回收但
赖着不走」的文件缓存页。物理内存紧张的机器上，这些页与其他应用争内存
触发换页风暴，表现为整机卡死（显存全程健康）。

- ``trim_working_set``：把 working set 页挤到 standby list（系统需要时
  秒级回收、无换页 IO）；被挤出的热页由 soft fault 拉回，代价微秒级。
- ``available_ram_bytes``：ctypes 直读 GlobalMemoryStatusEx，零依赖。
"""

from __future__ import annotations

import logging
import sys


logger = logging.getLogger(__name__)


def trim_working_set() -> bool:
    """把进程 working set 里的可回收页（mmap 文件缓存等）挤回系统。

    大权重加载完成后调用；非 Windows / 失败时 no-op 返回 False。
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = bool(ctypes.windll.psapi.EmptyWorkingSet(handle))
        if ok:
            logger.info("working set 已 trim（mmap 文件缓存页归还系统）")
        return ok
    except Exception:
        return False


def available_ram_bytes() -> int | None:
    """当前系统可用物理内存（字节）；查询失败返回 None。"""
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
            return None
        except Exception:
            return None
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


#: RAM 水位护栏阈值：加载大模型前系统可用内存低于此值时中止（粗估口径，
#: 不按文件大小精确预算——mmap 页可边读边回收，卡死风险来自 avail 逼近
#: 耗尽后的换页风暴，6GB 余量足以让 OS 从 standby 平滑回收）。
RAM_GUARD_MIN_BYTES = 6 * 1024**3


def check_ram_guard(enabled: bool, *, stage: str) -> None:
    """RAM 水位护栏：可用内存不足时 raise 可操作错误（好过整机卡死）。

    ``enabled`` 来自用户配置（Settings → 显存策略）；查询失败静默放行。
    """
    if not enabled:
        return
    avail = available_ram_bytes()
    if avail is None:
        return
    if avail < RAM_GUARD_MIN_BYTES:
        raise RuntimeError(
            f"系统可用内存不足（{avail / 1024**3:.1f}GB < "
            f"{RAM_GUARD_MIN_BYTES / 1024**3:.0f}GB），已中止{stage}以避免"
            f"整机换页卡死。请关闭其他占用内存的应用后重试；"
            f"如需强制继续，可在 设置 → 显存策略 关闭内存水位保护。"
        )
