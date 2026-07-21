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


#: RAM 预算基底：进程/torch 运行余量（真机实测进程基底 ~4GB + 换页安全边际）
_RAM_BASE_BYTES = 4 * 1024**3
#: VRAM 预算基底：CUDA context + 激活余量
_VRAM_BASE_BYTES = 3 * 1024**3


def _file_bytes(paths) -> int:
    """将读取的权重文件总大小（预算依据）；不存在的路径忽略。"""
    import os

    total = 0
    for p in paths or ():
        try:
            path = str(p)
            if os.path.isdir(path):
                for entry in os.scandir(path):
                    if entry.is_file():
                        total += entry.stat().st_size
            elif path:
                total += os.stat(path).st_size
        except OSError:
            continue
    return total


def check_load_budget(enabled: bool, *, weight_paths, stage: str) -> None:
    """双预算水位护栏：按即将加载的文件实际大小预算 RAM 与 VRAM。

    静态阈值只回答「现在还好吗」；预算制回答「做完这件事之后还好吗」：
    - RAM 需求 ≈ 文件总大小（mmap 读文件的瞬时峰值，真机实测 ≈1:1）+ 基底
    - VRAM 需求 ≈ 文件总大小（权重上卡）+ 基底。GPU free 检查天然拦住
      多进程叠加（另一个 daemon 驻留模型时第二个加载入口 fail-fast）
    ``enabled`` 来自用户配置（Settings → 显存策略）；查询失败静默放行。
    """
    if not enabled:
        return
    need = _file_bytes(weight_paths)
    if need <= 0:
        return

    avail = available_ram_bytes()
    if avail is not None and avail < need + _RAM_BASE_BYTES:
        raise RuntimeError(
            f"系统可用内存不足（{avail / 1024**3:.1f}GB，本次{stage}约需 "
            f"{(need + _RAM_BASE_BYTES) / 1024**3:.1f}GB：权重文件 "
            f"{need / 1024**3:.1f}GB + 运行余量），已中止以避免整机换页"
            f"卡死。请关闭其他占用内存的应用后重试；如需强制继续，可在 "
            f"设置 → 显存策略 关闭内存水位保护。"
        )

    free = gpu_free_bytes_global()
    if free is not None and free < need + _VRAM_BASE_BYTES:
        raise RuntimeError(
            f"GPU 空闲显存不足（{free / 1024**3:.1f}GB，本次{stage}"
            f"约需 {(need + _VRAM_BASE_BYTES) / 1024**3:.1f}GB）。"
            f"可能有其他进程占用显存（另一个出图/训练任务？）——"
            f"请先释放后重试；如需强制继续，可在 设置 → 显存策略 "
            f"关闭内存水位保护。"
        )


#: pinned（页锁定）内存的安全上限占可用物理内存的比例。
#: 页锁定内存**不可换页、trim_working_set 对它无效**，占满会拖垮整机（与
#: mmap working set 卡死同源但更硬），所以留足余量。
_PINNED_SAFE_FRACTION = 0.6


def check_pinned_budget(need_bytes: int, *, blocks: int) -> None:
    """block swap 的 pinned 内存预算护栏（docs/design/block-swap.md §3.2 ①）。

    **不能复用 check_load_budget**：那套按「文件大小 ≈ mmap 瞬时峰值」预算，
    假设内存可回收；pinned 是**永久锁定**、``trim_working_set`` 对它无效，
    同样字节数的危害等级不同。

    只在训练启动期调用一次（B6：失败即报错、不静默降级；分配失败只可能发生
    在启动那一刻，见 §8.1）。查询失败静默放行，与既有护栏口径一致。
    """
    if need_bytes <= 0:
        return
    avail = available_ram_bytes()
    if avail is None:
        return
    safe = int(avail * _PINNED_SAFE_FRACTION)
    if need_bytes > safe:
        raise RuntimeError(
            f"内存不足以换出 {blocks} 层：需锁定 {need_bytes / 1024**3:.1f}GB，"
            f"当前可用 {avail / 1024**3:.1f}GB（安全上限 {safe / 1024**3:.1f}GB）。"
            f"换出的层权重会**锁定**在内存里不可换页，占满会拖慢整机。"
            f"请调小 blocks_to_swap，或关闭其他占用内存的应用后重试。"
        )


def gpu_free_bytes_global() -> int | None:
    """全卡真实空闲显存；查询失败返回 None。

    必须走 NVML：WDDM 下 ``cudaMemGetInfo`` 是**每进程虚拟化视角**，
    看不到其他进程的占用（真机实测：他进程持有 20GB 时它仍报全量
    free）——用它做跨进程护栏形同虚设。NVML 是全卡视角。
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0])
    except Exception:
        pass
    return None
