"""系统资源采集 (CPU / RAM / GPU / VRAM)。

供 topbar 实时小组件按 2-3s 轮询使用。

设计：
    - pynvml 懒初始化一次；失败 (无 NVIDIA / 驱动缺失 / 库未装) 永久标记，
      之后所有调用直接返回 gpu=None — 不重试、不刷日志。
    - psutil 几乎不会失败；仍 try/except 兜底，让前端轮询不会因偶发问题挂掉。
    - 模块无状态导出，调用 collect_stats() 即可。
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import psutil

logger = logging.getLogger(__name__)

# psutil.cpu_percent(interval=None) 第一次调用返回 0.0 (无 baseline)，
# 之后返回「距上次调用以来」的平均占用。模块导入时 prime 一下，让首请求
# 就能拿到从启动到首请求的平均值，避免前端首次轮询永远显示 0%。
psutil.cpu_percent(interval=None)


# ── NVML 懒初始化 ─────────────────────────────────────────────────────
_nvml_lock = threading.Lock()
_nvml_state: dict[str, Any] = {"inited": False, "ok": False}


def _ensure_nvml() -> bool:
    with _nvml_lock:
        if _nvml_state["inited"]:
            return _nvml_state["ok"]
        _nvml_state["inited"] = True
        try:
            import pynvml  # type: ignore[import-untyped]
            pynvml.nvmlInit()
            _nvml_state["ok"] = True
        except Exception as e:
            _nvml_state["ok"] = False
            logger.info("pynvml unavailable; GPU stats disabled (%s)", e)
        return _nvml_state["ok"]


# ── active GPU（torch 实际在用的卡）解析 ─────────────────────────────
# 多卡机器上 torch（默认 FASTEST_FIRST，快卡在前）与 NVML/nvidia-smi
# （PCI 插槽顺序）是**两套编号**，gpu[0] 不一定是训练/出图在用的卡
# （#491：console 报 3070、topbar 显示 2080）。active 判定三级：
#   1. 单卡 → 就是它（零成本，绝大多数用户）；
#   2. 选卡 env 已注入（CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=n，
#      PCI 序与 NVML 同构）→ NVML index n（零成本）；
#   3. 多卡且无 env → 问 torch 当前设备的 PCI bus id，与 NVML 各卡比对。
#      懒解析 + 永久缓存（含失败）：import torch + CUDA init 有一次性
#      开销，不能让 2.5s 的采样 tick 反复付。
_torch_pci_lock = threading.Lock()
_torch_pci_state: dict[str, Any] = {"resolved": False, "bus_id": None}


def _torch_pci_bus_id() -> Optional[str]:
    """torch 当前 CUDA 设备的 PCI bus id（NVML busId 格式）；失败 None。"""
    with _torch_pci_lock:
        if _torch_pci_state["resolved"]:
            return _torch_pci_state["bus_id"]
        _torch_pci_state["resolved"] = True
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(
                    torch.cuda.current_device())
                domain = getattr(props, "pci_domain_id", None)
                bus = getattr(props, "pci_bus_id", None)
                device = getattr(props, "pci_device_id", None)
                if None not in (domain, bus, device):
                    _torch_pci_state["bus_id"] = (
                        f"{domain:08X}:{bus:02X}:{device:02X}.0"
                    )
        except Exception:  # noqa: BLE001
            logger.info("torch PCI bus id 查询失败；GPU active 标记停用")
        return _torch_pci_state["bus_id"]


def _env_selected_index() -> Optional[int]:
    """选卡设置注入的 env → NVML index；非注入形态（缺失/UUID/列表）→ None。"""
    import os

    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        return None
    raw = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    return int(raw) if raw.isdigit() else None


def _resolve_active_index(
    pynvml: Any, handles: list[Any],
) -> Optional[int]:
    if len(handles) == 1:
        return 0
    env_idx = _env_selected_index()
    if env_idx is not None:
        return env_idx if 0 <= env_idx < len(handles) else None
    bus_id = _torch_pci_bus_id()
    if bus_id is None:
        return None
    for i, h in enumerate(handles):
        try:
            nvml_bus = pynvml.nvmlDeviceGetPciInfo(h).busId
            if isinstance(nvml_bus, bytes):
                nvml_bus = nvml_bus.decode(errors="replace")
            if nvml_bus.upper() == bus_id.upper():
                return i
        except Exception:  # noqa: BLE001
            continue
    return None


# ── 数据结构 ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GpuStats:
    index: int
    name: str
    util_pct: int
    vram_used_gb: float
    vram_total_gb: float
    temp_c: Optional[int] = None
    #: torch 实际在用的卡（多卡机器前端显示这张，而不是盲选 gpu[0]）。
    #: 解析不出（CPU-only torch / PCI 匹配失败）时全 False，前端回退 gpu[0]。
    active: bool = False


@dataclass(frozen=True)
class SystemStats:
    cpu_pct: float
    ram_used_gb: float
    ram_total_gb: float
    # None = NVML 不可用；[] = NVML 可用但 0 卡 (前端两种都隐藏 GPU pill)
    gpu: Optional[list[GpuStats]]


# ── 采集 ─────────────────────────────────────────────────────────────
def _bytes_to_gb(n: int) -> float:
    return round(n / (1024 ** 3), 2)


def _collect_gpu() -> Optional[list[GpuStats]]:
    if not _ensure_nvml():
        return None
    try:
        import pynvml  # type: ignore[import-untyped]
        count = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        active_idx = _resolve_active_index(pynvml, handles) if count else None
        out: list[GpuStats] = []
        for i, h in enumerate(handles):
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            out.append(GpuStats(
                index=i,
                name=name,
                util_pct=int(util.gpu),
                vram_used_gb=_bytes_to_gb(mem.used),
                vram_total_gb=_bytes_to_gb(mem.total),
                temp_c=int(temp) if temp is not None else None,
                active=(i == active_idx),
            ))
        return out
    except Exception:
        logger.exception("gpu stats collection failed")
        return None


def collect_stats() -> SystemStats:
    try:
        # interval=None: 返回自上次调用以来的 CPU 占用；首次调用返回 0.0，
        # 后续轮询拿到的就是 2-3s 平均值，对实时监控刚好。
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used = _bytes_to_gb(mem.total - mem.available)
        ram_total = _bytes_to_gb(mem.total)
    except Exception:
        logger.exception("psutil stats collection failed")
        cpu = 0.0
        ram_used = 0.0
        ram_total = 0.0
    return SystemStats(
        cpu_pct=round(float(cpu), 1),
        ram_used_gb=ram_used,
        ram_total_gb=ram_total,
        gpu=_collect_gpu(),
    )


def stats_to_json(s: SystemStats) -> dict[str, Any]:
    return {
        "cpu_pct": s.cpu_pct,
        "ram_used_gb": s.ram_used_gb,
        "ram_total_gb": s.ram_total_gb,
        "gpu": [asdict(g) for g in s.gpu] if s.gpu is not None else None,
    }


# ── SSE sampler ──────────────────────────────────────────────────────
class SystemStatsSampler:
    """后台线程：周期性采集系统资源 → callback (通常是 bus.publish)。

    取代每个客户端独立轮询 /api/system/stats — 云部署场景下避免污染
    server access log、DevTools Network 面板、跨公网 RTT 开销。前端只在
    mount 时 GET 一次冷启动，之后走 SSE 持续接收。
    """

    def __init__(
        self,
        on_sample: Callable[[dict[str, Any]], None],
        *,
        interval: float = 2.5,
    ) -> None:
        self._on_sample = on_sample
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(
            target=self._run, name="system-stats-sampler", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload = stats_to_json(collect_stats())
                self._on_sample(payload)
            except Exception:
                logger.exception("system stats sampler tick failed")
            self._stop.wait(self._interval)
