"""计算显卡选择（多卡机器，issue #491）。

用户在 Settings → PyTorch 里选定计算用哪张卡（存
``secrets.system.gpu_index``，NVML/nvidia-smi 的 PCI 序号）。本模块在
**启动早期、torch 首次碰 CUDA 之前**把选择翻译成两个环境变量：

- ``CUDA_DEVICE_ORDER=PCI_BUS_ID``：逼 CUDA 放弃默认的 FASTEST_FIRST
  （快卡优先）排序，改用与 NVML/nvidia-smi 相同的 PCI 插槽序——设置里
  存的序号、topbar 监控的序号、CUDA 实际选中的卡才是同一个意思。
- ``CUDA_VISIBLE_DEVICES=<n>``：只露出选中的卡。

launcher（cli.py）与 server（studio.server）各调一次：launcher 注入后
子进程（server → supervisor spawn 的训练/正则 AI/出图 daemon/worker）
全部继承，无需逐 spawn 处理；server 侧再调兜底「绕过 launcher 直跑
server」的场景。幂等，重复调用安全。

用户在 studio.bat 等处**手动**设过这两个变量 → 尊重不动。区分「我们
注入的」与「用户手设的」靠 marker 环境变量：launcher 是常驻进程，用户
改设置后触发的重启仍在同一 launcher 进程里跑本函数——上一轮注入的值
必须能被新设置覆盖或撤销，而用户手设的值永远不能动。
"""
from __future__ import annotations

import logging
import os
from typing import MutableMapping, Optional

logger = logging.getLogger(__name__)

#: 「这两个 CUDA 变量是本模块注入的」标记；随 env 传染给子进程无副作用。
_MARKER = "LORA_GPU_SELECT_APPLIED"


def _selected_index() -> Optional[int]:
    """secrets.system.gpu_index；未设置/读取异常 → None（不注入）。"""
    try:
        from studio.infrastructure import secrets

        idx = secrets.load().system.gpu_index
        if idx is None or int(idx) < 0:
            return None
        return int(idx)
    except Exception:  # noqa: BLE001
        logger.warning(
            "read the compute GPU setting failed; using the CUDA default device",
            exc_info=True,
        )
        return None


def _nvml_device_count() -> Optional[int]:
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        return None


def _revoke(environ: MutableMapping[str, str]) -> None:
    environ.pop("CUDA_VISIBLE_DEVICES", None)
    environ.pop("CUDA_DEVICE_ORDER", None)
    environ.pop(_MARKER, None)


def apply_gpu_selection_env(
    environ: Optional[MutableMapping[str, str]] = None,
) -> None:
    """按 secrets.system.gpu_index 注入/撤销 CUDA 选卡 env。

    必须在 torch 首次初始化 CUDA 之前调用——CUDA 只在 init 时读一次这
    两个变量，之后改无效。失败只记日志不抛：选卡打不上顶多回 CUDA
    默认行为，不能挡启动。
    """
    env = os.environ if environ is None else environ
    applied_by_us = env.get(_MARKER) == "1"
    if not applied_by_us and (
        "CUDA_VISIBLE_DEVICES" in env or "CUDA_DEVICE_ORDER" in env
    ):
        return  # 用户手动控制 CUDA env：尊重，永不覆盖
    idx = _selected_index()
    if idx is None:
        # 设置清回默认（或从未设置）：撤销上一轮注入。launcher 常驻，
        # 不撤销的话「改回自动」重启后永远不生效。
        if applied_by_us:
            _revoke(env)
        return
    count = _nvml_device_count()
    if count is not None and idx >= count:
        # 卡不在了（eGPU 拔线 / 换硬件）：忽略选择回 CUDA 默认——注入
        # 一个不存在的序号会让 torch 面对空设备列表，训练/出图全瘫。
        logger.warning(
            "compute GPU setting gpu_index=%d is out of range (device_count=%d); "
            "ignored, using the CUDA default device", idx, count,
        )
        if applied_by_us:
            _revoke(env)
        return
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(idx)
    env[_MARKER] = "1"
    logger.debug(
        "compute GPU pinned: pci_index=%d (CUDA_DEVICE_ORDER=PCI_BUS_ID)", idx,
    )
