"""xformers 安装服务（简化版，类比 flash_attention_setup）。

xformers 与 flash_attn 同为 attention 加速 C extension，但安装路径**显著简单**：
  - flash_attn：依赖 dao-AILab + mjun0812 prebuild 的 GitHub Releases，
    每个 torch+cuda+python 组合一个 wheel，需要解析 wheel 名 + 评分匹配
  - xformers：facebook 官方 PyPI 直接发 wheel，与 torch+cuda 强绑定但
    PyTorch 官方 wheel index (download.pytorch.org/whl/cuXXX) 已经
    把对应 cu_tag 的 wheel 集中起来。

所以本服务只暴露：
  - current_status() → {installed, version}
  - install() → pip install xformers --index-url <torch-cuda-index>

不复刻 flash_attention_setup 的 GitHub Releases 解析 / 候选列表 UI。
装失败时把 stderr 透传，让用户自己看（多数失败 = 上游没出对应 torch+cu
组合的 wheel，需要换 torch 版本或等上游覆盖）。
"""
from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from typing import Any, Optional


def current_status() -> dict[str, Any]:
    """xformers 当前安装状态：{installed: bool, version: str|None}。"""
    try:
        version = importlib.metadata.version("xformers")
        return {"installed": True, "version": version}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def detect_attention_backend() -> str:
    """根据当前装了什么决定 attention backend。
    优先级 flash_attn > xformers > none（PyTorch SDPA）。
    给 secrets.generate.attention_backend='auto' 时用。
    """
    try:
        importlib.metadata.version("flash_attn")
        return "flash_attn"
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        importlib.metadata.version("xformers")
        return "xformers"
    except importlib.metadata.PackageNotFoundError:
        pass
    return "none"


def disable_triton_probe(env: dict[str, str]) -> None:
    """替 xformers 短路 triton 探测（子进程 env 注入用）。

    xformers 启用后其 `_is_triton_available()` 会 `import triton`。triton 官方
    不发 Windows wheel，未安装时必然 ImportError，xformers 用
    `logger.warning(..., exc_info=True)` 把完整 traceback 打进 task log ——
    还会被失败摘要 `_tail_log_for_error_msg`（取最后一处 Traceback）误当
    失败原因展示给用户。

    无条件短路：xformers 里 triton 只服务 LLM 型 kernel（fmha triton_splitk /
    rmsnorm / rope_padded / tiled_matmul），本 app 的 memory_efficient_attention
    （含 NaViT varlen）走 cutlass/flash C++ kernel，triton 装没装、好没好都
    零参与，探测结果与 warning 对用户均无价值。torch.compile 的 triton 使用
    不读本变量，不受影响。`XFORMERS_FORCE_DISABLE_TRITON=1` 在 xformers 源码
    里的检查位于 `import triton` 之前，设了就完全跳过探测；setdefault 保证
    用户显式设过的值优先，且 xformers 侧 `XFORMERS_ENABLE_TRITON=1` 优先级
    更高，仍是强开逃生口。
    """
    env.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")


def _torch_cuda_index() -> Optional[str]:
    """从 `torch.__version__` 的 `+cuXXX` 后缀推 PyTorch CUDA index URL。

    xformers wheel 与 torch ABI 强绑定（每个 xformers 版本锁定特定 torch+cuda），
    必须装与当前 torch 同 CUDA 的 wheel。PyTorch 官方 index 按 cu_tag 分组：
        https://download.pytorch.org/whl/cu128
        https://download.pytorch.org/whl/cu130
        ...

    ABI 检测原则与 flash_attention_setup.detect_env() 一致：从 torch 拿，
    不从 nvidia-smi 拿（nvidia-smi 是 driver 支持的 CUDA，不是 PyTorch 编译的）。
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    m = re.search(r"\+cu(\d+)", torch.__version__)
    if m:
        return f"https://download.pytorch.org/whl/cu{m.group(1)}"
    return None


def install() -> dict[str, Any]:
    """pip install xformers，自动按当前 torch 的 CUDA index 选 wheel。

    返回 {installed, version, stdout_tail, restart_required}。
    安装失败抛 RuntimeError，message 含 stderr 末尾（多数 wheel 找不到时
    pip 会打印「No matching distribution found for xformers」）。

    `restart_required=True` 因为 xformers 是 C extension —— 装好后必须重启
    Studio 进程才能 import（与 flash_attn 同）。
    """
    cmd = [sys.executable, "-m", "pip", "install", "xformers"]
    index = _torch_cuda_index()
    if index:
        cmd += ["--index-url", index]

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("pip install xformers 超时（10 分钟）") from exc

    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-1500:]
        raise RuntimeError(
            f"pip install xformers 失败 (exit {r.returncode}):\n{tail}"
        )

    status = current_status()
    return {
        **status,
        "stdout_tail": (r.stdout or "")[-1500:],
        "restart_required": True,
    }
