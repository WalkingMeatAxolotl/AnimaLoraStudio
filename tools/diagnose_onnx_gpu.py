"""onnxruntime-gpu 静默降级 CPU 诊断 — 在跑这个 PR 的分支后，
跑一次打标遇到「CUDA EP 静默降级到 CPU」warning 后用本脚本定位根因。

用法（在 studio 同一个 venv 里）：

    python tools/diagnose_onnx_gpu.py

把整段 stdout 贴回 PR 评论 / issue。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def section(title: str) -> None:
    print()
    print("=" * 60)
    print("==", title)
    print("=" * 60)


def main() -> None:
    section("platform")
    print("python:", sys.version.split()[0])
    print("platform:", sys.platform)
    print("executable:", sys.executable)

    section("onnxruntime_setup.current_runtime()")
    try:
        from studio.services.runtime import onnxruntime as o
    except ImportError as exc:
        print("studio import 失败 — 是不是没在 studio 仓库根目录跑？")
        print("原因:", exc)
        return
    rt = o.current_runtime()
    print(json.dumps(rt, indent=2, ensure_ascii=False, default=str))

    section("系统 CUDA 检测（决定 preload 是否被 skip）")
    print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
    print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
    print("/usr/local/cuda/lib64 存在:", os.path.isdir("/usr/local/cuda/lib64"))
    import ctypes.util
    print("ld 路径里 cublas:", ctypes.util.find_library("cublas"))
    print("ld 路径里 cudnn:", ctypes.util.find_library("cudnn"))
    print("_has_system_cuda_libs():", o._has_system_cuda_libs())

    section("torch")
    try:
        import torch
        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device_name:", torch.cuda.get_device_name(0))
            print("cudnn_version:", torch.backends.cudnn.version())
    except Exception as exc:  # noqa: BLE001
        print("torch import 失败:", exc)

    section("nvidia-*-cu12 wheels（torch wheel preload 来源） + onnxruntime")
    out = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        capture_output=True, text=True,
    ).stdout
    keep = ("nvidia", "cudnn", "onnxruntime", "torch")
    for line in out.splitlines():
        low = line.lower()
        if any(k in low for k in keep):
            print(" ", line)

    section("nvidia-smi")
    try:
        nv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        print("rc:", nv.returncode)
        print("stdout:", nv.stdout.strip())
        print("stderr:", nv.stderr.strip())
    except FileNotFoundError:
        print("nvidia-smi 不在 PATH（云机器可能在 docker 里没暴露）")
    except Exception as exc:  # noqa: BLE001
        print("nvidia-smi 跑失败:", exc)

    section("尝试真创 InferenceSession（用 onnxruntime 内置 test model 或现有 wd14 模型）")
    try:
        import onnxruntime as ort
        print("ort.__version__:", ort.__version__)
        print("ort.get_available_providers():", ort.get_available_providers())
        import glob
        candidates = (
            glob.glob("models/wd14/**/model.onnx", recursive=True)
            + glob.glob("models/cltagger/**/*.onnx", recursive=True)
        )
        if not candidates:
            print("没找到本地 onnx 模型，跳过 session 创建测试")
            return
        path = candidates[0]
        print("用模型:", path)
        sess = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        actual = sess.get_providers()
        print("实际 session.get_providers():", actual)
        if "CUDAExecutionProvider" not in actual:
            print(">>> 确认静默降级：请求过 CUDA 但 session 用的是", actual)
        else:
            print(">>> CUDA EP 真生效")
    except Exception as exc:  # noqa: BLE001
        print("session 创建抛异常:", exc)


if __name__ == "__main__":
    main()
