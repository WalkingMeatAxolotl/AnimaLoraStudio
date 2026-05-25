"""PP8 — onnxruntime 运行时检测 / 安装。

由于 onnxruntime（CPU 包）和 onnxruntime-gpu 共享同一个 import 名 `onnxruntime`，
两者**互斥**（同 PyPI 包名），不能同装。requirements.txt 不写死它，由本模块在
启动期检测 GPU 后决定装哪一个，避免用户机器 CUDA 与硬编码包不匹配的踩坑。

主路径：
    bootstrap()   — cli.cmd_run / cmd_dev 启动前调；未装 → install("auto")
    install_runtime(target) — Settings 页「重装为 X」按钮调；同步 pip
    current_runtime()       — Settings 页展示当前状态
    detect_cuda()           — nvidia-smi 探针

约定：
- 「装错了 cuda 版本」体现为 `import onnxruntime` 成功但 `CUDAExecutionProvider`
  不在 providers 里；不自动重装（用户可能故意），UI 给手动按钮 + 警告
- 装包用 `subprocess.run([sys.executable, "-m", "pip", ...])`，不调内部 pip API

PP9.5 — CUDA 共享库预加载 + session 创建 fallback：
- onnxruntime-gpu 不带 CUDA runtime so（libcurand / libcublas / libcudnn ...）。
  Linux 上常见踩坑：`get_available_providers()` 报 CUDA EP 可用，但创 session
  时 dlopen 挂在 `libcurand.so.10: cannot open shared object file`。
- 解法：模块顶层在 import onnxruntime **之前**用 ctypes RTLD_GLOBAL 预加载
  torch 自带的 `nvidia/*/lib/*.so`（PyTorch 默认安装 nvidia-* wheel 到这里）。
  这条路在 ComfyUI / WD14 生态里**没人做**，但是最便宜的通用 fix。
- 失败时 wd14_tagger 仍会捕异常降 CPU；本模块用 record_cuda_load_error 把
  原因 stash 出来给 UI 显示。
"""
from __future__ import annotations

import ctypes
import importlib
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

GPU_PACKAGE = "onnxruntime-gpu"
CPU_PACKAGE = "onnxruntime"
# onnxruntime-gpu 1.19+ PyPI 默认 CUDA 12.x，覆盖 RTX 30/40/50（5090 Blackwell 需 1.20+）
GPU_VERSION_SPEC = ">=1.20"
CPU_VERSION_SPEC = ">=1.16"

# PP9.6 — onnxruntime-gpu wheel 不打包 CUDA runtime so（libcurand / libcublas
# / ...），用户机器没系统装 CUDA 时 dlopen 直接挂。靠 PyPI 上的 nvidia-*-cu12
# wheel 把它们装到 venv 的 site-packages/nvidia/*/lib/，配合本模块顶层的
# RTLD_GLOBAL preload 让 onnxruntime 后续 dlopen 找到符号。
#
# 注意：
# - **不含 nvidia-cudnn-cu12**：torch 的 GPU build 已经把它装上 + 锁死，再
#   `pip install nvidia-cudnn-cu12` 不带版本会被升到最新，破坏 torch；只在
#   它**完全没装**时才补。
# - 这套 wheel 只有 manylinux 平台；Windows / macOS 上不可用 → 安装函数会
#   早返回。Windows 上正确路径是用户系统装 CUDA Toolkit + cuDNN。
_NVIDIA_CUDA_RUNTIME_WHEELS: tuple[str, ...] = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusolver-cu12",
)
_NVIDIA_CUDNN_WHEEL = "nvidia-cudnn-cu12"

# torch 装 GPU build 时拉的 nvidia-* wheel 安装到 site-packages/nvidia/<sub>/lib/
# 下面这张表覆盖 onnxruntime-gpu 1.20 (CUDA 12.x / cuDNN 9.x) 创 session 时
# 真正用到的 so。同 soname 多个候选包按顺序 try（cublasLt 通常在 cublas 包里）。
_TORCH_NVIDIA_LIBS_LINUX: tuple[tuple[str, str], ...] = (
    ("nvidia.cuda_runtime", "libcudart.so.12"),
    ("nvidia.cuda_nvrtc", "libnvrtc.so.12"),
    ("nvidia.cublas", "libcublas.so.12"),
    ("nvidia.cublas", "libcublasLt.so.12"),
    ("nvidia.cufft", "libcufft.so.11"),
    ("nvidia.curand", "libcurand.so.10"),
    ("nvidia.cusparse", "libcusparse.so.12"),
    ("nvidia.cusolver", "libcusolver.so.11"),
    ("nvidia.cudnn", "libcudnn.so.9"),
)


# ---------------------------------------------------------------------------
# CUDA 共享库预加载（PP9.5）
# ---------------------------------------------------------------------------


_PRELOAD_RESULT: Optional[dict[str, Any]] = None
_CUDA_LOAD_ERROR: Optional[str] = None


# onnxruntime-gpu 1.20+ 的 CUDA EP 按 CUDA 12.x 编译，dlopen 时找的是带版本号
# 的 soname。检测系统能否直接 dlopen 这些精确 soname，比 find_library("cublas")
# 严格 —— 后者匹配任意版本 cuBLAS，无法分辨 12 vs 13。
#
# 用户实测踩坑（2026-05-25）：云镜像装的是 CUDA 13（ld 路径里 libcublas.so.13），
# 旧版 find_library 返回 True → 跳过 torch wheel preload → ORT dlopen
# libcublasLt.so.12 失败 → 静默降 CPU。现在两个 soname 都必须能 dlopen 才认。
_ORT_REQUIRED_LINUX_SONAMES: tuple[str, ...] = (
    "libcublasLt.so.12",
    "libcudnn.so.9",
)


def _can_dlopen(soname: str) -> bool:
    # ctypes.RTLD_LAZY 仅 POSIX 有；Windows 上 ctypes 没这常量。本函数只在
    # Linux preload 路径里被实际调用，但跨平台导入 / 单测时不能 AttributeError。
    mode = getattr(ctypes, "RTLD_LAZY", 0)
    try:
        ctypes.CDLL(soname, mode=mode)
        return True
    except OSError:
        return False


def _has_system_cuda_libs() -> bool:
    """系统能否直接 dlopen ORT CUDA EP 需要的精确 soname。

    返回 True → 跳过 PP9.5 preload，让 onnxruntime 用系统的 CUDA so
    （避免 torch wheel cu128 与 ORT 编译目标 cu12.x 子版本错位塞进
    RTLD_GLOBAL 导致 CUBLAS_STATUS_INVALID_VALUE）。返回 False → 走
    preload，torch GPU build 自带的 nvidia-*-cu12 wheels 兜底补齐。

    必须两个关键 so 都能 dlopen 才算系统完整：
    - libcublasLt.so.12：ORT 12.x CUDA EP 直接 dlopen 这个 soname；系统是
      CUDA 13 时只有 .so.13 → 返回 False，触发 preload
    - libcudnn.so.9：cuDNN 跨 CUDA 12/13 同 soname；云镜像装 CUDA Toolkit
      但没装 cuDNN 是非常常见的场景，缺它必须让 torch wheel 补

    副作用：成功的 dlopen 把 so 加载进进程地址空间。这是预期的 ——
    跳过 preload 路径下 ORT 也会立即 dlopen 同一 so（同一 handle 被复用），
    等同于提前热身一次。
    """
    return all(_can_dlopen(s) for s in _ORT_REQUIRED_LINUX_SONAMES)


def _add_torch_dll_dirs_windows() -> dict[str, Any]:
    """PR — Windows 上把 torch 自带的 CUDA DLL 目录加入 Python DLL 搜索路径。

    Python 3.8+ Windows 出于安全废除了 PATH 自动 dlopen native DLL —— 必须
    `os.add_dll_directory()` 主动声明。onnxruntime 在 import 期 dlopen
    `cublasLt64_12.dll` / `cudnn_*.dll` 时找不到，`get_available_providers()`
    照样列 CUDAExecutionProvider，但 InferenceSession 内部 silently 降 CPU
    （onnx_tagger_base._create_session 已经能识别这种降级）。

    torch GPU build wheel 把全套 CUDA DLL 放在 `site-packages/torch/lib/`
    （cublasLt64_12 / cudnn*_9 / curand64_10 / cufft64_11 / cusparse64_12 /
    cudart64_12 / nvrtc）。加进 DLL search path，后续 onnxruntime 的 dlopen
    就能找到。

    返回 `{"added", "errors", "candidates"}`：
    - `added`：成功 add_dll_directory 的目录列表
    - `errors`：尝试但失败的 (dir, reason) 列表
    - `candidates`：发现的候选目录数（=0 表示 venv 没装 torch GPU build）
    """
    added: list[str] = []
    errors: list[tuple[str, str]] = []
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return {"added": added, "errors": errors, "candidates": 0}
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(lib):
        return {"added": added, "errors": errors, "candidates": 0}
    try:
        # Python 3.8+ Windows API；其他平台没有
        os.add_dll_directory(lib)  # type: ignore[attr-defined]
        added.append(lib)
    except (OSError, AttributeError) as exc:
        errors.append((lib, str(exc)))
    return {"added": added, "errors": errors, "candidates": 1}


def _preload_torch_cuda_libs() -> dict[str, Any]:
    """跨平台预加载 torch 自带的 CUDA 库，让 onnxruntime-gpu dlopen 找得到。

    背景：onnxruntime-gpu wheel 不打包 CUDA runtime；用户机器没系统装 CUDA
    时，CUDA EP 在 `get_available_providers()` 里看着可用，但创 session 时
    dlopen 失败（Linux: `libcurand.so.10`；Windows: `cublasLt64_12.dll`）。
    onnxruntime 不抛异常，会**静默降级到 CPU**（onnx_tagger_base 已有检测）。

    PyTorch GPU build 自带所有需要的 CUDA 库：
    - **Linux**：装到 `site-packages/nvidia/*/lib/` —— `ctypes.CDLL(RTLD_GLOBAL)`
      预加载到全局符号表
    - **Windows**：装到 `site-packages/torch/lib/` —— `os.add_dll_directory()`
      加入 DLL 搜索路径（Python 3.8+ 必需）

    **注意**：Linux 上系统已有 CUDA（如 nvidia docker 镜像）时跳过 preload。
    torch wheel 的 CUDA 版本（cu128 = cuBLAS 12.8）与 onnxruntime-gpu 编译目标
    的 CUDA 子版本不一致时，强行 RTLD_GLOBAL 覆盖会导致推理时
    CUBLAS_STATUS_INVALID_VALUE。Windows 上 `add_dll_directory` 只是把目录加进
    搜索路径，不强行覆盖已加载的符号，所以无此问题。

    只对**当前进程**生效；server 子进程必须自己再跑一次（本模块在 import 时
    自动跑）。

    返回 `{"applied", "platform_skip", "system_cuda_skip", "preloaded", "errors", "candidates"}`：
    - `platform_skip=True`：非 Linux / 非 Windows（如 macOS），整体跳过
    - `system_cuda_skip=True`：Linux 系统 CUDA 路径，跳过 preload 让 onnxruntime
      自己 dlopen 系统提供的版本
    - `preloaded`：成功 dlopen / add_dll_directory 的绝对路径列表
    - `errors`：尝试但失败的 (path, reason) 列表
    - `candidates`：检视的候选数（Linux: nvidia.* 子包；Windows: torch/lib 目录）
    """
    if sys.platform == "win32":
        wres = _add_torch_dll_dirs_windows()
        return {
            "applied": True,
            "platform_skip": False,
            "system_cuda_skip": False,
            "preloaded": wres["added"],
            "errors": wres["errors"],
            "candidates": wres["candidates"],
        }
    if not sys.platform.startswith("linux"):
        return {
            "applied": False,
            "platform_skip": True,
            "system_cuda_skip": False,
            "preloaded": [],
            "errors": [],
            "candidates": 0,
        }
    if _has_system_cuda_libs():
        return {
            "applied": False,
            "platform_skip": False,
            "system_cuda_skip": True,
            "preloaded": [],
            "errors": [],
            "candidates": 0,
        }
    preloaded: list[str] = []
    errors: list[tuple[str, str]] = []
    seen: set[str] = set()
    candidates = 0
    for pkg, soname in _TORCH_NVIDIA_LIBS_LINUX:
        if soname in seen:
            continue
        try:
            mod = importlib.import_module(pkg)
        except ImportError:
            continue
        candidates += 1
        for base in getattr(mod, "__path__", []):
            candidate = os.path.join(base, "lib", soname)
            if not os.path.exists(candidate):
                continue
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                errors.append((candidate, str(exc)))
                continue
            preloaded.append(candidate)
            seen.add(soname)
            break
    return {
        "applied": True,
        "platform_skip": False,
        "system_cuda_skip": False,
        "preloaded": preloaded,
        "errors": errors,
        "candidates": candidates,
    }


def _ensure_preload() -> dict[str, Any]:
    """幂等触发预加载；首次调用时跑一次，后续返回 cached 结果。"""
    global _PRELOAD_RESULT
    if _PRELOAD_RESULT is None:
        _PRELOAD_RESULT = _preload_torch_cuda_libs()
        if _PRELOAD_RESULT["preloaded"]:
            logger.info(
                "[onnx_setup] 预加载 torch 自带 CUDA 库 %d 个: %s",
                len(_PRELOAD_RESULT["preloaded"]),
                ", ".join(
                    os.path.basename(p) for p in _PRELOAD_RESULT["preloaded"]
                ),
            )
        elif _PRELOAD_RESULT.get("system_cuda_skip"):
            logger.info(
                "[onnx_setup] 检测到系统 CUDA，跳过 torch wheel preload（避免 cuBLAS 版本错位）"
            )
        elif _PRELOAD_RESULT["applied"] and _PRELOAD_RESULT["candidates"] == 0:
            logger.debug(
                "[onnx_setup] 未发现 torch 自带 CUDA wheel；GPU EP 依赖系统 CUDA"
            )
    return _PRELOAD_RESULT


def record_cuda_load_error(msg: Optional[str]) -> None:
    """wd14_tagger.prepare 创 InferenceSession 失败 → 调本函数 stash 原因。

    None 表示成功 / 清空（成功的 session 创建会清旧错误）。
    """
    global _CUDA_LOAD_ERROR
    _CUDA_LOAD_ERROR = msg


def get_cuda_load_error() -> Optional[str]:
    return _CUDA_LOAD_ERROR


# 模块加载即触发预加载 —— 必须在任何地方 `import onnxruntime` 之前生效。
# server.py 顶层 `from .services import onnxruntime_setup` 已经覆盖 server 子进程；
# cli.py 也在 cmd_run 早期 import 本模块。
_ensure_preload()


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def detect_cuda() -> dict[str, Any]:
    """运行 nvidia-smi 探针。返回 {"available": bool, "driver_version": str|None, "gpu_name": str|None}。

    nvidia-smi 不需要 root，是最低成本的 GPU 检测；找不到 / 跑失败都视作无 GPU。
    """
    nv = shutil.which("nvidia-smi")
    if not nv:
        return {"available": False, "driver_version": None, "gpu_name": None}
    try:
        out = subprocess.run(
            [
                nv,
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("nvidia-smi exec failed: %s", exc)
        return {"available": False, "driver_version": None, "gpu_name": None}
    if out.returncode != 0:
        return {"available": False, "driver_version": None, "gpu_name": None}
    line = (out.stdout or "").strip().splitlines()
    if not line:
        return {"available": False, "driver_version": None, "gpu_name": None}
    parts = [p.strip() for p in line[0].split(",", 1)]
    driver = parts[0] if parts else None
    name = parts[1] if len(parts) > 1 else None
    return {"available": True, "driver_version": driver, "gpu_name": name}


def current_runtime() -> dict[str, Any]:
    """返回当前进程视角的 onnxruntime 信息。

    `installed` 来自 dist-info（pip 视角）；`providers` 是 import 后实际可用 EP（已加载
    的 native 模块视角）。两者**可能不一致** —— 装完包不重启 → dist-info 显示新包，
    providers 仍是旧包的。`restart_required` 表示这种状态。
    """
    installed_pkg, installed_ver = _query_dist_info()
    process_version: Optional[str] = None
    providers: list[str] = []
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
        providers = list(ort.get_available_providers())
        process_version = getattr(ort, "__version__", None)
    except ImportError:
        pass

    # 检测「pip 装的包名/版本」 vs 「进程里 import 的版本」不一致
    # （onnxruntime 是 C extension，pip 重装不会热替换已 import 的 .pyd）
    restart_required = False
    if installed_pkg is not None and process_version is not None:
        # 版本号不一致直接判定 stale
        if installed_ver != process_version:
            restart_required = True
        # 包名: GPU 包应该有 CUDA EP，CPU 包不会有
        elif installed_pkg == GPU_PACKAGE and "CUDAExecutionProvider" not in providers:
            restart_required = True

    return {
        "installed": installed_pkg,
        "version": installed_ver or process_version,
        "providers": providers,
        "cuda_available": "CUDAExecutionProvider" in providers,
        "restart_required": restart_required,
        # PP9.5 — 创 InferenceSession 时实际 dlopen 报的错（如 `libcurand.so.10`
        # 缺失）；wd14_tagger.prepare 降 CPU 后填进来。None=没碰过 / 上次成功。
        "cuda_load_error": _CUDA_LOAD_ERROR,
        # PP9.5 — torch 自带 CUDA so 预加载结果（Linux only）；UI 诊断用
        "preload": _PRELOAD_RESULT,
    }


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


_PIP_FALLBACK_MIRROR = "https://mirrors.cloud.tencent.com/pypi/simple/"


def _pip(args: list[str], *, mirror: str = "") -> tuple[int, str]:
    """跑 `<sys.executable> -m pip <args>`；返回 (rc, combined_output)。

    mirror 非空时追加 `-i {mirror}`（用于镜像 fallback 重试）。
    """
    cmd = [sys.executable, "-m", "pip", *args]
    if mirror:
        cmd += ["-i", mirror]
    logger.info("[onnx_setup] %s", " ".join(cmd))
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # pip install 几分钟级别
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return 1, f"pip 超时（10 分钟）: {exc}"
    except Exception as exc:  # noqa: BLE001
        return 1, f"pip 调用失败: {exc}"
    text = (out.stdout or "") + (out.stderr or "")
    return out.returncode, text


def _decide_target(target: str) -> str:
    """auto/gpu/cpu → 实际包名（带版本约束）。"""
    if target == "gpu":
        return f"{GPU_PACKAGE}{GPU_VERSION_SPEC}"
    if target == "cpu":
        return f"{CPU_PACKAGE}{CPU_VERSION_SPEC}"
    if target == "auto":
        cuda = detect_cuda()
        if cuda["available"]:
            return f"{GPU_PACKAGE}{GPU_VERSION_SPEC}"
        return f"{CPU_PACKAGE}{CPU_VERSION_SPEC}"
    raise ValueError(f"非法 target: {target!r}（应为 auto/gpu/cpu）")


def _is_dist_installed(pkg: str) -> bool:
    """dist-info 里有没有这个包；不 import，避免触发 native 模块加载。"""
    try:
        from importlib.metadata import PackageNotFoundError, version as _ver
        try:
            _ver(pkg)
            return True
        except PackageNotFoundError:
            return False
    except Exception:  # noqa: BLE001
        return False


def _install_cuda_runtime_wheels() -> dict[str, Any]:
    """PP9.6 — Linux 上把 onnxruntime-gpu 跑起来需要的 CUDA runtime wheels 装上。

    返回 `{"installed": [...新装的], "skipped": [...原本就有的], "platform_skip": bool, "stdout": str}`。
    失败抛 RuntimeError，并在抛之前**回滚本次刚装的包**（保持 venv 不被污染）。

    cuDNN 单独处理：原本就有就不动（避免撞 torch 锁的版本）；没有才补。
    """
    if not sys.platform.startswith("linux"):
        # Windows / macOS：nvidia-*-cu12 wheel 不可用；用户应靠系统 CUDA Toolkit
        return {
            "installed": [],
            "skipped": [],
            "platform_skip": True,
            "stdout": "non-linux platform; skip nvidia-*-cu12 wheels",
        }
    targets: list[str] = []
    skipped: list[str] = []
    # cuDNN：只在缺时装（torch GPU build 通常已带）
    if _is_dist_installed(_NVIDIA_CUDNN_WHEEL):
        skipped.append(_NVIDIA_CUDNN_WHEEL)
    else:
        targets.append(_NVIDIA_CUDNN_WHEEL)
    # 其余 6 个：缺啥装啥
    for pkg in _NVIDIA_CUDA_RUNTIME_WHEELS:
        if _is_dist_installed(pkg):
            skipped.append(pkg)
        else:
            targets.append(pkg)
    if not targets:
        return {
            "installed": [],
            "skipped": skipped,
            "platform_skip": False,
            "stdout": "all CUDA runtime wheels already present",
        }
    rc, out = _pip(["install", *targets])
    if rc != 0:
        logger.warning("[onnx_setup] CUDA wheels pip 官方源失败，切换腾讯镜像重试...")
        rc, out = _pip(["install", *targets], mirror=_PIP_FALLBACK_MIRROR)
    if rc != 0:
        # 回滚：把本次想装的从 venv 里再卸掉，保持装包前的状态
        # （pip install 失败时部分包可能已装；不区分，统一卸）
        rb_rc, rb_out = _pip(["uninstall", "-y", *targets])
        raise RuntimeError(
            f"安装 CUDA runtime wheels 失败（rc={rc}）:\n{out}\n"
            f"--- rollback (rc={rb_rc}) ---\n{rb_out}"
        )
    return {
        "installed": targets,
        "skipped": skipped,
        "platform_skip": False,
        "stdout": out,
    }


def install_runtime(target: str = "auto") -> dict[str, Any]:
    """先 uninstall 两个互斥包再装目标。

    target: "auto" | "gpu" | "cpu"
    返回 `{"target", "installed_pkg", "installed_version", "restart_required": True,
           "stdout", "cuda_runtime"}`，最后一个字段是 PP9.6 装的 nvidia-*-cu12 报告
    （仅 GPU 路径；CPU 路径为 None）。失败抛 RuntimeError。

    **重要**：onnxruntime 是 C extension，pip 卸装重装后**当前进程**里已 import 的
    .pyd/.so 不会被热替换 —— 必须重启 Studio 才能切换 EP。所以本函数不再尝试 reload；
    返回 `restart_required=True` 让 UI 提示用户重启。
    """
    spec = _decide_target(target)
    rc1, log1 = _pip(["uninstall", "-y", GPU_PACKAGE, CPU_PACKAGE])
    rc2, log2 = _pip(["install", "--upgrade", spec])
    if rc2 != 0:
        logger.warning("[onnx_setup] pip 官方源失败，切换腾讯镜像重试...")
        rc2, log2 = _pip(["install", "--upgrade", spec], mirror=_PIP_FALLBACK_MIRROR)
    if rc2 != 0:
        raise RuntimeError(f"安装 {spec} 失败（rc={rc2}）:\n{log2}")

    # PP9.6 — GPU 路径补齐 CUDA runtime wheels（onnxruntime-gpu 不打包它们）。
    # CPU 路径或 auto 检测为 CPU 时跳过。
    cuda_runtime: Optional[dict[str, Any]] = None
    if GPU_PACKAGE in spec:
        try:
            cuda_runtime = _install_cuda_runtime_wheels()
        except RuntimeError as exc:
            # CUDA wheels 装失败不致命：onnxruntime-gpu 已装上，让用户去 Settings 页
            # 看到 cuda_load_error + 手动修。日志记下原因，UI 也能拿到。
            logger.error("[onnx_setup] CUDA runtime wheels 装失败: %s", exc)
            cuda_runtime = {
                "installed": [],
                "skipped": [],
                "platform_skip": False,
                "stdout": str(exc),
                "error": str(exc),
            }

    # 直接读 dist-info 拿新装的版本（不 import；进程里仍是旧的 native 模块）
    new_pkg, new_ver = _query_dist_info()
    return {
        "target": spec,
        "installed_pkg": new_pkg,
        "installed_version": new_ver,
        "restart_required": True,
        "stdout": log1 + log2,
        "cuda_runtime": cuda_runtime,
    }


def _query_dist_info() -> tuple[Optional[str], Optional[str]]:
    """从 dist-info 读两个互斥包的安装状态。返回 (pkg_name, version)。"""
    try:
        from importlib.metadata import PackageNotFoundError, version as _ver
        for pkg in (GPU_PACKAGE, CPU_PACKAGE):
            try:
                return pkg, _ver(pkg)
            except PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return None, None


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def bootstrap() -> dict[str, Any]:
    """启动期一次性检查：

    - 未装 → 自动 install_runtime("auto")
    - 装了但 EP 不匹配机器（有 GPU 但只有 CPU EP）→ 仅 log warn，不动
    - 装了且 EP 匹配 → 静默

    始终返回 current_runtime()（含 detect_cuda 信息），失败不抛出（仅 log）。
    """
    cuda = detect_cuda()
    rt = current_runtime()
    state = {**rt, "cuda_detect": cuda}

    if rt["installed"] is None:
        target = "gpu" if cuda["available"] else "cpu"
        logger.info(
            "[onnx_setup] onnxruntime 未安装，按检测自动装 (target=%s, gpu=%s, driver=%s)",
            target,
            cuda["available"],
            cuda.get("driver_version"),
        )
        try:
            install_runtime(target)
            # bootstrap 在 cli.py 起 server 子进程前跑；server 是新进程会 fresh import → 装完直接重读
            state.update(current_runtime())
        except RuntimeError as exc:
            logger.error("[onnx_setup] 自动安装失败: %s", exc)
            state["error"] = str(exc)
        return state

    # 已装 - 检查 GPU/EP 是否匹配
    if cuda["available"] and not rt["cuda_available"]:
        logger.warning(
            "[onnx_setup] 检测到 NVIDIA GPU 但 onnxruntime 只有 CPU EP "
            "(installed=%s, providers=%s)。WD14 打标会跑在 CPU 上（很慢）。"
            "可在 Settings → WD14 点「重装为 GPU 版」。",
            rt["installed"],
            rt["providers"],
        )
    return state
