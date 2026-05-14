"""PP8 — onnxruntime 启动期检测 / 装包逻辑（mock subprocess）。

不真跑 pip / 真启 nvidia-smi；用 monkeypatch 替 subprocess.run + shutil.which
覆盖装包决策表。
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from studio.services import onnxruntime_setup as ors


# ---------------------------------------------------------------------------
# detect_cuda
# ---------------------------------------------------------------------------


def test_detect_cuda_no_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ors.shutil, "which", lambda _: None)
    res = ors.detect_cuda()
    assert res == {"available": False, "driver_version": None, "gpu_name": None}


def test_detect_cuda_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ors.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    fake = MagicMock(returncode=0, stdout="551.86, NVIDIA GeForce RTX 5090\n", stderr="")
    monkeypatch.setattr(ors.subprocess, "run", lambda *a, **k: fake)
    res = ors.detect_cuda()
    assert res == {
        "available": True,
        "driver_version": "551.86",
        "gpu_name": "NVIDIA GeForce RTX 5090",
    }


def test_detect_cuda_returncode_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ors.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    fake = MagicMock(returncode=9, stdout="", stderr="error")
    monkeypatch.setattr(ors.subprocess, "run", lambda *a, **k: fake)
    res = ors.detect_cuda()
    assert res["available"] is False


def test_detect_cuda_subprocess_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ors.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def _raise(*_a, **_k):
        raise OSError("permission denied")

    monkeypatch.setattr(ors.subprocess, "run", _raise)
    res = ors.detect_cuda()
    assert res["available"] is False


# ---------------------------------------------------------------------------
# _decide_target
# ---------------------------------------------------------------------------


def test_decide_target_explicit() -> None:
    assert ors._decide_target("gpu").startswith("onnxruntime-gpu")
    assert ors._decide_target("cpu").startswith("onnxruntime")
    assert "gpu" not in ors._decide_target("cpu")


def test_decide_target_auto_with_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": True, "driver_version": "551.86", "gpu_name": "RTX 5090"},
    )
    assert ors._decide_target("auto").startswith("onnxruntime-gpu")


def test_decide_target_auto_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": False, "driver_version": None, "gpu_name": None},
    )
    res = ors._decide_target("auto")
    assert res.startswith("onnxruntime")
    assert "gpu" not in res


def test_decide_target_invalid() -> None:
    with pytest.raises(ValueError):
        ors._decide_target("xpu")


# ---------------------------------------------------------------------------
# install_runtime — mock pip
# ---------------------------------------------------------------------------


def test_install_runtime_runs_uninstall_then_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_pip(args):
        calls.append(args)
        return 0, "ok"

    monkeypatch.setattr(ors, "_pip", fake_pip)
    monkeypatch.setattr(
        ors, "_query_dist_info",
        lambda: ("onnxruntime-gpu", "1.20.0"),
    )
    res = ors.install_runtime("gpu")
    assert len(calls) == 2
    assert calls[0][0] == "uninstall"
    assert "onnxruntime-gpu" in calls[0]
    assert "onnxruntime" in calls[0]
    assert calls[1][0] == "install"
    assert any("onnxruntime-gpu" in a for a in calls[1])
    assert res["installed_pkg"] == "onnxruntime-gpu"
    assert res["installed_version"] == "1.20.0"
    # 装完必须返回 restart_required 提示前端
    assert res["restart_required"] is True


def test_install_runtime_install_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 注：_pip 支持 mirror= kwarg（官方源失败时切镜像重试），mock 必须接受同签名
    def fake_pip(args, mirror=None):
        if args[0] == "install":
            return 1, "ERROR: no matching distribution"
        return 0, ""

    monkeypatch.setattr(ors, "_pip", fake_pip)
    with pytest.raises(RuntimeError, match="安装"):
        ors.install_runtime("gpu")


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_installs_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": True, "driver_version": "551.86", "gpu_name": "RTX 5090"},
    )
    # current_runtime 第一次返回未安装；install 完后第二次返回新状态
    rt_calls = iter([
        {"installed": None, "version": None, "providers": [], "cuda_available": False, "restart_required": False},
        {
            "installed": "onnxruntime-gpu",
            "version": "1.20.0",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "restart_required": False,
        },
    ])
    monkeypatch.setattr(ors, "current_runtime", lambda: next(rt_calls))
    captured: dict = {}

    def fake_install(target):
        captured["target"] = target
        return {
            "target": "onnxruntime-gpu>=1.20",
            "installed_pkg": "onnxruntime-gpu",
            "installed_version": "1.20.0",
            "restart_required": True,
            "stdout": "",
        }

    monkeypatch.setattr(ors, "install_runtime", fake_install)
    state = ors.bootstrap()
    assert captured["target"] == "gpu"
    assert state["installed"] == "onnxruntime-gpu"
    assert state["cuda_available"] is True


def test_bootstrap_warns_on_cpu_pkg_with_gpu_present(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """有 GPU 但只装了 CPU 包 → 不自动重装，仅 warn。"""
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": True, "driver_version": "551.86", "gpu_name": "RTX 5090"},
    )
    monkeypatch.setattr(
        ors, "current_runtime",
        lambda: {
            "installed": "onnxruntime",
            "version": "1.18.0",
            "providers": ["CPUExecutionProvider"],
            "cuda_available": False,
        },
    )
    install_called = []
    monkeypatch.setattr(
        ors, "install_runtime",
        lambda *a: install_called.append(a) or {},
    )
    with caplog.at_level("WARNING"):
        state = ors.bootstrap()
    assert install_called == []
    assert state["installed"] == "onnxruntime"
    assert any("CPU EP" in r.message for r in caplog.records)


def test_bootstrap_silent_when_already_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": True, "driver_version": "551.86", "gpu_name": "RTX 5090"},
    )
    monkeypatch.setattr(
        ors, "current_runtime",
        lambda: {
            "installed": "onnxruntime-gpu",
            "version": "1.20.0",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
        },
    )
    install_called = []
    monkeypatch.setattr(ors, "install_runtime", lambda *a: install_called.append(a) or {})
    state = ors.bootstrap()
    assert install_called == []
    assert state["cuda_available"] is True


def test_current_runtime_flags_restart_when_dist_version_mismatches_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pip 装了 onnxruntime-gpu==1.25.1，但已 import 的进程内还是旧 1.18.0 →
    restart_required=True（C extension 不能热替换）。"""
    monkeypatch.setattr(ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.25.1"))
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = ["AzureExecutionProvider", "CPUExecutionProvider"]
    fake_ort.__version__ = "1.18.0"
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    rt = ors.current_runtime()
    assert rt["restart_required"] is True


def test_current_runtime_flags_restart_when_gpu_pkg_but_no_cuda_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dist-info 写 onnxruntime-gpu 但 providers 没 CUDA EP → 进程仍是旧 CPU 包。"""
    monkeypatch.setattr(ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.20.0"))
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = ["AzureExecutionProvider", "CPUExecutionProvider"]
    fake_ort.__version__ = "1.20.0"
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    rt = ors.current_runtime()
    assert rt["restart_required"] is True


def test_current_runtime_no_restart_when_gpu_pkg_and_cuda_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.20.0"))
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    fake_ort.__version__ = "1.20.0"
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    rt = ors.current_runtime()
    assert rt["restart_required"] is False
    assert rt["cuda_available"] is True


def test_bootstrap_install_failure_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": False, "driver_version": None, "gpu_name": None},
    )
    monkeypatch.setattr(
        ors, "current_runtime",
        lambda: {"installed": None, "version": None, "providers": [], "cuda_available": False},
    )

    def _raise(_):
        raise RuntimeError("pip exploded")

    monkeypatch.setattr(ors, "install_runtime", _raise)
    state = ors.bootstrap()
    assert "error" in state
    assert "pip exploded" in state["error"]


# ---------------------------------------------------------------------------
# PP9.5 — preload + cuda_load_error
# ---------------------------------------------------------------------------


def test_preload_skips_on_unsupported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 Linux / 非 Windows（如 macOS）→ 整体跳过。"""
    monkeypatch.setattr(ors.sys, "platform", "darwin")
    res = ors._preload_torch_cuda_libs()
    assert res["platform_skip"] is True
    assert res["applied"] is False
    assert res["preloaded"] == []
    assert res["candidates"] == 0


def test_preload_windows_adds_torch_lib_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Windows + 装了 torch GPU build → os.add_dll_directory(torch/lib) 被调，
    返回结果 preloaded 含目录路径（onnxruntime dlopen cublasLt 等找得到）。"""
    monkeypatch.setattr(ors.sys, "platform", "win32")

    # 仿造 torch.__file__ 指向带 lib/ 的目录
    torch_pkg = tmp_path / "torch"
    (torch_pkg / "lib").mkdir(parents=True)
    fake_torch = MagicMock()
    fake_torch.__file__ = str(torch_pkg / "__init__.py")

    def _import(name: str):
        if name == "torch":
            return fake_torch
        raise ImportError(name)

    monkeypatch.setattr(ors.importlib, "import_module", _import)
    # 旁路：venv/Scripts/python.exe 下 `import torch` 走 sys.modules，
    # 但 _add_torch_dll_dirs_windows 用的是 `import torch` 函数局部 —— 用
    # monkeypatch sys.modules 直接喂
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    called: list[str] = []
    monkeypatch.setattr(
        ors.os,
        "add_dll_directory",
        lambda d: called.append(d) or MagicMock(),
        raising=False,
    )

    res = ors._preload_torch_cuda_libs()
    assert res["applied"] is True
    assert res["platform_skip"] is False
    assert str(torch_pkg / "lib") in res["preloaded"]
    assert called == [str(torch_pkg / "lib")]


def test_preload_windows_noop_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows 但 venv 没 torch → applied 仍为 True（平台支持），candidates=0。"""
    monkeypatch.setattr(ors.sys, "platform", "win32")
    monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)

    # 让 `import torch` 失败：覆盖 importlib.import_module 不够，因为函数内
    # 用的是字面 import；改 sys.modules 哨兵 + meta_path 不太干净。简单做法：
    # 把 ors.os.path.isdir 在没 torch 时也走 False 路径 —— 但实际函数体先 import
    # 失败就提前 return。这里直接构造 import 错误：
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    res = ors._preload_torch_cuda_libs()
    assert res["applied"] is True
    assert res["candidates"] == 0
    assert res["preloaded"] == []


def test_preload_noop_when_no_torch_nvidia_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """venv 里没 torch CUDA wheel → preload 不报错、candidates=0、preloaded 空。"""
    monkeypatch.setattr(ors.sys, "platform", "linux")

    def _no_pkg(_name: str):
        raise ImportError("not installed")

    monkeypatch.setattr(ors.importlib, "import_module", _no_pkg)
    res = ors._preload_torch_cuda_libs()
    assert res["applied"] is True
    assert res["candidates"] == 0
    assert res["preloaded"] == []


def test_preload_loads_present_libs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """模拟一个 nvidia.curand 包：mod.__path__ 下有 lib/libcurand.so.10 →
    应被 ctypes.CDLL 加载到，且 RTLD_GLOBAL 模式。"""
    monkeypatch.setattr(ors.sys, "platform", "linux")

    # 仿造 nvidia.curand 的 __path__ + lib/libcurand.so.10 文件
    pkg_root = tmp_path / "nvidia_curand_pkg"
    (pkg_root / "lib").mkdir(parents=True)
    so = pkg_root / "lib" / "libcurand.so.10"
    so.write_bytes(b"")  # 内容不重要，ctypes.CDLL 由我们 mock

    fake_mod = MagicMock()
    fake_mod.__path__ = [str(pkg_root)]

    def _import(name: str):
        if name == "nvidia.curand":
            return fake_mod
        raise ImportError(name)

    monkeypatch.setattr(ors.importlib, "import_module", _import)

    cdll_calls: list[tuple[str, int]] = []

    def _fake_cdll(path, mode=0):
        cdll_calls.append((path, mode))
        return MagicMock()

    monkeypatch.setattr(ors.ctypes, "CDLL", _fake_cdll)

    res = ors._preload_torch_cuda_libs()
    assert str(so) in res["preloaded"]
    assert any(p == str(so) for p, _ in cdll_calls)
    # 必须用 RTLD_GLOBAL，否则后续 onnxruntime dlopen 看不到符号
    mode = next(m for p, m in cdll_calls if p == str(so))
    assert mode == ors.ctypes.RTLD_GLOBAL


def test_record_cuda_load_error_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors, "_CUDA_LOAD_ERROR", None, raising=False)
    assert ors.get_cuda_load_error() is None
    ors.record_cuda_load_error("libcurand.so.10: cannot open shared object file")
    assert "libcurand" in (ors.get_cuda_load_error() or "")
    ors.record_cuda_load_error(None)
    assert ors.get_cuda_load_error() is None


# ---------------------------------------------------------------------------
# PP9.6 — CUDA runtime wheels 安装 / 回滚
# ---------------------------------------------------------------------------


def test_install_cuda_runtime_wheels_skip_on_non_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors.sys, "platform", "win32")
    res = ors._install_cuda_runtime_wheels()
    assert res["platform_skip"] is True
    assert res["installed"] == []


def test_install_cuda_runtime_wheels_installs_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux 上首次装：cuDNN 已存在 → 跳过；其它 6 个全装。"""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    # cuDNN 已装（torch 带的）；其它都没装
    monkeypatch.setattr(
        ors,
        "_is_dist_installed",
        lambda p: p == ors._NVIDIA_CUDNN_WHEEL,
    )
    pip_calls: list[list[str]] = []

    def fake_pip(args):
        pip_calls.append(args)
        return 0, "Successfully installed nvidia-curand-cu12 ..."

    monkeypatch.setattr(ors, "_pip", fake_pip)
    res = ors._install_cuda_runtime_wheels()
    assert res["platform_skip"] is False
    assert ors._NVIDIA_CUDNN_WHEEL in res["skipped"]
    assert ors._NVIDIA_CUDNN_WHEEL not in res["installed"]
    # 6 个都进了 install args
    for pkg in ors._NVIDIA_CUDA_RUNTIME_WHEELS:
        assert pkg in res["installed"]
    # 单次 pip install 调用
    install_calls = [c for c in pip_calls if c[0] == "install"]
    assert len(install_calls) == 1


def test_install_cuda_runtime_wheels_noop_when_all_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(ors, "_is_dist_installed", lambda _p: True)
    pip_calls: list[list[str]] = []
    monkeypatch.setattr(
        ors, "_pip", lambda args: (pip_calls.append(args) or (0, "")),
    )
    res = ors._install_cuda_runtime_wheels()
    assert res["installed"] == []
    assert pip_calls == []  # 全装好了不调 pip


def test_install_cuda_runtime_wheels_rolls_back_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pip install 挂 → 把本次想装的全卸掉再抛。venv 不被污染。"""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(ors, "_is_dist_installed", lambda _p: False)
    pip_calls: list[list[str]] = []

    def fake_pip(args, mirror=None):
        pip_calls.append(args)
        if args[0] == "install":
            return 1, "ERROR: pip 解析依赖失败"
        return 0, "uninstalled"

    monkeypatch.setattr(ors, "_pip", fake_pip)
    with pytest.raises(RuntimeError, match="CUDA runtime wheels"):
        ors._install_cuda_runtime_wheels()
    # 必须有一次 uninstall 调用做回滚
    assert any(c[0] == "uninstall" for c in pip_calls)


def test_install_runtime_gpu_path_calls_cuda_wheels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """install_runtime("gpu") 在 onnxruntime-gpu 装好后必须调 _install_cuda_runtime_wheels。"""
    monkeypatch.setattr(ors, "_pip", lambda _args: (0, "ok"))
    monkeypatch.setattr(
        ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.20.0"),
    )
    called: list[bool] = []

    def fake_install_wheels():
        called.append(True)
        return {"installed": ["nvidia-curand-cu12"], "skipped": [], "platform_skip": False, "stdout": ""}

    monkeypatch.setattr(ors, "_install_cuda_runtime_wheels", fake_install_wheels)
    res = ors.install_runtime("gpu")
    assert called == [True]
    assert res["cuda_runtime"]["installed"] == ["nvidia-curand-cu12"]


def test_install_runtime_cpu_path_skips_cuda_wheels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors, "_pip", lambda _args: (0, "ok"))
    monkeypatch.setattr(
        ors, "_query_dist_info", lambda: ("onnxruntime", "1.18.0"),
    )
    called: list[bool] = []
    monkeypatch.setattr(
        ors, "_install_cuda_runtime_wheels",
        lambda: (called.append(True), {"installed": []})[1],
    )
    res = ors.install_runtime("cpu")
    assert called == []  # CPU 路径不调
    assert res["cuda_runtime"] is None


def test_install_runtime_does_not_fail_when_cuda_wheels_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """onnxruntime-gpu 装好之后 CUDA wheels 装失败 → 不抛，记录 error 让 UI 显示。"""
    monkeypatch.setattr(ors, "_pip", lambda _args: (0, "ok"))
    monkeypatch.setattr(
        ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.20.0"),
    )

    def boom():
        raise RuntimeError("pip resolver could not satisfy")

    monkeypatch.setattr(ors, "_install_cuda_runtime_wheels", boom)
    res = ors.install_runtime("gpu")
    # ort-gpu 已装；不抛
    assert res["installed_pkg"] == "onnxruntime-gpu"
    assert "error" in res["cuda_runtime"]
    assert "pip resolver" in res["cuda_runtime"]["error"]


def test_current_runtime_exposes_cuda_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.20.0"))
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider", "CPUExecutionProvider"
    ]
    fake_ort.__version__ = "1.20.0"
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    ors.record_cuda_load_error("simulated dlopen failure")
    try:
        rt = ors.current_runtime()
        assert rt["cuda_load_error"] == "simulated dlopen failure"
        assert "preload" in rt
    finally:
        ors.record_cuda_load_error(None)


# ---------------------------------------------------------------------------
# PR-3 — 系统 CUDA 检测：避免覆盖系统 cuBLAS 造成 ABI 错位
# ---------------------------------------------------------------------------


def test_has_system_cuda_libs_via_cuda_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """CUDA_HOME + 系统也有 cuDNN → True。"""
    fake_root = tmp_path / "fake-cuda"
    (fake_root / "lib64").mkdir(parents=True)
    monkeypatch.setenv("CUDA_HOME", str(fake_root))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(ors.os.path, "isdir", lambda p: p.endswith(str(fake_root / "lib64")))
    import ctypes.util as _cu  # noqa: PLC0415
    # cuDNN 在系统 ld 里
    monkeypatch.setattr(_cu, "find_library", lambda name: "libcudnn.so.9" if name == "cudnn" else None)
    assert ors._has_system_cuda_libs() is True


def test_has_system_cuda_libs_via_default_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/usr/local/cuda/lib64 + 系统 cuDNN → True。"""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(
        ors.os.path,
        "isdir",
        lambda p: p == "/usr/local/cuda/lib64",
    )
    import ctypes.util as _cu  # noqa: PLC0415
    monkeypatch.setattr(_cu, "find_library", lambda name: "libcudnn.so.9" if name == "cudnn" else None)
    assert ors._has_system_cuda_libs() is True


def test_has_system_cuda_libs_returns_false_when_no_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(ors.os.path, "isdir", lambda _p: False)
    import ctypes.util as _cu  # noqa: PLC0415
    monkeypatch.setattr(_cu, "find_library", lambda _name: None)
    assert ors._has_system_cuda_libs() is False


def test_has_system_cuda_libs_returns_false_when_cudnn_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关键修复回归：系统有 CUDA Toolkit（cuBLAS 在 /usr/local/cuda）但**没装 cuDNN** —
    必须返回 False，让 torch wheel preload 兜底补 cuDNN。否则 onnxruntime
    dlopen libcudnn.so.9 失败 → 静默降 CPU（用户在云上实测踩到）。"""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(
        ors.os.path,
        "isdir",
        lambda p: p == "/usr/local/cuda/lib64",
    )
    import ctypes.util as _cu  # noqa: PLC0415
    # cuBLAS 在系统里，cuDNN 不在
    monkeypatch.setattr(_cu, "find_library", lambda name: "libcublas.so.12" if name == "cublas" else None)
    assert ors._has_system_cuda_libs() is False


def test_has_system_cuda_libs_returns_false_when_only_cublas_in_ld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没 CUDA_HOME、没 /usr/local/cuda，只有 ld 路径里的 cuBLAS（apt 装的）+
    没 cuDNN → False（同上：部分系统 CUDA 也算不完整）。"""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(ors.os.path, "isdir", lambda _p: False)
    import ctypes.util as _cu  # noqa: PLC0415
    monkeypatch.setattr(_cu, "find_library", lambda name: "libcublas.so.12" if name == "cublas" else None)
    assert ors._has_system_cuda_libs() is False


def test_preload_skips_when_system_cuda_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux + 系统 CUDA → 跳过 preload，避免 torch wheel 与系统 cuBLAS ABI 冲突。"""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(ors, "_has_system_cuda_libs", lambda: True)
    res = ors._preload_torch_cuda_libs()
    assert res["system_cuda_skip"] is True
    assert res["applied"] is False
    assert res["preloaded"] == []
    assert res["candidates"] == 0


def test_preload_runs_when_system_cuda_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux + 没系统 CUDA → 走原 preload 路径（哪怕没 torch wheel，至少 applied=True）。"""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(ors, "_has_system_cuda_libs", lambda: False)

    def _no_pkg(_name: str):
        raise ImportError("not installed")

    monkeypatch.setattr(ors.importlib, "import_module", _no_pkg)
    res = ors._preload_torch_cuda_libs()
    assert res["applied"] is True
    assert res["system_cuda_skip"] is False
