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
    def fake_pip(args):
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


def test_preload_skips_on_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ors.sys, "platform", "win32")
    res = ors._preload_torch_cuda_libs()
    assert res["platform_skip"] is True
    assert res["applied"] is False
    assert res["preloaded"] == []
    assert res["candidates"] == 0


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
