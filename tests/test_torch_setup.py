"""PR-S2 — torch_setup 服务：detect + recommend + reinstall。"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock

import pytest

from studio.services import torch_setup as ts


# ---------------------------------------------------------------------------
# recommend_cu_tag: 驱动版本 → cu wheel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver,expected", [
    ("570.15", "cu128"),  # >= 555 → cu128
    ("555.86", "cu128"),  # 边界
    ("550.10", "cu126"),  # 边界
    ("547.0", "cu124"),
    ("530.30", "cu118"),  # >= 470
    ("470.0", "cu118"),   # 边界
    ("460.50", "cpu"),    # 太老
    (None, "cpu"),
    ("", "cpu"),
    ("not-a-version", "cpu"),
])
def test_recommend_cu_tag(driver, expected) -> None:
    assert ts.recommend_cu_tag(driver) == expected


# ---------------------------------------------------------------------------
# detect_torch
# ---------------------------------------------------------------------------


def test_detect_torch_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_pkg):
        raise PackageNotFoundError
    monkeypatch.setattr(ts, "_pkg_version", _raise)
    res = ts.detect_torch()
    assert res == {
        "installed": False, "version": None, "cuda_build": None,
        "cuda_available": False, "device_name": None,
    }


def test_detect_torch_cpu_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch 2.5.0+cpu → cuda_build='cpu', cuda_available=False。"""
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cpu")
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.5.0+cpu"
    fake_torch.cuda.is_available.return_value = False
    fake_torch.version.cuda = None
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    res = ts.detect_torch()
    assert res["installed"] is True
    assert res["version"] == "2.5.0+cpu"
    assert res["cuda_build"] == "cpu"
    assert res["cuda_available"] is False


def test_detect_torch_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch 2.5.0+cu128 + cuda 可用 → 全部 OK 字段。"""
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cu128")
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.5.0+cu128"
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.get_device_name.return_value = "RTX 5090"
    fake_torch.version.cuda = "12.8"
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    res = ts.detect_torch()
    assert res["cuda_build"] == "cu128"
    assert res["cuda_available"] is True
    assert res["device_name"] == "RTX 5090"


def test_detect_torch_cuda_build_no_suffix_falls_back_to_version_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """老 torch __version__ 没 +suffix → 用 torch.version.cuda 推 cu tag。"""
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "1.13.0")
    fake_torch = MagicMock()
    fake_torch.__version__ = "1.13.0"  # 没 + suffix
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.get_device_name.return_value = "Tesla T4"
    fake_torch.version.cuda = "11.8"
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    res = ts.detect_torch()
    assert res["cuda_build"] == "cu118"
    assert res["cuda_available"] is True


# ---------------------------------------------------------------------------
# current_status: 误装诊断
# ---------------------------------------------------------------------------


def test_current_status_flags_cpu_with_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU torch + 检测到 NVIDIA → is_cpu_with_gpu=True，UI 大警告。"""
    monkeypatch.setattr(ts, "detect_torch", lambda: {
        "installed": True, "version": "2.5.0+cpu", "cuda_build": "cpu",
        "cuda_available": False, "device_name": None,
    })
    monkeypatch.setattr(ts.onnxruntime_setup, "detect_cuda", lambda: {
        "available": True, "driver_version": "555.86", "gpu_name": "RTX 5090",
    })
    s = ts.current_status()
    assert s["is_cpu_with_gpu"] is True
    assert s["is_cuda_build_unavailable"] is False
    assert s["recommended_cu_tag"] == "cu128"


def test_current_status_flags_cuda_build_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """装 cu128 但 cuda.is_available()=False → is_cuda_build_unavailable=True。"""
    monkeypatch.setattr(ts, "detect_torch", lambda: {
        "installed": True, "version": "2.5.0+cu128", "cuda_build": "cu128",
        "cuda_available": False, "device_name": None,
    })
    monkeypatch.setattr(ts.onnxruntime_setup, "detect_cuda", lambda: {
        "available": True, "driver_version": "470.0", "gpu_name": "Tesla M40",
    })
    s = ts.current_status()
    assert s["is_cpu_with_gpu"] is False
    assert s["is_cuda_build_unavailable"] is True


def test_current_status_no_issue_when_cuda_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ts, "detect_torch", lambda: {
        "installed": True, "version": "2.5.0+cu128", "cuda_build": "cu128",
        "cuda_available": True, "device_name": "RTX 5090",
    })
    monkeypatch.setattr(ts.onnxruntime_setup, "detect_cuda", lambda: {
        "available": True, "driver_version": "555.86", "gpu_name": "RTX 5090",
    })
    s = ts.current_status()
    assert s["is_cpu_with_gpu"] is False
    assert s["is_cuda_build_unavailable"] is False
    assert s["cuda_available"] is True


# ---------------------------------------------------------------------------
# reinstall: 调 pip uninstall + install --index-url
# ---------------------------------------------------------------------------


def test_reinstall_invalid_target_raises() -> None:
    with pytest.raises(ValueError, match="非法 target"):
        ts.reinstall("xpu")


def test_reinstall_auto_picks_recommended(monkeypatch: pytest.MonkeyPatch) -> None:
    """target='auto' → 用 detect_cuda 驱动 → recommend_cu_tag。"""
    monkeypatch.setattr(ts.onnxruntime_setup, "detect_cuda", lambda: {
        "available": True, "driver_version": "555.86", "gpu_name": "RTX 5090",
    })
    pip_calls: list[list[str]] = []

    def fake_pip(args, **_kw):
        pip_calls.append(args)
        return 0, "ok"

    monkeypatch.setattr(ts, "_pip", fake_pip)
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cu128")

    res = ts.reinstall("auto")
    assert res["tag"] == "cu128"
    assert res["index_url"] == "https://download.pytorch.org/whl/cu128"
    assert res["restart_required"] is True
    # 第一 call uninstall，第二 call install
    assert pip_calls[0][:2] == ["uninstall", "-y"]
    assert "torch" in pip_calls[0] and "torchvision" in pip_calls[0]
    assert pip_calls[1][0] == "install"
    assert "--index-url" in pip_calls[1]
    assert "https://download.pytorch.org/whl/cu128" in pip_calls[1]


def test_reinstall_explicit_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    pip_calls: list[list[str]] = []
    monkeypatch.setattr(ts, "_pip", lambda args, **_kw: (pip_calls.append(args) or (0, "ok")))
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cu118")

    res = ts.reinstall("cu118")
    assert res["tag"] == "cu118"
    assert "https://download.pytorch.org/whl/cu118" in pip_calls[1]


def test_reinstall_cpu_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """cpu target 也走 PyTorch 自家 cpu index（避免 PyPI 默认歧义）。"""
    pip_calls: list[list[str]] = []
    monkeypatch.setattr(ts, "_pip", lambda args, **_kw: (pip_calls.append(args) or (0, "ok")))
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cpu")

    res = ts.reinstall("cpu")
    assert res["tag"] == "cpu"
    assert "https://download.pytorch.org/whl/cpu" in pip_calls[1]


def test_reinstall_pip_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pip(args, **_kw):
        if args[0] == "install":
            return 1, "ERROR: bad wheel"
        return 0, ""
    monkeypatch.setattr(ts, "_pip", fake_pip)
    with pytest.raises(RuntimeError, match="安装 torch"):
        ts.reinstall("cu128")


def test_cleanup_zombie_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """site-packages 里 `~*` 目录 + 文件应被清；正常包不动。"""
    fake_site = tmp_path / "site-packages"
    fake_site.mkdir()
    # 僵尸：~orch dir + ~orchvision dir + ~stray file
    (fake_site / "~orch-2.11.0.dist-info").mkdir()
    (fake_site / "~orch-2.11.0.dist-info" / "METADATA").write_text("fake")
    (fake_site / "~orchvision").mkdir()
    (fake_site / "~stray.txt").write_text("zombie file")
    # 正常包（不应动）
    (fake_site / "torch").mkdir()
    (fake_site / "torch" / "__init__.py").write_text("")
    (fake_site / "numpy").mkdir()

    monkeypatch.setattr(ts.sysconfig, "get_path", lambda _key: str(fake_site))
    cleaned = ts._cleanup_zombie_dirs()
    assert sorted(cleaned) == ["~orch-2.11.0.dist-info", "~orchvision", "~stray.txt"]
    assert not (fake_site / "~orch-2.11.0.dist-info").exists()
    assert not (fake_site / "~orchvision").exists()
    assert not (fake_site / "~stray.txt").exists()
    # 正常包没动
    assert (fake_site / "torch").exists()
    assert (fake_site / "numpy").exists()


def test_cleanup_zombie_dirs_empty_when_no_zombies(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake_site = tmp_path / "site-packages"
    fake_site.mkdir()
    (fake_site / "torch").mkdir()
    monkeypatch.setattr(ts.sysconfig, "get_path", lambda _key: str(fake_site))
    assert ts._cleanup_zombie_dirs() == []


def test_reinstall_calls_cleanup_zombie(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """reinstall 应在 pip 前后都调一次 cleanup（双保险）。"""
    cleanup_calls: list[None] = []
    monkeypatch.setattr(
        ts, "_cleanup_zombie_dirs",
        lambda: (cleanup_calls.append(None), [])[1],
    )
    monkeypatch.setattr(ts, "_pip", lambda args, **_kw: (0, "ok"))
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cu128")

    res = ts.reinstall("cu128")
    # cleanup 调了两次（pip 前 + uninstall 后 install 前）
    assert len(cleanup_calls) == 2
    assert "cleaned_zombies" in res


def test_reinstall_returns_stdout_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """stdout 长度 > 40 行时只保留尾部。"""
    long_log = "\n".join(f"line {i}" for i in range(100))
    monkeypatch.setattr(ts, "_pip", lambda args, **_kw: (0, long_log))
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cu128")

    res = ts.reinstall("cu128")
    tail_lines = res["stdout_tail"].splitlines()
    assert len(tail_lines) <= 40
    assert tail_lines[-1] == "line 99"
