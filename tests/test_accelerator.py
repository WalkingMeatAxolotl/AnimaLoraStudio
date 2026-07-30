from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from utils.accelerator import configure_miopen_cache, detect_accelerator, torch_device_type


def _fake_torch(*, hip=None, cuda=None, available=False, name="GPU"):
    cuda_api = MagicMock()
    cuda_api.is_available.return_value = available
    cuda_api.device_count.return_value = 1 if available else 0
    cuda_api.get_device_name.return_value = name
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip, cuda=cuda),
        cuda=cuda_api,
    )


def test_detect_rocm_uses_cuda_device_api() -> None:
    fake = _fake_torch(hip="7.14", available=True, name="AMD Radeon RX 7900 XTX")
    info = detect_accelerator(fake)
    assert info.backend == "rocm"
    assert info.build == "rocm7.14"
    assert info.available is True
    assert info.device_name == "AMD Radeon RX 7900 XTX"
    assert torch_device_type(fake) == "cuda"


def test_detect_cuda() -> None:
    info = detect_accelerator(_fake_torch(cuda="12.8", available=True, name="RTX"))
    assert info.backend == "cuda"
    assert info.build == "cu128"


def test_detect_cpu() -> None:
    info = detect_accelerator(_fake_torch())
    assert info.backend == "cpu"
    assert info.available is False
    assert torch_device_type(_fake_torch()) == "cpu"


def test_configure_miopen_cache_uses_writable_app_directory(tmp_path) -> None:
    env = {}
    configured = configure_miopen_cache(tmp_path, env)

    assert configured["MIOPEN_USER_DB_PATH"] == str(tmp_path / ".cache" / "miopen" / "db")
    assert configured["MIOPEN_CUSTOM_CACHE_DIR"] == str(tmp_path / ".cache" / "miopen" / "kernels")
    assert (tmp_path / ".cache" / "miopen" / "db").is_dir()
    assert (tmp_path / ".cache" / "miopen" / "kernels").is_dir()


def test_configure_miopen_cache_preserves_user_overrides(tmp_path) -> None:
    env = {
        "MIOPEN_USER_DB_PATH": "D:/custom/db",
        "MIOPEN_CUSTOM_CACHE_DIR": "D:/custom/kernels",
    }
    assert configure_miopen_cache(tmp_path, env) == env
