"""PR-S1a — tools/select_torch_index.py bootstrap helper。

不真跑 nvidia-smi，用 monkeypatch 模拟。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HELPER_PATH = _REPO_ROOT / "tools" / "select_torch_index.py"


@pytest.fixture
def helper_module():
    """通过文件路径手动加载 helper —— tools/ 不是 package（无 __init__.py）。"""
    spec = importlib.util.spec_from_file_location(
        "_select_torch_index_for_test", _HELPER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# select_index_url: driver 主号 → URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("major,expected_tag", [
    (596, "cu128"),  # 用户实测驱动
    (570, "cu128"),
    (555, "cu128"),  # 边界
    (554, "cu126"),  # 边界下方
    (550, "cu126"),
    (549, "cu124"),
    (545, "cu124"),
    (470, "cu118"),
    (469, None),     # 太旧
    (None, None),
])
def test_select_index_url(helper_module, major, expected_tag) -> None:
    res = helper_module.select_index_url(major)
    if expected_tag is None:
        assert res is None
    else:
        assert res == f"https://download.pytorch.org/whl/{expected_tag}"


# ---------------------------------------------------------------------------
# detect_driver_major: nvidia-smi 解析
# ---------------------------------------------------------------------------


def test_detect_driver_major_no_nvidia_smi(
    helper_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a, **_k):
        raise FileNotFoundError("no nvidia-smi")
    monkeypatch.setattr(helper_module.subprocess, "run", boom)
    assert helper_module.detect_driver_major() is None


def test_detect_driver_major_returncode_nonzero(
    helper_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        helper_module.subprocess, "run",
        lambda *a, **k: MagicMock(returncode=9, stdout="", stderr="error"),
    )
    assert helper_module.detect_driver_major() is None


def test_detect_driver_major_parses_output(
    helper_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        helper_module.subprocess, "run",
        lambda *a, **k: MagicMock(returncode=0, stdout="596.36\n", stderr=""),
    )
    assert helper_module.detect_driver_major() == 596


def test_detect_driver_major_garbage_output(
    helper_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        helper_module.subprocess, "run",
        lambda *a, **k: MagicMock(returncode=0, stdout="not a version\n", stderr=""),
    )
    assert helper_module.detect_driver_major() is None


def test_detect_driver_major_first_line_when_multi_gpu(
    helper_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """nvidia-smi 多卡时一行一个；用第一个就行（同机驱动版本一致）。"""
    monkeypatch.setattr(
        helper_module.subprocess, "run",
        lambda *a, **k: MagicMock(returncode=0, stdout="555.86\n555.86\n", stderr=""),
    )
    assert helper_module.detect_driver_major() == 555


# ---------------------------------------------------------------------------
# main(): 端到端 + 输出格式
# ---------------------------------------------------------------------------


def test_main_outputs_url_no_trailing_newline(
    helper_module, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """shell 用 $() / for /f 取输出；末尾不带换行避免歧义。"""
    monkeypatch.setattr(helper_module, "detect_driver_major", lambda: 555)
    rc = helper_module.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert out == "https://download.pytorch.org/whl/cu128"
    assert not out.endswith("\n")


def test_main_outputs_nothing_when_no_driver(
    helper_module, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(helper_module, "detect_driver_major", lambda: None)
    rc = helper_module.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_outputs_nothing_when_driver_too_old(
    helper_module, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """旧驱动（< 470）→ 静默 → caller 走 PyPI 默认（CPU torch）。"""
    monkeypatch.setattr(helper_module, "detect_driver_major", lambda: 460)
    rc = helper_module.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# 与 studio.services.torch_setup 的映射保持一致
# ---------------------------------------------------------------------------


def test_mapping_matches_torch_setup_canonical(helper_module) -> None:
    """helper 里的 _DRIVER_TO_CU 必须与 torch_setup 的源一致（避免 drift）。"""
    from studio.services.runtime.torch import _DRIVER_TO_BEST_CU
    canonical = [(int(thresh), tag) for thresh, tag in _DRIVER_TO_BEST_CU]
    assert helper_module._DRIVER_TO_CU == canonical
