"""PR-7a — flash_attention_setup wheel 解析 / 评分 / GitHub API。

不真触发 GitHub / nvidia-smi / pip：用 monkeypatch 替 urlopen / subprocess。
"""
from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock

import pytest

from studio.services.runtime import flash_attention as fa


# ---------------------------------------------------------------------------
# _parse_wheel
# ---------------------------------------------------------------------------


def test_parse_wheel_canonical() -> None:
    """官方命名：flash_attn-2.8.3+cu130torch2.11-cp312-cp312-win_amd64.whl"""
    tags = fa._parse_wheel("flash_attn-2.8.3+cu130torch2.11-cp312-cp312-win_amd64.whl")
    assert tags == {
        "cuda": "cu130",
        "torch": "torch2.11",
        "python": "cp312",
        "platform": "win_amd64",
    }


def test_parse_wheel_linux_with_minor_torch() -> None:
    tags = fa._parse_wheel(
        "flash_attn-2.7.0+cu124torch2.4.0-cp310-cp310-linux_x86_64.whl"
    )
    assert tags is not None
    assert tags["cuda"] == "cu124"
    assert tags["torch"] == "torch2.4.0"
    assert tags["platform"] == "linux_x86_64"


@pytest.mark.parametrize("name", [
    "flash_attn-2.8.3.whl",  # 缺 cuda/torch tag
    "flash_attn-2.8.3+cu130torch2.11-cp312-cp312-macosx_arm64.whl",  # macOS arm 也能 parse 但实际无候选
    "totally-not-a-wheel.txt",
    "",
])
def test_parse_wheel_invalid(name: str) -> None:
    """不匹配命名规则的 → None。注意 macosx 那个 regex 仍能 match（[\\w]+ 涵盖），
    但因为 fa.find_candidates 平台过滤会丢弃。这里只验证基本不挂。"""
    res = fa._parse_wheel(name)
    if name == "":
        assert res is None
    elif "macosx" in name:
        # regex 设计上接受任意 platform tag；过滤交给 find_candidates
        assert res is not None
    else:
        assert res is None


# ---------------------------------------------------------------------------
# _cuda_major
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tag,expected", [
    ("cu130", 13),
    ("cu128", 12),
    ("cu124", 12),
    ("cu99", 9),
    ("invalid", -1),
    ("", -1),
])
def test_cuda_major(tag: str, expected: int) -> None:
    assert fa._cuda_major(tag) == expected


# ---------------------------------------------------------------------------
# detect_env
# ---------------------------------------------------------------------------


def _patch_torch(monkeypatch: pytest.MonkeyPatch, version: str | None) -> None:
    """注入 / 移除 fake torch 模块。version=None 表示 torch 不可 import。"""
    import sys
    import types
    if version is None:
        monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]
        # `import torch` 在 sys.modules[torch]=None 时抛 ImportError，正是要的
        return
    fake = types.ModuleType("torch")
    fake.__version__ = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_detect_env_no_nvidia_smi_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """nvidia-smi + torch 都缺 → cuda_tag/cuda_ver/driver_cuda_ver 全 None。"""
    def boom(*_a, **_k):
        raise FileNotFoundError("no nvidia-smi")
    monkeypatch.setattr(fa.subprocess, "run", boom)
    _patch_torch(monkeypatch, None)
    env = fa.detect_env()
    assert env["cuda_tag"] is None
    assert env["cuda_ver"] is None
    assert env["driver_cuda_ver"] is None
    assert env["torch_tag"] is None
    assert env["python_tag"].startswith("cp")


def test_detect_env_prefers_torch_cu_tag_over_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关键回归：torch 装的是 cu128，driver 报 13.0 → cuda_tag = cu128（torch 的 ABI），
    driver_cuda_ver = "13.0" 单独存供 UI 显示。

    此前把 nvidia-smi 当 cuda_tag 来源 → 选 cu130 wheel → ABI 不匹配 → import 失败。
    """
    fake = MagicMock(returncode=0, stdout="CUDA Version: 13.0 \n", stderr="")
    monkeypatch.setattr(fa.subprocess, "run", lambda *a, **k: fake)
    _patch_torch(monkeypatch, "2.11.0+cu128")
    env = fa.detect_env()
    assert env["cuda_tag"] == "cu128"
    assert env["cuda_ver"] == "12.8"
    assert env["driver_cuda_ver"] == "13.0"
    assert env["torch_tag"] == "torch2.11"
    assert env["torch_ver"] == "2.11.0+cu128"


def test_detect_env_falls_back_to_nvidia_smi_when_torch_has_no_cu_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-only torch（无 +cu 后缀）→ fallback nvidia-smi 给 cuda_tag。"""
    fake = MagicMock(returncode=0, stdout="CUDA Version: 12.8 \n", stderr="")
    monkeypatch.setattr(fa.subprocess, "run", lambda *a, **k: fake)
    _patch_torch(monkeypatch, "2.11.0")  # 没 +cu
    env = fa.detect_env()
    assert env["cuda_tag"] == "cu128"
    assert env["cuda_ver"] == "12.8"
    assert env["driver_cuda_ver"] == "12.8"
    assert env["torch_tag"] == "torch2.11"


def test_detect_env_no_torch_falls_back_to_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch 没装时（venv 还没初始化）→ 用 nvidia-smi 拿 cuda_tag，让用户至少能选个候选。"""
    fake = MagicMock(returncode=0, stdout="CUDA Version: 12.8 \n", stderr="")
    monkeypatch.setattr(fa.subprocess, "run", lambda *a, **k: fake)
    _patch_torch(monkeypatch, None)
    env = fa.detect_env()
    assert env["cuda_tag"] == "cu128"
    assert env["cuda_ver"] == "12.8"
    assert env["driver_cuda_ver"] == "12.8"
    assert env["torch_tag"] is None


# ---------------------------------------------------------------------------
# find_candidates
# ---------------------------------------------------------------------------


def _mock_releases(monkeypatch: pytest.MonkeyPatch, payload: Any) -> None:
    """把 urllib.request.urlopen 替成返回 payload（list / dict / str / Exception）。"""
    if isinstance(payload, BaseException):
        def _raise(*_a, **_k):
            raise payload
        monkeypatch.setattr(fa.urllib.request, "urlopen", _raise)
        return
    body = json.dumps(payload).encode("utf-8")
    monkeypatch.setattr(
        fa.urllib.request,
        "urlopen",
        lambda *_a, **_k: io.BytesIO(body),
    )


def test_find_candidates_no_platform_returns_empty() -> None:
    """env.platform=None（macOS arm64 等）→ 直接返回空，不打 GitHub。"""
    candidates, err = fa.find_candidates({"platform": None})
    assert candidates == []
    assert err is None


def test_find_candidates_rate_limited_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub API 限流时返回 dict {"message": "..."} → fetch_error 带消息。"""
    _mock_releases(monkeypatch, {"message": "API rate limit exceeded for ..."})
    candidates, err = fa.find_candidates({
        "platform": "win_amd64", "torch_tag": "torch2.5", "cuda_tag": "cu128",
        "python_tag": "cp311",
    })
    assert candidates == []
    assert err is not None and "rate limit" in err


def test_find_candidates_network_error_returns_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_releases(monkeypatch, urllib.error.URLError("getaddrinfo failed"))
    candidates, err = fa.find_candidates({
        "platform": "linux_x86_64", "torch_tag": "torch2.5", "cuda_tag": "cu128",
        "python_tag": "cp311",
    })
    assert candidates == []
    assert err is not None and "getaddrinfo" in err


def test_find_candidates_filters_platform_and_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """平台 / torch 不匹配的 wheel 直接丢弃，不进 candidates。"""
    payload = [{
        "assets": [
            {  # 平台不匹配
                "name": "flash_attn-2.8+cu128torch2.5-cp311-cp311-linux_x86_64.whl",
                "browser_download_url": "https://x/linux.whl",
            },
            {  # torch 不匹配
                "name": "flash_attn-2.8+cu128torch2.4-cp311-cp311-win_amd64.whl",
                "browser_download_url": "https://x/torch24.whl",
            },
            {  # 完美匹配
                "name": "flash_attn-2.8+cu128torch2.5-cp311-cp311-win_amd64.whl",
                "browser_download_url": "https://x/perfect.whl",
            },
        ],
    }]
    _mock_releases(monkeypatch, payload)
    candidates, err = fa.find_candidates({
        "platform": "win_amd64", "torch_tag": "torch2.5",
        "cuda_tag": "cu128", "python_tag": "cp311",
    })
    assert err is None
    assert len(candidates) == 1
    assert candidates[0]["name"].startswith("flash_attn-2.8+cu128torch2.5-cp311-cp311-win_amd64")
    assert candidates[0]["usable"] is True


def test_find_candidates_python_mismatch_marks_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python ABI 不一致 → usable=False；usable wheel 不存在时 find_best_wheel 返回 None。"""
    payload = [{
        "assets": [{
            "name": "flash_attn-2.8+cu128torch2.5-cp310-cp310-win_amd64.whl",
            "browser_download_url": "https://x/cp310.whl",
        }],
    }]
    _mock_releases(monkeypatch, payload)
    env = {
        "platform": "win_amd64", "torch_tag": "torch2.5",
        "cuda_tag": "cu128", "python_tag": "cp311",
    }
    candidates, err = fa.find_candidates(env)
    assert err is None
    assert len(candidates) == 1
    assert candidates[0]["usable"] is False
    assert any("Python 不兼容" in n for n in candidates[0]["notes"])
    # find_best_wheel 拒绝 unusable
    assert fa.find_best_wheel(env) is None


def test_find_candidates_cuda_minor_diff_scores_lower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同大版本 CUDA 仍 usable，但 score 比精确匹配低。"""
    payload = [{
        "assets": [
            {
                "name": "flash_attn-2.8+cu124torch2.5-cp311-cp311-win_amd64.whl",
                "browser_download_url": "https://x/cu124.whl",
            },
            {
                "name": "flash_attn-2.8+cu128torch2.5-cp311-cp311-win_amd64.whl",
                "browser_download_url": "https://x/cu128.whl",
            },
        ],
    }]
    _mock_releases(monkeypatch, payload)
    env = {
        "platform": "win_amd64", "torch_tag": "torch2.5",
        "cuda_tag": "cu128", "python_tag": "cp311",
    }
    candidates, err = fa.find_candidates(env)
    assert err is None
    # 排序：cu128 (精确, score 40) 在 cu124 (同大版本, score 30) 之前
    assert candidates[0]["name"].startswith("flash_attn-2.8+cu128")
    assert candidates[0]["score"] > candidates[1]["score"]
    # find_best_wheel 选最高分 usable
    assert fa.find_best_wheel(env) == "https://x/cu128.whl"


def test_find_candidates_cuda_major_diff_negative_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA 大版本不同（cu118 vs cu130）→ score 倒扣，notes 警告。"""
    payload = [{
        "assets": [{
            "name": "flash_attn-2.8+cu118torch2.5-cp311-cp311-win_amd64.whl",
            "browser_download_url": "https://x/cu118.whl",
        }],
    }]
    _mock_releases(monkeypatch, payload)
    env = {
        "platform": "win_amd64", "torch_tag": "torch2.5",
        "cuda_tag": "cu130", "python_tag": "cp311",
    }
    candidates, err = fa.find_candidates(env)
    assert err is None
    assert len(candidates) == 1
    # 大版本不同：Python +20, CUDA -5 = 15
    assert candidates[0]["score"] == 15
    assert any("CUDA 大版本不同" in n for n in candidates[0]["notes"])


# ---------------------------------------------------------------------------
# current_status / install
# ---------------------------------------------------------------------------


def test_current_status_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_pkg):
        raise fa.importlib.metadata.PackageNotFoundError
    monkeypatch.setattr(fa.importlib.metadata, "version", _raise)
    s = fa.current_status()
    assert s == {"installed": False, "version": None}


def test_current_status_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fa.importlib.metadata, "version", lambda _: "2.8.3")
    s = fa.current_status()
    assert s == {"installed": True, "version": "2.8.3"}


def test_install_no_url_unsupported_platform_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fa, "detect_env",
        lambda: {"platform": None, "torch_tag": "torch2.5", "python_tag": "cp311"},
    )
    with pytest.raises(RuntimeError, match="不支持的平台"):
        fa.install()


def test_install_no_url_no_torch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fa, "detect_env",
        lambda: {"platform": "win_amd64", "torch_tag": None, "python_tag": "cp311"},
    )
    with pytest.raises(RuntimeError, match="未检测到 PyTorch"):
        fa.install()


def test_install_no_candidate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fa, "detect_env",
        lambda: {
            "platform": "win_amd64", "torch_tag": "torch2.5",
            "python_tag": "cp311", "cuda_tag": "cu128",
        },
    )
    monkeypatch.setattr(fa, "find_best_wheel", lambda _env: None)
    with pytest.raises(RuntimeError, match="未找到可用 wheel"):
        fa.install()


def test_install_pip_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock(returncode=1, stdout="", stderr="ERROR: some pip failure")
    monkeypatch.setattr(fa.subprocess, "run", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="pip install 失败"):
        fa.install("https://x/wheel.whl")


def test_install_success_returns_status(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock(returncode=0, stdout="Successfully installed flash_attn-2.8.3\n", stderr="")
    monkeypatch.setattr(fa.subprocess, "run", lambda *a, **k: fake)
    monkeypatch.setattr(fa.importlib.metadata, "version", lambda _: "2.8.3")
    res = fa.install("https://x/wheel.whl")
    assert res["installed"] is True
    assert res["version"] == "2.8.3"
    assert res["url"] == "https://x/wheel.whl"
    assert res["restart_required"] is True
    assert "Successfully installed" in res["stdout_tail"]
