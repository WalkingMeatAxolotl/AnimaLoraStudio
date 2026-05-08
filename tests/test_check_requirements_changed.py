"""PR-S1b — tools/check_requirements_changed.py bootstrap helper。

不读真 requirements.txt，用 tmp_path 隔离测试。
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

_HELPER_PATH = Path(__file__).resolve().parent.parent / "tools" / "check_requirements_changed.py"


@pytest.fixture
def helper_module():
    spec = importlib.util.spec_from_file_location(
        "_check_req_for_test", _HELPER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_req(path: Path, content: str) -> None:
    """写 requirements.txt，强制用二进制避免 Windows 平台 \\n 被改成 \\r\\n。"""
    path.write_bytes(content.encode("utf-8"))


# ---------------------------------------------------------------------------
# compute_req_hash
# ---------------------------------------------------------------------------


def test_compute_hash_consistent(helper_module, tmp_path: Path) -> None:
    req = tmp_path / "requirements.txt"
    _make_req(req, "torch>=2.0.0\nnumpy>=1.24\n")
    h1 = helper_module.compute_req_hash(req)
    h2 = helper_module.compute_req_hash(req)
    assert h1 == h2
    assert h1 == hashlib.sha256(b"torch>=2.0.0\nnumpy>=1.24\n").hexdigest()


def test_compute_hash_changes_with_content(
    helper_module, tmp_path: Path
) -> None:
    req = tmp_path / "requirements.txt"
    _make_req(req, "torch>=2.0.0\n")
    h1 = helper_module.compute_req_hash(req)
    _make_req(req, "torch>=2.0.0\nmodelscope\n")  # 加新依赖
    h2 = helper_module.compute_req_hash(req)
    assert h1 != h2


# ---------------------------------------------------------------------------
# main(): stale / current / missing / written
# ---------------------------------------------------------------------------


def test_missing_requirements_outputs_missing(
    helper_module, tmp_path: Path, capsys
) -> None:
    rc = helper_module.main([
        "--marker", str(tmp_path / "marker"),
        "--requirements", str(tmp_path / "nonexistent.txt"),
    ])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "missing"


def test_no_marker_outputs_stale(
    helper_module, tmp_path: Path, capsys
) -> None:
    """老 venv 没 marker → 视为 stale，触发 caller 同步一次。"""
    req = tmp_path / "requirements.txt"
    _make_req(req, "torch\n")
    rc = helper_module.main([
        "--marker", str(tmp_path / "marker.sha256"),
        "--requirements", str(req),
    ])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "stale"


def test_marker_matches_outputs_current(
    helper_module, tmp_path: Path, capsys
) -> None:
    req = tmp_path / "requirements.txt"
    _make_req(req, "torch\n")
    marker = tmp_path / "marker.sha256"
    marker.write_text(helper_module.compute_req_hash(req), encoding="utf-8")

    helper_module.main([
        "--marker", str(marker), "--requirements", str(req),
    ])
    assert capsys.readouterr().out.strip() == "current"


def test_marker_differs_outputs_stale(
    helper_module, tmp_path: Path, capsys
) -> None:
    req = tmp_path / "requirements.txt"
    _make_req(req, "torch\nmodelscope\n")  # 新加 dep
    marker = tmp_path / "marker.sha256"
    marker.write_text("a" * 64, encoding="utf-8")  # 旧 hash

    helper_module.main([
        "--marker", str(marker), "--requirements", str(req),
    ])
    assert capsys.readouterr().out.strip() == "stale"


def test_corrupt_marker_outputs_stale(
    helper_module, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """marker 文件损坏（read 抛 OSError）→ 当 stale，下次同步重写。"""
    req = tmp_path / "requirements.txt"
    _make_req(req, "torch\n")
    marker = tmp_path / "marker.sha256"
    marker.write_text("anything", encoding="utf-8")

    real_read = Path.read_text

    def boom(self, *a, **k):
        if self == marker:
            raise OSError("simulated corruption")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", boom)
    helper_module.main([
        "--marker", str(marker), "--requirements", str(req),
    ])
    assert capsys.readouterr().out.strip() == "stale"


def test_update_marker_writes_hash(
    helper_module, tmp_path: Path, capsys
) -> None:
    req = tmp_path / "requirements.txt"
    _make_req(req, "torch\nmodelscope\n")
    marker = tmp_path / "subdir" / "marker.sha256"  # 不存在的子目录

    rc = helper_module.main([
        "--marker", str(marker), "--requirements", str(req),
        "--update-marker",
    ])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "written"
    # 父目录被自动创建（marker 在 venv/ 里，可能不存在）
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == helper_module.compute_req_hash(req)


def test_update_marker_overwrites_existing(
    helper_module, tmp_path: Path
) -> None:
    req = tmp_path / "requirements.txt"
    _make_req(req, "v2\n")
    marker = tmp_path / "marker.sha256"
    marker.write_text("old-hash", encoding="utf-8")

    helper_module.main([
        "--marker", str(marker), "--requirements", str(req),
        "--update-marker",
    ])
    assert marker.read_text(encoding="utf-8") == helper_module.compute_req_hash(req)


# ---------------------------------------------------------------------------
# 端到端：写 → 检查 current → 改 req → 检查 stale → 再写 → current
# ---------------------------------------------------------------------------


def test_full_lifecycle(helper_module, tmp_path: Path, capsys) -> None:
    req = tmp_path / "requirements.txt"
    marker = tmp_path / "marker.sha256"
    _make_req(req, "torch\n")

    # 1. 首次：没 marker → stale
    helper_module.main(["--marker", str(marker), "--requirements", str(req)])
    assert capsys.readouterr().out.strip() == "stale"

    # 2. 写 marker（caller pip install 成功后）
    helper_module.main([
        "--marker", str(marker), "--requirements", str(req), "--update-marker",
    ])
    assert capsys.readouterr().out.strip() == "written"

    # 3. 再检查 → current
    helper_module.main(["--marker", str(marker), "--requirements", str(req)])
    assert capsys.readouterr().out.strip() == "current"

    # 4. 改 req（git pull 加了新 dep）
    _make_req(req, "torch\nmodelscope\n")
    helper_module.main(["--marker", str(marker), "--requirements", str(req)])
    assert capsys.readouterr().out.strip() == "stale"

    # 5. 再次同步后写 marker
    helper_module.main([
        "--marker", str(marker), "--requirements", str(req), "--update-marker",
    ])
    helper_module.main(["--marker", str(marker), "--requirements", str(req)])
    capsys.readouterr()  # drain "written"
    helper_module.main(["--marker", str(marker), "--requirements", str(req)])
    assert capsys.readouterr().out.strip() == "current"
