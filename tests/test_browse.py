"""目录浏览端点测试。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import server
from studio.services.dataset import browse


def _setup_tree(root: Path) -> None:
    (root / "configs").mkdir()
    (root / "models").mkdir()
    (root / "models" / "anima.safetensors").write_bytes(b"x")
    (root / "README.md").write_text("hi", encoding="utf-8")


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake = tmp_path / "repo"
    fake.mkdir()
    _setup_tree(fake)
    monkeypatch.setattr(server, "REPO_ROOT", fake)
    monkeypatch.setattr(browse, "REPO_ROOT", fake)
    # PR-5 /api/browse 搬到 api/routers/browse.py，handler 用自己 import 的 REPO_ROOT
    from studio.api.routers import browse as _browse_router
    monkeypatch.setattr(_browse_router, "REPO_ROOT", fake)
    return fake


def test_list_dir_returns_sorted_entries(fake_repo: Path) -> None:
    result = browse.list_dir(fake_repo)
    names = [e["name"] for e in result["entries"]]
    # 目录排在文件前
    assert names == ["configs", "models", "README.md"]
    types = [e["type"] for e in result["entries"]]
    assert types == ["dir", "dir", "file"]
    assert result["selected"] is None


def test_list_dir_returns_posix_paths(fake_repo: Path) -> None:
    """path / parent 始终用 forward slash，避免与前端拼接混用 Windows 反斜杠。"""
    result = browse.list_dir(fake_repo / "models")
    assert "\\" not in result["path"]
    assert result["path"].endswith("/repo/models")


def test_list_dir_rejects_outside_repo(fake_repo: Path, tmp_path: Path) -> None:
    """lib 层默认仍然挡外部路径；server 端显式 opt-in 才放行。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(browse.BrowseError, match="outside repo"):
        browse.list_dir(outside)


def test_list_dir_allows_outside_when_opted_in(fake_repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    result = browse.list_dir(outside, allow_outside_repo=True)
    assert result["entries"] == []


def test_list_dir_missing_path(fake_repo: Path) -> None:
    with pytest.raises(browse.BrowseError, match="does not exist"):
        browse.list_dir(fake_repo / "nope")


def test_list_dir_file_path_falls_back_to_parent(fake_repo: Path) -> None:
    """传入文件路径时回退到父目录，并通过 selected 字段告诉前端高亮。"""
    result = browse.list_dir(fake_repo / "models" / "anima.safetensors")
    assert result["path"].endswith("/repo/models")
    assert result["selected"] == "anima.safetensors"
    names = [e["name"] for e in result["entries"]]
    assert "anima.safetensors" in names


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_api_browse_default(fake_repo: Path) -> None:
    client = TestClient(server.app)
    resp = client.get("/api/browse")
    assert resp.status_code == 200
    # 返回路径统一 POSIX 风格（不含反斜杠）
    assert "\\" not in resp.json()["path"]
    assert resp.json()["path"].endswith("/repo")
    names = [e["name"] for e in resp.json()["entries"]]
    assert "configs" in names


def test_api_browse_relative_path(fake_repo: Path) -> None:
    client = TestClient(server.app)
    resp = client.get("/api/browse?path=models")
    assert resp.status_code == 200
    names = [e["name"] for e in resp.json()["entries"]]
    assert names == ["anima.safetensors"]


def test_api_browse_allows_outside(fake_repo: Path, tmp_path: Path) -> None:
    """PathPicker 允许浏览外部绝对路径（设置页/预设页选数据盘上的模型）。"""
    client = TestClient(server.app)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.safetensors").write_bytes(b"x")
    resp = client.get(f"/api/browse?path={outside}")
    assert resp.status_code == 200
    names = [e["name"] for e in resp.json()["entries"]]
    assert names == ["model.safetensors"]


def test_api_browse_file_path_falls_back(fake_repo: Path) -> None:
    """传入文件路径不再 404，回退到父目录并设置 selected。"""
    client = TestClient(server.app)
    target = fake_repo / "models" / "anima.safetensors"
    resp = client.get(f"/api/browse?path={target}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected"] == "anima.safetensors"
    assert body["path"].endswith("/repo/models")
