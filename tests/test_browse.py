"""目录浏览端点测试。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import browse, server


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
    return fake


def test_list_dir_returns_sorted_entries(fake_repo: Path) -> None:
    result = browse.list_dir(fake_repo)
    names = [e["name"] for e in result["entries"]]
    # 目录排在文件前
    assert names == ["configs", "models", "README.md"]
    types = [e["type"] for e in result["entries"]]
    assert types == ["dir", "dir", "file"]


def test_list_dir_rejects_outside_repo(fake_repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(browse.BrowseError, match="outside repo"):
        browse.list_dir(outside)


def test_list_dir_missing_path(fake_repo: Path) -> None:
    with pytest.raises(browse.BrowseError, match="does not exist"):
        browse.list_dir(fake_repo / "nope")


def test_list_dir_not_a_directory(fake_repo: Path) -> None:
    with pytest.raises(browse.BrowseError, match="not a directory"):
        browse.list_dir(fake_repo / "README.md")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_api_browse_default(fake_repo: Path) -> None:
    client = TestClient(server.app)
    resp = client.get("/api/browse")
    assert resp.status_code == 200
    assert resp.json()["path"].replace("\\", "/").endswith("/repo")
    names = [e["name"] for e in resp.json()["entries"]]
    assert "configs" in names


def test_api_browse_relative_path(fake_repo: Path) -> None:
    client = TestClient(server.app)
    resp = client.get("/api/browse?path=models")
    assert resp.status_code == 200
    names = [e["name"] for e in resp.json()["entries"]]
    assert names == ["anima.safetensors"]


def test_api_browse_404_outside(fake_repo: Path, tmp_path: Path) -> None:
    client = TestClient(server.app)
    outside = tmp_path / "outside"
    outside.mkdir()
    resp = client.get(f"/api/browse?path={outside}")
    assert resp.status_code == 404
