"""preprocess_crop worker: 实际切 PNG + 更新 manifest 的端到端测试。

不通过 supervisor 启子进程，直接调 _run_crop 函数，构造 PIL 图像断言文件 + manifest 状态。
覆盖：单裁剪覆盖、多裁剪 fan-out 删原图、源不存在 skip、多次裁剪链式命名 origin 保持。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from studio.services.projects import projects
from studio.services.preprocess import manifest as pm
from studio.workers import preprocess_worker as worker


@pytest.fixture
def project_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """造一个不依赖 db 的最小 project dict + 目录结构。"""
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    pdir = tmp_path / "projects" / "1-test"
    (pdir / "download").mkdir(parents=True)
    (pdir / "preprocess").mkdir(parents=True)
    return {
        "project": {"id": 1, "slug": "test"},
        "pdir": pdir,
    }


def _silence(*_args, **_kwargs) -> None:
    pass


def _make_image(path: Path, size: tuple[int, int] = (200, 100), color=(255, 0, 0)) -> None:
    Image.new("RGB", size, color).save(path, format="PNG")


def test_crop_single_rect_overwrites_source(project_env) -> None:
    """N=1：写 stem.png 覆盖源；manifest 写 origin。"""
    pdir = project_env["pdir"]
    src = pdir / "preprocess" / "X.png"
    _make_image(src, size=(200, 100))
    # 先存一条放大 entry
    pm.add_processed(pdir, "X.png", {"source": "X.png"})

    rc = worker._run_crop(
        project_env["project"],
        {"crops": {"X.png": [{"x": 0.2, "y": 0.0, "w": 0.5, "h": 1.0}]}},
        _silence,
        _silence,
    )
    assert rc == 0
    out = pdir / "preprocess" / "X.png"
    assert out.is_file()
    with Image.open(out) as img:
        # 200×0.5 = 100, 100×1.0 = 100
        assert img.size == (100, 100)
    entry = pm.get_entry(pdir, "X.png")
    assert entry is not None
    assert entry["origin"] == "X.png"


def test_crop_multi_rect_fans_out_and_deletes_source(project_env) -> None:
    """N>1：写 X_c0.png / X_c1.png，删原 X.png；manifest 替换为 2 个 entry。"""
    pdir = project_env["pdir"]
    src = pdir / "preprocess" / "X.png"
    _make_image(src, size=(200, 100))
    pm.add_processed(pdir, "X.png", {"source": "X.png"})

    rc = worker._run_crop(
        project_env["project"],
        {"crops": {"X.png": [
            {"x": 0.0, "y": 0.0, "w": 0.4, "h": 1.0},
            {"x": 0.6, "y": 0.0, "w": 0.4, "h": 1.0},
        ]}},
        _silence,
        _silence,
    )
    assert rc == 0
    assert not src.exists(), "源 X.png 应被删（已 fan-out）"
    assert (pdir / "preprocess" / "X_c0.png").is_file()
    assert (pdir / "preprocess" / "X_c1.png").is_file()
    m = pm.load(pdir)
    assert "X.png" not in m["images"]
    assert m["images"]["X_c0.png"]["origin"] == "X.png"
    assert m["images"]["X_c1.png"]["origin"] == "X.png"


def test_crop_falls_back_to_download(project_env) -> None:
    """preprocess/ 没有源 → 回退 download/，origin = 源名（不经过 upscale 也能裁）。"""
    pdir = project_env["pdir"]
    dl = pdir / "download" / "Y.png"
    _make_image(dl, size=(120, 80))
    rc = worker._run_crop(
        project_env["project"],
        {"crops": {"Y.png": [{"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}]}},
        _silence,
        _silence,
    )
    assert rc == 0
    out = pdir / "preprocess" / "Y.png"
    assert out.is_file()
    entry = pm.get_entry(pdir, "Y.png")
    assert entry is not None
    assert entry["origin"] == "Y.png"


def test_crop_chain_preserves_root_origin(project_env) -> None:
    """先 fan-out 出 X_c0/X_c1，再对 X_c0 裁剪 → X_c0_c0/X_c0_c1，origin 保持 root。"""
    pdir = project_env["pdir"]
    src = pdir / "preprocess" / "X.png"
    _make_image(src, size=(200, 100))
    pm.add_processed(pdir, "X.png", {"source": "X.jpg"})

    # 一次 fan-out
    worker._run_crop(
        project_env["project"],
        {"crops": {"X.png": [
            {"x": 0.0, "y": 0.0, "w": 0.4, "h": 1.0},
            {"x": 0.5, "y": 0.0, "w": 0.4, "h": 1.0},
        ]}},
        _silence, _silence,
    )
    # 二次：对 X_c0.png 再 fan-out
    worker._run_crop(
        project_env["project"],
        {"crops": {"X_c0.png": [
            {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0},
            {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0},
        ]}},
        _silence, _silence,
    )
    m = pm.load(pdir)
    # X_c0.png 应被替换；X_c1.png 留着
    assert "X_c0.png" not in m["images"]
    assert "X_c1.png" in m["images"]
    assert m["images"]["X_c0_c0.png"]["origin"] == "X.jpg"
    assert m["images"]["X_c0_c1.png"]["origin"] == "X.jpg"


def test_crop_skips_missing_source(project_env) -> None:
    """preprocess/ 和 download/ 都没源 → skip，不抛错。"""
    pdir = project_env["pdir"]
    rc = worker._run_crop(
        project_env["project"],
        {"crops": {"ghost.png": [{"x": 0, "y": 0, "w": 0.5, "h": 0.5}]}},
        _silence, _silence,
    )
    assert rc == 0
    # 没生成产物，manifest 仍为空
    assert pm.load(pdir) == {"images": {}}


def test_crop_empty_payload(project_env) -> None:
    rc = worker._run_crop(
        project_env["project"], {"crops": {}}, _silence, _silence
    )
    assert rc == 0
