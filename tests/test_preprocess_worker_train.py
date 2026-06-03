"""ADR 0010 — preprocess_worker train-scope path（PR-2 step D）。

直接调 `_run_crop_train` / `train_swap_entry` 行为，不通过 supervisor 子进程。
不跑真 upscaler（torch 模型加载耗时）—— upscale 主流程由 _run_upscale_train
仅做 path/manifest 派生测试，模型调用在端到端手测。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from studio.services.projects import projects
from studio.services.preprocess import manifest as pm
from studio.workers import preprocess_worker as worker


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """最小 project + version dict + 目录结构（不依赖 db）。"""
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    pdir = tmp_path / "projects" / "1-test"
    (pdir / "download").mkdir(parents=True)
    (pdir / "preprocess").mkdir(parents=True)
    train_root = pdir / "versions" / "v1" / "train"
    sub = train_root / "1_data"
    sub.mkdir(parents=True)
    return {
        "project": {"id": 1, "slug": "test"},
        "version": {"id": 10, "label": "v1"},
        "pdir": pdir,
        "sub": sub,
    }


def _silence(*_args, **_kwargs) -> None:
    pass


def _make_image(path: Path, size: tuple[int, int] = (200, 100), color=(255, 0, 0)) -> None:
    Image.new("RGB", size, color).save(path, format="PNG")


# ---------------------------------------------------------------------------
# train_swap_entry (Step D 配套 helper)
# ---------------------------------------------------------------------------


def test_swap_entry_removes_old_writes_new(env) -> None:
    """upscale 改扩展名（X.jpg → X.png）场景：swap entry 原子替换。"""
    sub = env["sub"]
    (sub / "X.jpg").write_bytes(b"jpg")
    (sub / "X.png").write_bytes(b"png" * 100)
    # 老 entry "1_data/X.jpg"
    pm.train_add_processed(
        env["pdir"], "v1", "1_data/X.jpg", {"origin": "X.jpg"},
    )

    pm.train_swap_entry(
        env["pdir"], "v1",
        old_name="1_data/X.jpg",
        new_name="1_data/X.png",
        meta={"origin": "X.jpg"},  # origin 不变（仍指向 download/X.jpg）
    )

    m = pm.train_load(env["pdir"], "v1")
    assert "1_data/X.jpg" not in m["images"]
    assert m["images"]["1_data/X.png"]["origin"] == "X.jpg"
    assert m["images"]["1_data/X.png"]["size"] == 300


# ---------------------------------------------------------------------------
# _run_crop_train: N=1 覆盖
# ---------------------------------------------------------------------------


def test_crop_train_single_rect_overwrites_source(env) -> None:
    sub = env["sub"]
    _make_image(sub / "X.png", size=(200, 100))
    pm.train_add_processed(env["pdir"], "v1", "1_data/X.png", {"origin": "X.png"})

    rc = worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"1_data/X.png": [{"x": 0.2, "y": 0.0, "w": 0.5, "h": 1.0}]}},
        _silence, _silence,
    )
    assert rc == 0
    # 覆盖在原 path
    assert (sub / "X.png").is_file()
    with Image.open(sub / "X.png") as out:
        assert out.size == (100, 100)  # 0.5×200 = 100
    # manifest entry 仍是 1_data/X.png，origin 保持
    entry = pm.train_get_entry(env["pdir"], "v1", "1_data/X.png")
    assert entry is not None
    assert entry["origin"] == "X.png"


# ---------------------------------------------------------------------------
# _run_crop_train: N>1 fan-out
# ---------------------------------------------------------------------------


def test_crop_train_fan_out_writes_multiple(env) -> None:
    sub = env["sub"]
    _make_image(sub / "Y.png", size=(200, 100))
    pm.train_add_processed(env["pdir"], "v1", "1_data/Y.png", {"origin": "Y.png"})

    rc = worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"1_data/Y.png": [
            {"x": 0.0, "y": 0.0, "w": 0.4, "h": 1.0},
            {"x": 0.5, "y": 0.0, "w": 0.4, "h": 1.0},
        ]}},
        _silence, _silence,
    )
    assert rc == 0
    # fan-out 派生 c0 / c1
    assert (sub / "Y_c0.png").is_file()
    assert (sub / "Y_c1.png").is_file()
    # 原 Y.png 物理被删（fan-out > 1）
    assert not (sub / "Y.png").is_file()
    # manifest 派生
    m = pm.train_load(env["pdir"], "v1")
    assert "1_data/Y.png" not in m["images"]
    assert "1_data/Y_c0.png" in m["images"]
    assert "1_data/Y_c1.png" in m["images"]
    # 多 crop 共享 origin
    assert m["images"]["1_data/Y_c0.png"]["origin"] == "Y.png"
    assert m["images"]["1_data/Y_c1.png"]["origin"] == "Y.png"


# ---------------------------------------------------------------------------
# _run_crop_train: 源不存在
# ---------------------------------------------------------------------------


def test_crop_train_skips_when_source_missing(env) -> None:
    rc = worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"1_data/ghost.png": [{"x": 0, "y": 0, "w": 0.5, "h": 0.5}]}},
        _silence, _silence,
    )
    assert rc == 0  # job 仍完成；单图 skip


# ---------------------------------------------------------------------------
# _run_crop_train: 拒非法 rel name
# ---------------------------------------------------------------------------


def test_crop_train_rejects_invalid_rel_name(env) -> None:
    rc = worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"../escape/X.png": [{"x": 0, "y": 0, "w": 0.5, "h": 0.5}]}},
        _silence, _silence,
    )
    # 路径校验 fail → skip 该项；job 仍 return 0
    assert rc == 0


# ---------------------------------------------------------------------------
# _run_crop_train: multi-crop 链式 origin 保持
# ---------------------------------------------------------------------------


def test_crop_train_origin_inherited_from_existing_entry(env) -> None:
    """老 entry 标 X_c0.png 的 origin=X.jpg；对 X_c0.png 再切 → 派生应继承
    origin=X.jpg（链式不丢 root）。"""
    sub = env["sub"]
    _make_image(sub / "X_c0.png", size=(100, 100))
    pm.train_add_processed(
        env["pdir"], "v1", "1_data/X_c0.png", {"origin": "X.jpg"},
    )

    worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"1_data/X_c0.png": [
            {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0},
            {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0},
        ]}},
        _silence, _silence,
    )
    m = pm.train_load(env["pdir"], "v1")
    assert m["images"]["1_data/X_c0_c0.png"]["origin"] == "X.jpg"
    assert m["images"]["1_data/X_c0_c1.png"]["origin"] == "X.jpg"


# ---------------------------------------------------------------------------
# _run_upscale_train: path 派生（不调真 upscaler）
# ---------------------------------------------------------------------------


def test_upscale_train_dispatches_to_train_paths(env, monkeypatch) -> None:
    """mock upscaler.upscale_file 验证 src/dst path + manifest 更新逻辑。"""
    sub = env["sub"]
    (sub / "X.jpg").write_bytes(b"raw")
    pm.train_add_processed(env["pdir"], "v1", "1_data/X.jpg", {"origin": "X.jpg"})

    called: dict = {}

    def fake_upscale(src, dst, **kwargs):
        called["src"] = src
        called["dst"] = dst
        # mock 产物：往 dst 写一个 PNG
        Image.new("RGB", (400, 200), (0, 255, 0)).save(dst, format="PNG")
        return {
            "model": "fake",
            "scale": 4,
            "src_size": [200, 100],
            "dst_size": [400, 200],
            "action": "upscale",
        }

    monkeypatch.setattr(worker.upscaler, "upscale_file", fake_upscale)
    monkeypatch.setattr(worker.upscaler, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(worker.upscaler, "resolve_device", lambda d: type("Dev", (), {"type": "cpu"})())
    monkeypatch.setattr(worker.upscaler, "resolve_dtype", lambda *a, **k: "float32")

    # mock model_downloader.upscaler_target → 返回一个存在的路径
    fake_model = env["pdir"] / "fake-model.pth"
    fake_model.write_bytes(b"weight")
    monkeypatch.setattr(
        worker.model_downloader, "upscaler_target", lambda label: fake_model,
    )

    rc = worker._run_upscale_train(
        env["project"], env["version"],
        {"mode": "all"},
        _silence, _silence,
    )
    assert rc == 0
    # src/dst path 对
    assert called["src"] == sub / "X.jpg"
    assert called["dst"] == sub / "X.png"
    # src 物理被删（扩展名变）
    assert not (sub / "X.jpg").is_file()
    assert (sub / "X.png").is_file()
    # manifest swap：old key 走，new key 入
    m = pm.train_load(env["pdir"], "v1")
    assert "1_data/X.jpg" not in m["images"]
    assert m["images"]["1_data/X.png"]["origin"] == "X.jpg"


def test_upscale_train_overwrites_when_extension_same(env, monkeypatch) -> None:
    """src 已是 .png（已处理过的图再上调）→ src == dst path → 普通 add_processed。"""
    sub = env["sub"]
    (sub / "X.png").write_bytes(b"old")
    pm.train_add_processed(env["pdir"], "v1", "1_data/X.png", {"origin": "X.png"})

    def fake_upscale(src, dst, **kwargs):
        Image.new("RGB", (400, 200), (0, 255, 0)).save(dst, format="PNG")
        return {"model": "fake", "scale": 4, "action": "upscale"}

    monkeypatch.setattr(worker.upscaler, "upscale_file", fake_upscale)
    monkeypatch.setattr(worker.upscaler, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(worker.upscaler, "resolve_device", lambda d: type("Dev", (), {"type": "cpu"})())
    monkeypatch.setattr(worker.upscaler, "resolve_dtype", lambda *a, **k: "float32")
    fake_model = env["pdir"] / "fake.pth"
    fake_model.write_bytes(b"w")
    monkeypatch.setattr(worker.model_downloader, "upscaler_target", lambda label: fake_model)

    rc = worker._run_upscale_train(
        env["project"], env["version"], {"mode": "all"}, _silence, _silence,
    )
    assert rc == 0
    # 仅 X.png（dst 覆盖 src）
    m = pm.train_load(env["pdir"], "v1")
    assert set(m["images"].keys()) == {"1_data/X.png"}
    assert (sub / "X.png").is_file()
