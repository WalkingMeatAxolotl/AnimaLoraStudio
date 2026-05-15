"""upscaler service 单元测试。

不需要真权重也不需要 GPU：用 stub model（bicubic 上采样）走通整条
read → tile → write 流水线，验证：

- tiled_inference 拼接结果跟整图前向一致（无可见接缝）
- upscale_file 写 PNG + sidecar，元数据字段齐全
- resolve_device 的 auto / cpu / cuda-fallback 行为
- load_model 缓存按路径生效

spandrel / GPU 的真实加载留给手测（下载权重后跑一次小图）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from studio.services import upscaler


# ---------------------------------------------------------------------------
# stub: spandrel ImageModelDescriptor 的最小替代
# ---------------------------------------------------------------------------


class BicubicScaleModel(nn.Module):
    """假的"放大模型"：固定 scale 倍 bicubic。用来跑通流水线，不需要权重。"""

    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


class StubDescriptor:
    """模拟 spandrel.ImageModelDescriptor 的最小属性。"""

    def __init__(self, scale: int = 4):
        self.model = BicubicScaleModel(scale=scale)
        self.scale = scale
        self.input_channels = 3
        self.output_channels = 3


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> StubDescriptor:
    """让 upscaler.load_model 返回 stub，不真去 spandrel 加载。"""
    stub = StubDescriptor(scale=4)

    def fake_load(model_path: Path, *, device=None):
        if device is not None:
            stub.model.to(device)
        return stub

    monkeypatch.setattr(upscaler, "load_model", fake_load)
    return stub


@pytest.fixture(autouse=True)
def _clear_cache():
    upscaler.clear_cache()
    yield
    upscaler.clear_cache()


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_explicit() -> None:
    assert upscaler.resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert upscaler.resolve_device("auto").type == "cpu"


def test_resolve_device_cuda_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 cuda 但不可用 → 降级 cpu，不抛。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert upscaler.resolve_device("cuda").type == "cpu"


# ---------------------------------------------------------------------------
# tiled_inference 拼接正确性
# ---------------------------------------------------------------------------


def test_tiled_inference_matches_full_forward() -> None:
    """tile + pad 拼接结果应跟整图单次前向数值接近（bicubic 是确定性的）。"""
    model = BicubicScaleModel(scale=4)
    torch.manual_seed(0)
    img = torch.rand(1, 3, 64, 64)

    with torch.inference_mode():
        full = model(img)

    tiled = upscaler.tiled_inference(model, img, scale=4, tile_size=24, tile_pad=8)

    assert full.shape == tiled.shape == (1, 3, 256, 256)
    # bicubic 边界一致性：tile_pad 充足时拼接结果应 bit-level 接近
    assert torch.allclose(full, tiled, atol=1e-4), (
        f"max diff = {(full - tiled).abs().max().item()}"
    )


def test_tiled_inference_no_tile_path() -> None:
    """tile_size<=0 走整图单次前向，结果跟直接调 model 一样。"""
    model = BicubicScaleModel(scale=2)
    img = torch.rand(1, 3, 32, 32)
    out = upscaler.tiled_inference(model, img, scale=2, tile_size=0)
    with torch.inference_mode():
        ref = model(img)
    assert torch.equal(out, ref)


def test_tiled_inference_handles_non_divisible_size() -> None:
    """图像尺寸不是 tile_size 整数倍时也能拼出完整结果。"""
    model = BicubicScaleModel(scale=4)
    img = torch.rand(1, 3, 70, 50)  # 故意不整除
    out = upscaler.tiled_inference(model, img, scale=4, tile_size=32, tile_pad=4)
    assert out.shape == (1, 3, 280, 200)


# ---------------------------------------------------------------------------
# upscale_file 文件级 API
# ---------------------------------------------------------------------------


def _write_test_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    """造一张可识别的渐变图。"""
    import numpy as np

    w, h = size
    grad = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.tile(grad[None, :], (h, 1))
    rgb = np.stack([arr, arr // 2, 255 - arr], axis=-1)
    Image.fromarray(rgb).save(path, format="PNG")


def test_upscale_file_writes_png_and_sidecar(
    tmp_path: Path, stub_model: StubDescriptor
) -> None:
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    _write_test_image(src, size=(32, 32))

    logs: list[str] = []
    meta = upscaler.upscale_file(
        src,
        dst,
        model_path=tmp_path / "dummy.pth",  # stub 不读
        label="4x-AnimeSharp",
        tile_size=16,
        tile_pad=4,
        device="cpu",
        on_log=logs.append,
    )

    assert dst.exists()
    with Image.open(dst) as out:
        assert out.size == (128, 128)
        assert out.mode == "RGB"

    sidecar = dst.with_suffix(dst.suffix + ".preprocess.json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["source"] == "in.png"
    assert data["model"] == "4x-AnimeSharp"
    assert data["scale"] == 4
    assert data["src_size"] == [32, 32]
    assert data["dst_size"] == [128, 128]
    assert data["device"] == "cpu"
    assert "elapsed_seconds" in data
    assert "mtime" in data

    assert meta["dst_size"] == [128, 128]
    assert any("in.png" in line for line in logs)


def test_upscale_file_prewarms_thumbnails(
    tmp_path: Path, stub_model: StubDescriptor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prewarm_thumb_sizes 应该让 thumb_cache 直接拿到预生成的缩略图。"""
    from studio import thumb_cache

    monkeypatch.setattr(thumb_cache, "THUMB_CACHE_DIR", tmp_path / "thumbs")

    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    _write_test_image(src, size=(32, 32))

    upscaler.upscale_file(
        src,
        dst,
        model_path=tmp_path / "dummy.pth",
        device="cpu",
        tile_size=16,
        prewarm_thumb_sizes=[64, 128],
    )

    # 两档缓存都应该存在 — 后续 get_or_make_thumb 直接返回，不重新解码
    for sz in (64, 128):
        cached = thumb_cache.get_or_make_thumb(dst, sz)
        assert cached.exists()
        assert cached.parent == thumb_cache.THUMB_CACHE_DIR
        # JPEG 头校验，确保是真生成了缩略图
        assert cached.read_bytes()[:2] == b"\xff\xd8"


def test_upscale_file_no_sidecar(
    tmp_path: Path, stub_model: StubDescriptor
) -> None:
    """write_sidecar=False 时只产物图，不写 .preprocess.json。"""
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    _write_test_image(src)
    upscaler.upscale_file(
        src,
        dst,
        model_path=tmp_path / "dummy.pth",
        device="cpu",
        write_sidecar=False,
        tile_size=16,
    )
    assert dst.exists()
    assert not dst.with_suffix(dst.suffix + ".preprocess.json").exists()


def test_upscale_file_converts_rgba(
    tmp_path: Path, stub_model: StubDescriptor
) -> None:
    """RGBA 输入应被转 RGB（alpha 丢弃），不抛。"""
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    rgba = Image.new("RGBA", (16, 16), (200, 100, 50, 128))
    rgba.save(src, format="PNG")

    upscaler.upscale_file(
        src,
        dst,
        model_path=tmp_path / "dummy.pth",
        device="cpu",
        tile_size=8,
    )
    with Image.open(dst) as out:
        assert out.mode == "RGB"
        assert out.size == (64, 64)


# ---------------------------------------------------------------------------
# load_model 缓存
# ---------------------------------------------------------------------------


def test_load_model_missing_file_raises(tmp_path: Path) -> None:
    """模型路径不存在时给清晰错误（不要在 spandrel 里炸）。"""
    with pytest.raises(FileNotFoundError):
        upscaler.load_model(tmp_path / "nope.pth")


def test_load_model_caches_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同 path 二次调用走缓存，不重复 instantiation。"""
    model_path = tmp_path / "fake.pth"
    model_path.write_bytes(b"x")  # 文件存在即可，spandrel 是 mock

    call_count = {"n": 0}

    class FakeLoader:
        def load_from_file(self, _path: str) -> StubDescriptor:
            call_count["n"] += 1
            return StubDescriptor(scale=4)

    # 注入假 spandrel 模块
    import sys
    import types

    fake_spandrel = types.ModuleType("spandrel")
    fake_spandrel.ModelLoader = FakeLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spandrel", fake_spandrel)

    d1 = upscaler.load_model(model_path)
    d2 = upscaler.load_model(model_path)
    assert d1 is d2
    assert call_count["n"] == 1


def test_load_model_without_spandrel_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """spandrel 没装时给可操作的错误信息，不静默 ImportError。"""
    model_path = tmp_path / "fake.pth"
    model_path.write_bytes(b"x")

    import sys
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spandrel" or name.startswith("spandrel."):
            raise ImportError("no spandrel")
        return real_import(name, *args, **kwargs)

    # 同时清掉可能已 import 的 spandrel 缓存
    monkeypatch.delitem(sys.modules, "spandrel", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="spandrel"):
        upscaler.load_model(model_path)
