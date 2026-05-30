"""inference_core 单测：rank/alpha 从 metadata 读 + 多 LoRA 各自 inject。

回归 PR #17 作者在 anima_generate.py / anima_reg_ai.py 引入的两个 P0 bug：
  1. rank/alpha 硬编码 32/32（应该从顶层 ss_network_dim / ss_network_alpha 读）
  2. 多 LoRA 张量直加合到一个 LycorisNetwork（应该每份独立 inject）
"""
from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file

from studio.services.inference.core import LoRASpec, apply_loras, read_lora_meta


@contextmanager
def _patched_adapter(factory: object) -> Iterator[None]:
    """注入 fake AnimaLycorisAdapter 到 utils.lycoris_adapter（绕开 lycoris-lora 真依赖）。

    inference_core.apply_loras 内部 `from utils.lycoris_adapter import
    AnimaLycorisAdapter` 是函数局部 lazy import，不能直接 patch
    inference_core 的 module attr。用 sys.modules 替换整个
    utils.lycoris_adapter 模块。
    """
    fake_mod = types.ModuleType("utils.lycoris_adapter")
    fake_mod.AnimaLycorisAdapter = factory  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"utils.lycoris_adapter": fake_mod}):
        yield


def _write_lora_safetensors(
    path: Path,
    *,
    rank: int,
    alpha: float,
    algo: str,
    factor: int,
    weight_decompose: bool = False,
    rs_lora: bool = False,
) -> None:
    """写一个伪 LoRA safetensors，带 ss_* metadata；tensor 内容空。"""
    sd = {"lora_unet_dummy.lokr_w1": torch.zeros(2, 2)}
    meta = {
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(alpha),
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps({
            "algo": algo,
            "factor": factor,
            "preset": "anima_full",
            "weight_decompose": weight_decompose,
            "rs_lora": rs_lora,
        }),
    }
    save_file(sd, str(path), metadata=meta)


def test_read_lora_meta_from_ss_network_dim_alpha(tmp_path: Path) -> None:
    p = tmp_path / "lora.safetensors"
    _write_lora_safetensors(p, rank=64, alpha=32.0, algo="lokr", factor=16)

    meta = read_lora_meta(str(p))
    assert meta.rank == 64
    assert meta.alpha == 32.0
    assert meta.algo == "lokr"
    assert meta.factor == 16


def test_read_lora_meta_unusual_dim(tmp_path: Path) -> None:
    """rank=8 训练的 LoRA 必须读到 8，不能 fallback 到 32（旧 bug 核心场景）。"""
    p = tmp_path / "lora_dim8.safetensors"
    _write_lora_safetensors(p, rank=8, alpha=4.0, algo="lokr", factor=8)

    meta = read_lora_meta(str(p))
    assert meta.rank == 8
    assert meta.alpha == 4.0


def test_read_lora_meta_missing_metadata(tmp_path: Path) -> None:
    """metadata 完全缺失时回退默认。"""
    p = tmp_path / "no_meta.safetensors"
    save_file({"x": torch.zeros(1)}, str(p))

    meta = read_lora_meta(str(p))
    assert meta.rank == 32
    assert meta.algo == "lokr"
    assert meta.factor == 8


def test_read_lora_meta_invalid_fields(tmp_path: Path) -> None:
    """字段是非法字符串时不崩、回退默认。"""
    p = tmp_path / "bad.safetensors"
    save_file({"x": torch.zeros(1)}, str(p), metadata={
        "ss_network_dim": "not_a_number",
        "ss_network_alpha": "also_bad",
        "ss_network_args": "not_json{",
    })
    meta = read_lora_meta(str(p))
    assert meta.rank == 32
    assert meta.alpha == 32.0  # alpha fallback to rank
    assert meta.algo == "lokr"


def test_read_lora_meta_dora_and_rs_lora(tmp_path: Path) -> None:
    """DoRA / RS-LoRA 训练标志必须从 ss_network_args 读回。

    回归 bug：训练侧开 weight_decompose=True 写入文件 dora_scale 张量；
    推理侧若漏读，构造的 LycorisNetwork 不带 DoRA → dora_scale 全部
    unexpected、forward 缺归一化项 → LoRA 效果错位。
    RS-LoRA 同理：α/√rank vs α/rank 强度差 √rank 倍。
    """
    p = tmp_path / "dora_rs.safetensors"
    _write_lora_safetensors(
        p, rank=64, alpha=8.0, algo="lokr", factor=4,
        weight_decompose=True, rs_lora=True,
    )
    meta = read_lora_meta(str(p))
    assert meta.weight_decompose is True
    assert meta.rs_lora is True


def test_read_lora_meta_defaults_without_dora_rs(tmp_path: Path) -> None:
    """不显式设置时 weight_decompose / rs_lora 默认 False。"""
    p = tmp_path / "plain.safetensors"
    _write_lora_safetensors(p, rank=16, alpha=8.0, algo="lokr", factor=8)
    meta = read_lora_meta(str(p))
    assert meta.weight_decompose is False
    assert meta.rs_lora is False


def test_apply_loras_propagates_dora_and_rs_lora(tmp_path: Path) -> None:
    """apply_loras 必须把 weight_decompose / rs_lora 透传到 AnimaLycorisAdapter，
    否则 inject 出的网络结构与文件不匹配。"""
    p = tmp_path / "dora.safetensors"
    _write_lora_safetensors(
        p, rank=64, alpha=8.0, algo="lokr", factor=4,
        weight_decompose=True, rs_lora=True,
    )

    created: list[MagicMock] = []

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.init_kwargs = dict(kwargs)
        m.network = MagicMock()
        m.network.loras = []
        m.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        created.append(m)
        return m

    model = MagicMock()
    with _patched_adapter(_fake_adapter):
        apply_loras(model, [LoRASpec(path=str(p), scale=1.0)], device="cpu", dtype=torch.float32)

    assert created[0].init_kwargs["weight_decompose"] is True
    assert created[0].init_kwargs["rs_lora"] is True


def test_apply_loras_each_lora_injects_separately(tmp_path: Path) -> None:
    """多 LoRA 必须每个独立 inject —— PR #17 旧 bug 回归测试。

    旧 bug：把多个 LoRA 的 tensor 直接 add 到一份 dict 然后灌进
    一个 AnimaLycorisAdapter，LoKr 的 lokr_w1/lokr_w2 子矩阵相加
    ≠ 权重 delta 相加，出图错。
    """
    p1 = tmp_path / "a.safetensors"
    p2 = tmp_path / "b.safetensors"
    _write_lora_safetensors(p1, rank=16, alpha=8.0, algo="lokr", factor=8)
    _write_lora_safetensors(p2, rank=8, alpha=4.0, algo="lokr", factor=8)

    created: list[MagicMock] = []

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.init_kwargs = dict(kwargs)
        m.network = MagicMock()
        m.network.loras = []
        m.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        created.append(m)
        return m

    model = MagicMock()

    with _patched_adapter(_fake_adapter):
        adapters = apply_loras(
            model,
            [LoRASpec(path=str(p1), scale=1.0), LoRASpec(path=str(p2), scale=0.5)],
            device="cpu",
            dtype=torch.float32,
        )

    assert len(adapters) == 2
    # 每个 adapter 各 inject(model) 一次（不是合并到一个）
    for a in adapters:
        a.inject.assert_called_once_with(model)
    # rank/alpha 从 metadata 读，不是硬编码 32/32
    assert created[0].init_kwargs["rank"] == 16
    assert created[0].init_kwargs["alpha"] == 8.0
    assert created[1].init_kwargs["rank"] == 8
    assert created[1].init_kwargs["alpha"] == 4.0
    # multiplier 设为 spec.scale
    assert created[0].network.multiplier == 1.0
    assert created[1].network.multiplier == 0.5


def test_apply_loras_skips_missing_path(tmp_path: Path) -> None:
    p_real = tmp_path / "real.safetensors"
    _write_lora_safetensors(p_real, rank=16, alpha=8.0, algo="lokr", factor=8)
    p_fake = tmp_path / "nonexistent.safetensors"

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.network = MagicMock()
        m.network.loras = []
        m.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        return m

    model = MagicMock()
    with _patched_adapter(_fake_adapter):
        adapters = apply_loras(
            model,
            [LoRASpec(path=str(p_fake)), LoRASpec(path=str(p_real))],
            device="cpu",
            dtype=torch.float32,
        )

    assert len(adapters) == 1


def test_apply_loras_empty_specs() -> None:
    model = MagicMock()
    assert apply_loras(model, [], device="cpu", dtype=torch.float32) == []


def test_model_cache_hot_reloads_same_topology_lora_ckpt(tmp_path: Path) -> None:
    """XY lora_ckpt 切同结构 checkpoint 时只换权重，不 detach/reinject。"""
    p1 = tmp_path / "a.safetensors"
    p2 = tmp_path / "b.safetensors"
    _write_lora_safetensors(p1, rank=16, alpha=8.0, algo="lokr", factor=8)
    _write_lora_safetensors(p2, rank=16, alpha=8.0, algo="lokr", factor=8)

    created: list[MagicMock] = []

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.network = MagicMock()
        m.network.loras = []
        m.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        created.append(m)
        return m

    from runtime.anima_daemon import ModelCache

    cache = ModelCache()
    cache.model = MagicMock()
    cache.device = "cpu"
    cache.dtype = torch.float32

    with _patched_adapter(_fake_adapter):
        first = cache.apply_loras([{"path": str(p1), "scale": 1.0}])
        second = cache.apply_loras([{"path": str(p2), "scale": 0.5}])

    assert first is second
    assert len(created) == 1
    created[0].detach.assert_not_called()
    assert created[0].inject.call_count == 1
    assert created[0].load_state_dict.call_count == 2
    assert created[0].network.multiplier == 0.5
    assert cache.last_lora_specs == [LoRASpec(path=str(p2), scale=0.5)]


def test_model_cache_reinjects_when_lora_topology_changes(tmp_path: Path) -> None:
    p1 = tmp_path / "rank16.safetensors"
    p2 = tmp_path / "rank8.safetensors"
    _write_lora_safetensors(p1, rank=16, alpha=8.0, algo="lokr", factor=8)
    _write_lora_safetensors(p2, rank=8, alpha=4.0, algo="lokr", factor=8)

    created: list[MagicMock] = []

    def _fake_adapter(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.network = MagicMock()
        m.network.loras = []
        m.detach.return_value = True
        m.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        created.append(m)
        return m

    from runtime.anima_daemon import ModelCache

    cache = ModelCache()
    cache.model = MagicMock()
    cache.device = "cpu"
    cache.dtype = torch.float32

    with _patched_adapter(_fake_adapter):
        cache.apply_loras([{"path": str(p1), "scale": 1.0}])
        cache.apply_loras([{"path": str(p2), "scale": 1.0}])

    assert len(created) == 2
    created[0].detach.assert_called_once()
    created[1].inject.assert_called_once_with(cache.model)


# ---------------------------------------------------------------------------
# generate tempdir helpers
# ---------------------------------------------------------------------------


def test_generate_tempdir_path() -> None:
    """tempdir 路径基于 task_id，落在系统 tempdir 下。"""
    import tempfile
    from studio.services.inference.core import (
        GENERATE_TEMP_PREFIX,
        generate_tempdir,
    )
    d = generate_tempdir(42)
    assert d.parent == Path(tempfile.gettempdir())
    assert d.name == f"{GENERATE_TEMP_PREFIX}42"


def test_cleanup_generate_tempdir_removes_dir() -> None:
    """cleanup_generate_tempdir 清掉对应 task 的目录。"""
    from studio.services.inference.core import (
        cleanup_generate_tempdir,
        generate_tempdir,
    )
    d = generate_tempdir(99999)
    d.mkdir(parents=True, exist_ok=True)
    (d / "img.png").write_bytes(b"\x89PNG")
    assert d.exists()

    cleanup_generate_tempdir(99999)
    assert not d.exists()


def test_cleanup_generate_tempdir_noop_when_missing() -> None:
    """目录不存在时调 cleanup 是 noop（非 generate task 也安全）。"""
    from studio.services.inference.core import (
        cleanup_generate_tempdir,
        generate_tempdir,
    )
    d = generate_tempdir(88888)
    if d.exists():
        import shutil
        shutil.rmtree(d)
    cleanup_generate_tempdir(88888)


def test_cleanup_stale_generate_tempdirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """启动扫清：把所有 anima_gen_* 目录全清。"""
    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    from studio.services.inference.core import (
        GENERATE_TEMP_PREFIX,
        cleanup_stale_generate_tempdirs,
    )

    leak1 = tmp_path / f"{GENERATE_TEMP_PREFIX}111"
    leak2 = tmp_path / f"{GENERATE_TEMP_PREFIX}222"
    keep = tmp_path / "unrelated_dir"
    leak1.mkdir()
    (leak1 / "x.png").write_bytes(b"\x89")
    leak2.mkdir()
    keep.mkdir()
    (keep / "important.txt").write_text("dont touch")

    cleanup_stale_generate_tempdirs()

    assert not leak1.exists()
    assert not leak2.exists()
    assert keep.exists()  # 不带前缀的不动
    assert (keep / "important.txt").exists()
