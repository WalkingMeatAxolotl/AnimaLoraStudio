"""anima_generate.py 的 XY 矩阵循环单测。

不跑实际推理（mock `_T.sample_image` 返回伪 PIL）；只校验：
  - 遍历顺序：(yi, xi) 双层循环；y=None 退化成单行
  - 文件命名 xy_x{xi:02d}_y{yi:02d}_s{seed}.png
  - update_monitor 收到 sample_path + xy 元数据
  - lora_scale 轴：每个 cell 进入前 multiplier 被重置 + 按 axis 值更新
  - cfg.seed=0：所有 cell 共享同一随机种子（仅 axis=seed 时才按 cell 覆盖）
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# 让 import runtime.anima_generate 能找到 anima_train（脚本顶部 sys.path 操作）
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "runtime"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


@pytest.fixture
def gen_module(monkeypatch):
    """import runtime.anima_generate，把外部依赖换成 mock。"""
    # 顶层 import 期间会 import anima_train + inference_core；用最小 stub 避免真依赖
    at = sys.modules.get("anima_train")
    if at is None or not hasattr(at, "sample_image"):
        at = types.ModuleType("anima_train")
        at.sample_image = lambda *a, **k: None  # 测试里用 monkeypatch 替换
        sys.modules["anima_train"] = at
    if "studio.services.inference_core" not in sys.modules:
        ic = types.ModuleType("studio.services.inference_core")

        class _LoRASpec:
            def __init__(self, path: str, scale: float = 1.0):
                self.path = path
                self.scale = scale

        ic.LoRASpec = _LoRASpec
        ic.apply_loras = lambda *a, **k: []
        sys.modules["studio.services.inference_core"] = ic

    import importlib

    if "runtime.anima_generate" in sys.modules:
        del sys.modules["runtime.anima_generate"]
    if "anima_generate" in sys.modules:
        del sys.modules["anima_generate"]
    mod = importlib.import_module("anima_generate")
    return mod


def _make_fake_img(tmp: Path):
    """伪 PIL.Image：有 save(path) 写空文件即可（测试只看是否落盘 + 文件名）。"""
    fake = MagicMock()
    fake.save = lambda p: Path(p).write_bytes(b"")
    return fake


def _mock_sample_image(records: list, fake_img):
    """sample_image 替身：记录每次入参 + 返回伪图。"""
    def _stub(*args, **kwargs):
        records.append({
            "steps": kwargs.get("steps"),
            "cfg_scale": kwargs.get("cfg_scale"),
            "sampler_name": kwargs.get("sampler_name"),
            "prompt": kwargs.get("prompt"),
        })
        return fake_img
    return _stub


# ---------------------------------------------------------------------------
# _set_lora_multiplier
# ---------------------------------------------------------------------------


def test_set_lora_multiplier_updates_network_and_per_lora(gen_module) -> None:
    """network.multiplier + 每个 lora.multiplier 都被设到指定值。"""
    fake_lora_a = MagicMock()
    fake_lora_a.multiplier = 1.0
    fake_lora_b = MagicMock()
    fake_lora_b.multiplier = 1.0
    fake_network = MagicMock()
    fake_network.multiplier = 1.0
    fake_network.loras = [fake_lora_a, fake_lora_b]
    fake_adapter = MagicMock()
    fake_adapter.network = fake_network

    gen_module._set_lora_multiplier(fake_adapter, 0.5)

    assert fake_network.multiplier == 0.5
    assert fake_lora_a.multiplier == 0.5
    assert fake_lora_b.multiplier == 0.5


def test_set_lora_multiplier_handles_no_network(gen_module) -> None:
    """adapter.network=None 时不报错。"""
    fake_adapter = MagicMock()
    fake_adapter.network = None
    gen_module._set_lora_multiplier(fake_adapter, 0.5)  # noop


# ---------------------------------------------------------------------------
# _apply_axis
# ---------------------------------------------------------------------------


def test_apply_axis_steps(gen_module) -> None:
    s, c, sd, sm = gen_module._apply_axis(
        {"axis": "steps"}, 30,
        cur_steps=25, cur_cfg_scale=4.0, cur_seed=42, cur_sampler="er_sde",
        base_specs=[], adapters=[],
    )
    assert s == 30
    assert (c, sd, sm) == (4.0, 42, "er_sde")


def test_apply_axis_cfg_scale_and_sampler(gen_module) -> None:
    s, c, sd, sm = gen_module._apply_axis(
        {"axis": "cfg_scale"}, 7.5,
        cur_steps=25, cur_cfg_scale=4.0, cur_seed=42, cur_sampler="er_sde",
        base_specs=[], adapters=[],
    )
    assert c == 7.5
    s, c, sd, sm = gen_module._apply_axis(
        {"axis": "sampler_name"}, "euler_a",
        cur_steps=s, cur_cfg_scale=c, cur_seed=sd, cur_sampler=sm,
        base_specs=[], adapters=[],
    )
    assert sm == "euler_a"


def test_apply_axis_lora_scale_mutates_adapter(gen_module) -> None:
    fake_network = MagicMock()
    fake_network.multiplier = 1.0
    fake_network.loras = []
    fake_adapter = MagicMock()
    fake_adapter.network = fake_network

    spec = MagicMock()
    spec.scale = 1.0
    gen_module._apply_axis(
        {"axis": "lora_scale", "lora_index": 0}, 0.7,
        cur_steps=25, cur_cfg_scale=4.0, cur_seed=42, cur_sampler="er_sde",
        base_specs=[spec], adapters=[fake_adapter],
    )
    assert fake_network.multiplier == 0.7


# ---------------------------------------------------------------------------
# _run_xy_matrix —— 遍历顺序 + 文件名 + monitor
# ---------------------------------------------------------------------------


def test_run_xy_matrix_x_only_no_y(gen_module, tmp_path, monkeypatch) -> None:
    """y=None 退化成 1×N（一行）。"""
    fake_img = _make_fake_img(tmp_path)
    records: list[dict] = []
    monkeypatch.setattr(gen_module._T, "sample_image", _mock_sample_image(records, fake_img))

    monitor_calls: list[dict] = []
    def fake_monitor(**kw):
        monitor_calls.append(kw)

    gen_module._run_xy_matrix(
        xy_matrix={"x": {"axis": "steps", "values": [20, 25, 30]}, "y": None},
        base_specs=[], adapters=[],
        prompt="test prompt",
        negative_prompt="",
        base_seed=42,
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=1024, width=1024,
        model=None, vae=None, qwen_model=None, qwen_tok=None, t5_tok=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=fake_monitor,
    )

    # 3 个 cell（X 3 值，Y=None）
    assert len(records) == 3
    # steps 按 X 值递增
    assert [r["steps"] for r in records] == [20, 25, 30]
    # 文件命名：yi=00 固定，xi 从 00 到 02，seed=42
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [
        "xy_x00_y00_s42.png",
        "xy_x01_y00_s42.png",
        "xy_x02_y00_s42.png",
    ]
    # monitor 收到 xy={xi,yi,xv,yv} 元数据
    assert len(monitor_calls) == 3
    assert monitor_calls[0]["xy"] == {"xi": 0, "yi": 0, "xv": 20, "yv": None}
    assert monitor_calls[2]["xy"] == {"xi": 2, "yi": 0, "xv": 30, "yv": None}


def test_run_xy_matrix_2d_traversal_order(gen_module, tmp_path, monkeypatch) -> None:
    """2×3 网格按 (yi 外, xi 内) 遍历 = 6 cells。"""
    fake_img = _make_fake_img(tmp_path)
    records: list[dict] = []
    monkeypatch.setattr(gen_module._T, "sample_image", _mock_sample_image(records, fake_img))

    gen_module._run_xy_matrix(
        xy_matrix={
            "x": {"axis": "steps", "values": [10, 20, 30]},
            "y": {"axis": "cfg_scale", "values": [3.0, 5.0]},
        },
        base_specs=[], adapters=[],
        prompt="p", negative_prompt="",
        base_seed=7,
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=512, width=512,
        model=None, vae=None, qwen_model=None, qwen_tok=None, t5_tok=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=None,
    )

    assert len(records) == 6
    # 顺序：y0(3.0) 的全部 x 跑完，再 y1(5.0)
    expected = [
        (10, 3.0), (20, 3.0), (30, 3.0),
        (10, 5.0), (20, 5.0), (30, 5.0),
    ]
    actual = [(r["steps"], r["cfg_scale"]) for r in records]
    assert actual == expected


def test_run_xy_matrix_lora_scale_resets_each_cell(gen_module, tmp_path, monkeypatch) -> None:
    """每个 cell 进入前 multiplier 重置到 base_scale，再按 axis 值改。"""
    fake_img = _make_fake_img(tmp_path)
    monkeypatch.setattr(
        gen_module._T, "sample_image",
        lambda *a, **k: fake_img,
    )

    fake_network = MagicMock()
    fake_network.loras = []
    fake_network.multiplier = 1.0
    fake_adapter = MagicMock()
    fake_adapter.network = fake_network

    multiplier_history: list[float] = []
    def _track_multiplier(value):
        multiplier_history.append(float(value))
    type(fake_network).multiplier = property(
        lambda self: multiplier_history[-1] if multiplier_history else 1.0,
        lambda self, v: _track_multiplier(v),
    )

    spec = MagicMock()
    spec.scale = 0.6  # base scale

    gen_module._run_xy_matrix(
        xy_matrix={
            "x": {"axis": "lora_scale", "values": [0.3, 0.9], "lora_index": 0},
            "y": None,
        },
        base_specs=[spec], adapters=[fake_adapter],
        prompt="p", negative_prompt="",
        base_seed=42,
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=512, width=512,
        model=None, vae=None, qwen_model=None, qwen_tok=None, t5_tok=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=None,
    )

    # 每个 cell：先重置到 base 0.6 → 再改成 axis 值
    # Cell 0 (xv=0.3): 0.6 → 0.3
    # Cell 1 (xv=0.9): 0.6 → 0.9
    assert 0.6 in multiplier_history       # base 重置至少一次
    assert 0.3 in multiplier_history
    assert 0.9 in multiplier_history
    # 重置必须发生在改值前 — 0.6 出现至少 2 次（每个 cell 一次）
    assert multiplier_history.count(0.6) >= 2


def test_run_xy_matrix_seed_axis_overrides_base(gen_module, tmp_path, monkeypatch) -> None:
    """axis=seed 时，cell 文件名用 axis 值而非 base_seed。"""
    fake_img = _make_fake_img(tmp_path)
    monkeypatch.setattr(gen_module._T, "sample_image", lambda *a, **k: fake_img)

    gen_module._run_xy_matrix(
        xy_matrix={"x": {"axis": "seed", "values": [100, 200, 300]}, "y": None},
        base_specs=[], adapters=[],
        prompt="p", negative_prompt="",
        base_seed=42,  # base 是 42 但 axis=seed 应覆盖
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=512, width=512,
        model=None, vae=None, qwen_model=None, qwen_tok=None, t5_tok=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=None,
    )

    files = sorted(p.name for p in tmp_path.iterdir())
    # 文件名包含 axis 值，不是 base_seed=42
    assert files == [
        "xy_x00_y00_s100.png",
        "xy_x01_y00_s200.png",
        "xy_x02_y00_s300.png",
    ]


def test_run_xy_matrix_zero_seed_randomizes_once(gen_module, tmp_path, monkeypatch) -> None:
    """base_seed=0 → 随机一次后所有 cell 共享。"""
    fake_img = _make_fake_img(tmp_path)
    monkeypatch.setattr(gen_module._T, "sample_image", lambda *a, **k: fake_img)
    # 固定 random.randint 返回值便于断言
    monkeypatch.setattr(gen_module.random, "randint", lambda a, b: 12345)

    gen_module._run_xy_matrix(
        xy_matrix={"x": {"axis": "steps", "values": [20, 25]}, "y": None},
        base_specs=[], adapters=[],
        prompt="p", negative_prompt="",
        base_seed=0,  # 触发 random
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=512, width=512,
        model=None, vae=None, qwen_model=None, qwen_tok=None, t5_tok=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=None,
    )

    files = sorted(p.name for p in tmp_path.iterdir())
    # 所有 cell 用同一个随机后的 seed 12345
    assert files == ["xy_x00_y00_s12345.png", "xy_x01_y00_s12345.png"]


def test_run_xy_matrix_skips_failing_cell(gen_module, tmp_path, monkeypatch) -> None:
    """单 cell 失败不影响其他 cell（容错）。"""
    fake_img = _make_fake_img(tmp_path)
    call_count = {"n": 0}

    def flaky_sample_image(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("CUDA OOM 模拟")
        return fake_img

    monkeypatch.setattr(gen_module._T, "sample_image", flaky_sample_image)

    gen_module._run_xy_matrix(
        xy_matrix={"x": {"axis": "steps", "values": [10, 20, 30]}, "y": None},
        base_specs=[], adapters=[],
        prompt="p", negative_prompt="",
        base_seed=42,
        base_steps=25, base_cfg_scale=4.0, base_sampler="er_sde",
        scheduler="simple",
        height=512, width=512,
        model=None, vae=None, qwen_model=None, qwen_tok=None, t5_tok=None,
        device="cpu", dtype=None,
        output_dir=tmp_path,
        update_monitor=None,
    )

    # 仅 cell 0 + cell 2 落盘（cell 1 抛了）
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["xy_x00_y00_s42.png", "xy_x02_y00_s42.png"]
