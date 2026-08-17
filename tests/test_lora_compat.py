"""LoRA ↔ 底模兼容契约（studio.services.inference.lora_compat）+ 三处消费。

同族不同层数（Anima 28 层 vs 第三方插层扩展 40 层）的 LoRA 按 blocks.N 名字
套用会静默错位；族标记分不出。契约 = 写（元数据）/ 读（键扫描兜底）/ 判（一个
纯函数），出图 apply、训练 resume_lora、恢复点三处消费同一判定。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from studio.services.inference.lora_compat import (
    KEY_BASE_MODEL_CHANNELS,
    KEY_BASE_MODEL_FILE,
    KEY_BASE_NUM_BLOCKS,
    LoraBaseArch,
    base_arch_network_args,
    base_arch_network_args_from_model,
    build_lora_metadata,
    check_lora_compat,
    lora_base_arch,
    lora_num_blocks_from_keys,
    model_num_blocks,
    read_lora_base_arch,
)


# ── 写 ──────────────────────────────────────────────────────────────────────

def test_base_arch_network_args_from_model_prefers_checkpoint_arch():
    model = SimpleNamespace(
        checkpoint_arch=SimpleNamespace(num_blocks=40, model_channels=2048),
        blocks=[object()] * 3,  # 与 arch 不一致时以 arch 为准（arch 是 header 真相）
    )
    args = base_arch_network_args_from_model(model, r"C:\models\Anima-2.9B-preview-v1.safetensors")
    assert args == {
        KEY_BASE_NUM_BLOCKS: 40,
        KEY_BASE_MODEL_CHANNELS: 2048,
        KEY_BASE_MODEL_FILE: "Anima-2.9B-preview-v1.safetensors",
    }


def test_base_arch_network_args_from_model_falls_back_to_blocks():
    """没有 checkpoint_arch 的族（krea2）：数 model.blocks；没有 blocks → 不写层数。"""
    model = SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(28)]))
    args = base_arch_network_args_from_model(model)
    assert args == {KEY_BASE_NUM_BLOCKS: 28}
    assert base_arch_network_args_from_model(object()) == {}
    assert base_arch_network_args(num_blocks=None) == {}


def test_build_lora_metadata_shape():
    meta = build_lora_metadata(
        rank=16, alpha=8.0, network_args={"algo": "lokr", KEY_BASE_NUM_BLOCKS: 40},
        extra={"anima_merge_sources": "[]"},
    )
    assert meta["ss_network_dim"] == "16"
    assert meta["ss_network_alpha"] == "8.0"
    assert meta["ss_network_module"] == "lycoris.kohya"
    assert json.loads(meta["ss_network_args"]) == {"algo": "lokr", KEY_BASE_NUM_BLOCKS: 40}
    assert meta["anima_merge_sources"] == "[]"
    assert all(isinstance(v, str) for v in meta.values())  # safetensors metadata 只收 str


# ── 读 ──────────────────────────────────────────────────────────────────────

def test_lora_num_blocks_from_keys_kohya_and_peft_and_adapter_exclusion():
    kohya = [
        "lora_unet_blocks_0_self_attn_q_proj.lokr_w1",
        "lora_unet_blocks_27_mlp_layer2.lokr_w2_a",
        "lora_unet_blocks_27_mlp_layer2.alpha",
    ]
    assert lora_num_blocks_from_keys(kohya) == 28
    peft = ["diffusion_model.blocks.39.self_attn.q_proj.lora_A.weight"]
    assert lora_num_blocks_from_keys(peft) == 40
    # llm_adapter 自己的 blocks 不算
    assert lora_num_blocks_from_keys(["lora_unet_llm_adapter_blocks_5_mlp_0.lora_down.weight"]) is None
    assert lora_num_blocks_from_keys(["lora_unet_final_layer_linear.lora_down.weight"]) is None
    assert lora_num_blocks_from_keys([]) is None


def test_lora_base_arch_metadata_wins_over_keys():
    args = {KEY_BASE_NUM_BLOCKS: 40, KEY_BASE_MODEL_CHANNELS: 2048, KEY_BASE_MODEL_FILE: "x.safetensors"}
    keys = ["lora_unet_blocks_3_self_attn_q_proj.lokr_w1"]  # 键只到 block 3
    arch = lora_base_arch(args, keys)
    assert arch == LoraBaseArch(40, "metadata", 2048, "x.safetensors")
    assert arch.explicit
    # 无元数据 → 键扫描（下界）
    assert lora_base_arch({}, keys) == LoraBaseArch(4, "keys")
    # 都没有 → unknown
    assert lora_base_arch({}, []) == LoraBaseArch(None, "unknown")
    # 元数据坏值 → 退回键
    assert lora_base_arch({KEY_BASE_NUM_BLOCKS: "abc"}, keys).source == "keys"


def test_read_lora_base_arch_header_only(tmp_path):
    p = tmp_path / "lora.safetensors"
    save_file(
        {"lora_unet_blocks_27_self_attn_q_proj.lokr_w1": torch.zeros(2, 2)},
        str(p),
        metadata={"ss_network_args": json.dumps({"algo": "lokr", KEY_BASE_NUM_BLOCKS: 28})},
    )
    assert read_lora_base_arch(p) == LoraBaseArch(28, "metadata")
    # 存量文件（无契约键）→ 键扫描
    q = tmp_path / "legacy.safetensors"
    save_file({"lora_unet_blocks_27_self_attn_q_proj.lokr_w1": torch.zeros(2, 2)}, str(q))
    assert read_lora_base_arch(q) == LoraBaseArch(28, "keys")
    # 坏文件 → unknown，不抛
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"nope")
    assert read_lora_base_arch(bad).source == "unknown"


def test_model_num_blocks_only_trusts_real_ints():
    from unittest.mock import MagicMock

    assert model_num_blocks(SimpleNamespace(checkpoint_arch=SimpleNamespace(num_blocks=40))) == 40
    assert model_num_blocks(SimpleNamespace(blocks=[1, 2, 3])) == 3
    assert model_num_blocks(object()) is None
    assert model_num_blocks(MagicMock()) is None  # 替身对象不判


# ── 判 ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "lora, model_blocks, level",
    [
        (LoraBaseArch(28, "metadata"), 40, "reject"),   # 元数据确证不等
        (LoraBaseArch(40, "metadata"), 28, "reject"),
        (LoraBaseArch(40, "metadata"), 40, "ok"),
        (LoraBaseArch(40, "keys"), 28, "reject"),       # 键必然越界
        (LoraBaseArch(28, "keys"), 40, "warn"),         # 键只是下界：可能老 LoRA / 部分层
        (LoraBaseArch(28, "keys"), 28, "ok"),
        (LoraBaseArch(None, "unknown"), 40, "ok"),      # 一无所知不判
        (LoraBaseArch(28, "metadata"), None, "ok"),     # 底模层数未知不判
    ],
)
def test_check_lora_compat_rules(lora, model_blocks, level):
    verdict = check_lora_compat(lora, model_blocks, lora_name="x.safetensors")
    assert verdict.level == level
    if level != "ok":
        assert "x.safetensors" in verdict.reason and "层" in verdict.reason


# ── 三处消费 ─────────────────────────────────────────────────────────────────

class _TinyDiT(torch.nn.Module):
    def __init__(self, num_blocks: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(num_blocks)])


def _write_lora(path: Path, *, num_blocks_meta=None, key_block: int = 0) -> Path:
    args = {"algo": "lokr", "factor": 8, "model_family": "anima"}
    if num_blocks_meta is not None:
        args[KEY_BASE_NUM_BLOCKS] = num_blocks_meta
    save_file(
        {f"lora_unet_blocks_{key_block}_self_attn_q_proj.lokr_w1": torch.zeros(2, 2)},
        str(path),
        metadata={"ss_network_dim": "8", "ss_network_alpha": "8", "ss_network_args": json.dumps(args)},
    )
    return path


def test_apply_loras_rejects_metadata_mismatch_and_warns_on_key_lower_bound(tmp_path):
    """出图侧：28 层 LoRA（元数据确证）挂 40 层底模 → 拒绝；无元数据只到 block 27
    的存量文件 → 警告回调（放行）。"""
    from studio.services.inference.core import LoRASpec, apply_loras

    model = _TinyDiT(40)
    p_reject = _write_lora(tmp_path / "old28.safetensors", num_blocks_meta=28)
    with pytest.raises(ValueError, match="层数不匹配"):
        apply_loras(model, [LoRASpec(path=str(p_reject))], "cpu", torch.float32, family_id="anima")

    p_warn = _write_lora(tmp_path / "legacy.safetensors", key_block=27)
    warnings: list[str] = []
    # 走到 inject 前就该 warn；用 fake adapter 截住后续（不依赖 lycoris）
    import sys
    import types
    from unittest.mock import MagicMock, patch

    def _fake_adapter(*a, **k):
        m = MagicMock()
        m.network = MagicMock()
        m.network.loras = []
        m.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        return m

    fake_mod = types.ModuleType("utils.lycoris_adapter")
    fake_mod.AnimaLycorisAdapter = _fake_adapter  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"utils.lycoris_adapter": fake_mod}):
        apply_loras(
            model, [LoRASpec(path=str(p_warn))], "cpu", torch.float32,
            family_id="anima", on_warning=warnings.append,
        )
    assert len(warnings) == 1 and "28" in warnings[0] and "40" in warnings[0]


def test_resume_lora_check_rejects_and_warns(tmp_path):
    from training.phases.models import _check_resume_lora_compat

    ctx = SimpleNamespace(
        model=_TinyDiT(40),
        family=SimpleNamespace(spec=SimpleNamespace(family_id="anima")),
    )
    with pytest.raises(RuntimeError, match="层数不匹配"):
        _check_resume_lora_compat(ctx, _write_lora(tmp_path / "old28.safetensors", num_blocks_meta=28))
    # 同层数放行
    _check_resume_lora_compat(ctx, _write_lora(tmp_path / "ok40.safetensors", num_blocks_meta=40))
    # 跨族仍拒绝
    k2 = tmp_path / "k2.safetensors"
    save_file({"x": torch.zeros(1)}, str(k2), metadata={
        "ss_network_args": json.dumps({"algo": "lokr", "model_family": "krea2"}),
    })
    with pytest.raises(RuntimeError, match="跨模型族"):
        _check_resume_lora_compat(ctx, k2)


def test_training_state_records_and_checks_base_num_blocks(tmp_path):
    from training.state import load_training_state, save_training_state

    class _Inj:
        def state_dict(self):
            return {"w": torch.zeros(1)}

        def load_state_dict(self, sd, strict=False):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.1)
    path = tmp_path / "state.pt"
    save_training_state(path, _Inj(), opt, epoch=1, global_step=10, base_num_blocks=28)
    # 同层数放行；不给 expected 也放行（老调用方）；不等 → 拒绝且说人话
    load_training_state(path, _Inj(), opt, expected_num_blocks=28)
    load_training_state(path, _Inj(), opt)
    with pytest.raises(RuntimeError, match="层数不匹配"):
        load_training_state(path, _Inj(), opt, expected_num_blocks=40)
    # 老恢复点（没记层数）→ 不判
    save_training_state(path, _Inj(), opt, epoch=1, global_step=10)
    load_training_state(path, _Inj(), opt, expected_num_blocks=40)


def test_lycoris_adapter_save_writes_base_arch_via_metadata_extra(tmp_path):
    """真 AnimaLycorisAdapter：metadata_extra 里的契约键随 ss_network_args 落盘。"""
    pytest.importorskip("lycoris")
    from studio.services.inference.core import read_lora_meta
    from utils.lycoris_adapter import AnimaLycorisAdapter

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
            for b in self.blocks:
                b.self_attn = torch.nn.Module()
                b.self_attn.q_proj = torch.nn.Linear(8, 8)

    model = _M()
    adapter = AnimaLycorisAdapter(
        preset={"target_name": ["*q_proj"], "exclude_name": [], "use_fnmatch": True, "lora_prefix": "lora_unet"},
        algo="lora", rank=2, alpha=2.0,
    )
    adapter.metadata_extra = {"model_family": "anima", **base_arch_network_args_from_model(model, "base.safetensors")}
    adapter.inject(model)
    out = tmp_path / "out.safetensors"
    adapter.save(out)
    meta = read_lora_meta(str(out))
    assert meta.base_arch == LoraBaseArch(2, "metadata", None, "base.safetensors")
