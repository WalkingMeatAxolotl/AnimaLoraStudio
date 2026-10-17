"""int8 等整数量化版底模 / TE（ComfyUI 逐层量化格式）的识别与报错。

真实文件形态（Bedovyy/Anima-INT8、Comfy-Org *_int8_convrot 等）：``__metadata__``
为空；每个量化 Linear 有 ``{layer}.weight``(I8) + ``{layer}.weight_scale``(F32)
+ ``{layer}.comfy_quant``(U8 JSON blob，如 ``{"format": "int8_tensorwise",
"convrot": true}``)。此前三条加载路径都以误导文案报错（「层数推断错」「结构
指纹不匹配」「权重不是 fp8」）；studio 扫描还会给它标层数。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from studio.infrastructure import log_messages
from studio.services.inference.checkpoint_arch import (
    UnsupportedQuantizationError,
    detect_unsupported_quant,
    inspect_anima_checkpoint,
    read_safetensors_header,
)

_CONVROT_CONF = {
    "format": "int8_tensorwise", "per_row": True,
    "convrot": True, "convrot_groupsize": 256,
}


@pytest.fixture(autouse=True)
def _zh(monkeypatch):
    monkeypatch.setenv(log_messages.UI_LANG_ENV, "zh")


def _blob(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _quantize_int8(sd: dict[str, torch.Tensor], layers: list[str], conf: dict | None) -> dict:
    """把 ``layers`` 的 ``.weight`` 换成 Comfy int8 形态（I8 + weight_scale [+ comfy_quant]）。"""
    out = dict(sd)
    for layer in layers:
        w = out[f"{layer}.weight"]
        out[f"{layer}.weight"] = torch.zeros(w.shape, dtype=torch.int8)
        out[f"{layer}.weight_scale"] = torch.ones(w.shape[0], 1, dtype=torch.float32)
        if conf is not None:
            out[f"{layer}.comfy_quant"] = _blob(conf)
    return out


def _anima_int8_file(path: Path, conf: dict | None = _CONVROT_CONF) -> Path:
    sd = {
        "net.x_embedder.proj.1.weight": torch.zeros(8, 68, dtype=torch.bfloat16),
        "net.final_layer.linear.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
    }
    for i in range(3):
        sd[f"net.blocks.{i}.self_attn.q_proj.weight"] = torch.zeros(8, 8, dtype=torch.bfloat16)
    layers = [f"net.blocks.{i}.self_attn.q_proj" for i in range(3)]
    save_file(_quantize_int8(sd, layers, conf), str(path))
    return path


# ── 纯判定 ────────────────────────────────────────────────────────────────

def test_detect_reads_format_from_comfy_quant_blob(tmp_path):
    p = _anima_int8_file(tmp_path / "anima-int8convrot.safetensors")
    header = read_safetensors_header(p)

    found = detect_unsupported_quant(header, path=p)
    assert found is not None
    assert found.format_label == "int8_tensorwise + ConvRot"
    assert found.layer_count == 3
    # 不给 path（只有 header）退回 dtype 名
    assert detect_unsupported_quant(header).format_label == "int8"


def test_detect_without_blob_falls_back_to_dtype(tmp_path):
    p = _anima_int8_file(tmp_path / "anima-int8.safetensors", conf=None)
    assert detect_unsupported_quant(read_safetensors_header(p), path=p).format_label == "int8"


def test_comfy_fp8_with_comfy_quant_is_not_flagged():
    """Comfy 新式 fp8 文件同样带 comfy_quant 键——那是支持的，判据只看 dtype。"""
    header = {
        "__metadata__": {},
        "blocks.0.q.weight": {"dtype": "F8_E4M3", "shape": [8, 8], "data_offsets": [0, 64]},
        "blocks.0.q.weight_scale": {"dtype": "F32", "shape": [], "data_offsets": [64, 68]},
        "blocks.0.q.comfy_quant": {"dtype": "U8", "shape": [32], "data_offsets": [68, 100]},
        "blocks.0.norm.weight": {"dtype": "BF16", "shape": [8], "data_offsets": [100, 116]},
    }
    assert detect_unsupported_quant(header) is None


# ── 路径 1：Anima（loader 第一步就是 inspect_anima_checkpoint）────────────

def test_anima_inspect_rejects_int8_with_clear_message(tmp_path):
    p = _anima_int8_file(tmp_path / "anima-int8convrot.safetensors")
    with pytest.raises(UnsupportedQuantizationError) as ei:
        inspect_anima_checkpoint(p)
    text = str(ei.value)
    assert "anima-int8convrot.safetensors" in text
    assert "int8_tensorwise + ConvRot" in text
    assert "bf16 或 fp8" in text
    assert "层数" not in text  # 不再是「层数推断错」的误导文案


def test_message_follows_ui_language(tmp_path, monkeypatch):
    monkeypatch.setenv(log_messages.UI_LANG_ENV, "en")
    p = _anima_int8_file(tmp_path / "a.safetensors")
    with pytest.raises(UnsupportedQuantizationError, match="bf16 or fp8 release"):
        inspect_anima_checkpoint(p)


def test_studio_scan_marks_int8_unsupported(tmp_path):
    from studio.services.models.families.custom_paths import arch_summary, clear_arch_cache

    clear_arch_cache()
    try:
        p = _anima_int8_file(tmp_path / "anima-int8convrot.safetensors")
        assert arch_summary(p) == {"unsupported_quant": "int8_tensorwise + ConvRot"}
    finally:
        clear_arch_cache()


# ── 路径 2：Krea2 DiT ─────────────────────────────────────────────────────

def test_krea2_inspect_rejects_int8_with_clear_message(tmp_path):
    from tests.test_krea2_loader import _state_dict, _tiny_config
    from training.families.krea2.loader import inspect_krea2_checkpoint, load_krea2_model

    config = _tiny_config()
    sd = _state_dict(config)
    layers = [k[: -len(".weight")] for k, v in sd.items() if k.endswith(".weight") and v.ndim == 2][:2]
    p = tmp_path / "krea2-raw-int8-convrot.safetensors"
    save_file(_quantize_int8(sd, layers, _CONVROT_CONF), str(p))

    calls = [
        lambda: inspect_krea2_checkpoint(p, config=config),
        # 训练 / 出图两个调用面
        lambda: load_krea2_model(p, "cpu", torch.bfloat16, config=config, purpose="train"),
        lambda: load_krea2_model(p, "cpu", torch.bfloat16, config=config, purpose="generate"),
    ]
    for call in calls:
        with pytest.raises(UnsupportedQuantizationError) as ei:
            call()
        text = str(ei.value)
        assert "int8_tensorwise + ConvRot" in text and "bf16 或 fp8" in text
        assert "结构指纹" not in text


# ── 路径 3：Krea2 TE comfy 单文件 ─────────────────────────────────────────

def test_krea2_single_file_te_rejects_int8_with_clear_message(tmp_path):
    pytest.importorskip("transformers")
    import training.families.krea2.text_encoding as te_mod

    te_dir = tmp_path / "qwen3vl-int8"
    te_dir.mkdir()
    sd = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.zeros(16, 8, dtype=torch.bfloat16),
    }
    save_file(
        _quantize_int8(sd, ["model.layers.0.self_attn.q_proj"], _CONVROT_CONF),
        str(te_dir / "qwen_3_vl_4b_int8_convrot.safetensors"),
    )
    # 在读 config / 建模型之前就拦下（目录里没有 config.json 也照样给出 int8 文案）
    with pytest.raises(UnsupportedQuantizationError) as ei:
        te_mod._default_model_loader(te_dir, torch.device("cpu"), torch.bfloat16)
    text = str(ei.value)
    assert "qwen_3_vl_4b_int8_convrot.safetensors" in text
    assert "int8_tensorwise + ConvRot" in text
    assert "不是 fp8" not in text
