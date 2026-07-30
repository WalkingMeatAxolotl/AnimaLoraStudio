from __future__ import annotations

from training.families.anima.loader import (
    _find_comfyui_root,
    _resolve_qwen_tokenizer_path,
    _resolve_t5_tokenizer_path,
)


def test_discovers_tokenizers_from_comfyui_single_file(tmp_path) -> None:
    comfy = tmp_path / "ComfyUI"
    checkpoint = comfy / "models" / "clip" / "anima.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    qwen = comfy / "comfy" / "text_encoders" / "qwen25_tokenizer"
    t5 = comfy / "comfy" / "text_encoders" / "t5_tokenizer"
    qwen.mkdir(parents=True)
    t5.mkdir(parents=True)
    (qwen / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (t5 / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    assert _find_comfyui_root(checkpoint) == comfy
    assert _resolve_qwen_tokenizer_path(checkpoint) == qwen
    assert _resolve_t5_tokenizer_path("", checkpoint) == t5


def test_explicit_hf_qwen_directory_wins(tmp_path) -> None:
    qwen = tmp_path / "Qwen3"
    qwen.mkdir()
    (qwen / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert _resolve_qwen_tokenizer_path(qwen) == qwen

