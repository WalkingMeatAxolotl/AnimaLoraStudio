"""anima_reg_ai 的 JSON caption filter + sidecar 重写单测（PR #185 follow-up）。

校验 _rewrite_json_caption_for_prompt 走 normalize_caption_json + 单层过滤后：
- meta.trigger 永远从 reg sidecar + prompt 里去掉（base prior 不带 LoRA handle）
- excluded_tags 在 space/underscore 两种形态下都能命中
- documented_full / 简化 shape 全部折叠成标准 shape 写回（reg 是派生产物）
- scalar 字段含逗号时按单 tag 过 excluded
- nl 自然语言保留到 prompt 末尾
- _clear_reg_dir 复用自 reg_builder.clear_reg_dir（不再本地复刻）

不跑 GPU，不触发 anima_train 真 import — top-level stub。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def reg_module():
    """import runtime.anima_reg_ai 一次复用，stub 掉 anima_train 重依赖。"""
    if "anima_train" not in sys.modules or not hasattr(sys.modules["anima_train"], "sample_image"):
        at = types.ModuleType("anima_train")
        at.sample_image = lambda *a, **k: None
        at.find_diffusion_pipe_root = lambda: Path(".")
        at.resolve_path_best_effort = lambda p, bases: p
        at.load_anima_model = lambda *a, **k: None
        at.load_vae = lambda *a, **k: None
        at.load_text_encoders = lambda *a, **k: (None, None, None)
        at.enable_xformers = lambda *a, **k: None
        sys.modules["anima_train"] = at

    import importlib
    if "anima_reg_ai" in sys.modules:
        del sys.modules["anima_reg_ai"]
    return importlib.import_module("anima_reg_ai")


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cap.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# trigger 不进 reg
# ---------------------------------------------------------------------------

def test_drops_meta_trigger_from_prompt_and_sidecar(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "meta": {"trigger": "aoirikko"},
        "tags": {
            "quality": ["masterpiece"],
            "appearance": ["brown hair"],
        },
    })
    prompt = reg_module._rewrite_json_caption_for_prompt(src, set())
    assert "aoirikko" not in prompt.lower()
    assert "brown hair" in prompt
    on_disk = _read(src)
    assert on_disk["meta"].get("trigger") in (None, "")


def test_drops_trigger_even_when_other_meta_kept(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "meta": {"trigger": "aoirikko", "tagger_version": "v2"},
        "tags": {"appearance": ["smile"]},
    })
    reg_module._rewrite_json_caption_for_prompt(src, set())
    on_disk = _read(src)
    assert "trigger" not in on_disk["meta"]
    assert on_disk["meta"].get("tagger_version") == "v2"


# ---------------------------------------------------------------------------
# excluded space / underscore 等价
# ---------------------------------------------------------------------------

def test_excluded_underscore_matches_space_form(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "tags": {"appearance": ["brown hair", "blue eyes"]},
    })
    # excluded_tags 传 underscore，命中 JSON 里的 space 形态
    excluded = {reg_module._tag_key("brown_hair")}
    prompt = reg_module._rewrite_json_caption_for_prompt(src, excluded)
    assert "brown hair" not in prompt
    assert "blue eyes" in prompt


def test_excluded_space_matches_underscore_form(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "tags": {"appearance": ["brown_hair", "blue_eyes"]},
    })
    excluded = {reg_module._tag_key("brown hair")}
    prompt = reg_module._rewrite_json_caption_for_prompt(src, excluded)
    assert "brown" not in prompt or "hair" not in prompt
    # blue_eyes 经 normalize 会被 split_tags 收敛到 "blue_eyes" 一项；
    # 取一个轻断言：另一个 tag 仍在
    assert "blue" in prompt


# ---------------------------------------------------------------------------
# shape 折叠：documented_full → standard
# ---------------------------------------------------------------------------

def test_documented_full_shape_folds_to_standard_on_disk(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "fixed": {"quality": "masterpiece, best quality", "series": "kaguya", "artist": ""},
        "character": {"name": "Misaka", "variant": "Sisters", "full": "Misaka Mikoto"},
        "ai_output": {
            "count": "1girl",
            "appearance": ["brown hair"],
            "tags": ["sitting"],
            "environment": ["indoors"],
            "nl": "She is reading a book.",
        },
        "from_path": {"appearance": ["smile"], "extra_tags": ["solo"]},
    })
    prompt = reg_module._rewrite_json_caption_for_prompt(src, set())
    on_disk = _read(src)

    # 落盘 shape：标准 shape，原 fixed/character/ai_output/from_path 顶层 key 全没了
    assert isinstance(on_disk.get("tags"), dict)
    assert "fixed" not in on_disk
    assert "ai_output" not in on_disk
    assert "from_path" not in on_disk
    assert "character" not in on_disk  # character 折到 tags.character

    # prompt：character 折成 full、appearance / tags 合并 from_path 部分、nl 接在末尾
    assert "misaka mikoto" in prompt.lower()
    assert "brown hair" in prompt
    assert "smile" in prompt
    assert "sitting" in prompt
    assert "solo" in prompt
    assert "She is reading a book." in prompt


def test_simplified_tags_list_top_level_shape(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "tags": ["1girl", "brown hair", "smile"],
        "meta": {"trigger": "aoirikko"},
    })
    prompt = reg_module._rewrite_json_caption_for_prompt(src, set())
    assert "aoirikko" not in prompt.lower()
    assert "1girl" in prompt
    assert "brown hair" in prompt
    on_disk = _read(src)
    assert isinstance(on_disk["tags"], dict)  # 折成标准 shape
    assert "1girl" in on_disk["tags"]["tags"]


# ---------------------------------------------------------------------------
# character / scalar 字段
# ---------------------------------------------------------------------------

def test_character_dict_full_name_excluded_drops_character(reg_module, tmp_path: Path) -> None:
    """character 是 dict 形态时折叠到 full 字符串，excluded 命中 full 名直接清空。"""
    src = _write(tmp_path, {
        "fixed": {"quality": "best quality"},
        "character": {"name": "Misaka", "variant": "", "full": "Misaka Mikoto"},
        "ai_output": {"tags": [], "appearance": [], "environment": []},
    })
    excluded = {reg_module._tag_key("Misaka Mikoto")}
    prompt = reg_module._rewrite_json_caption_for_prompt(src, excluded)
    assert "misaka" not in prompt.lower()
    assert "best quality" in prompt


def test_scalar_field_with_comma_per_tag_exclude(reg_module, tmp_path: Path) -> None:
    """count="1girl, 1boy" 里 exclude="1boy" 只删 1boy 一个 token。"""
    src = _write(tmp_path, {
        "tags": {"count": "1girl, 1boy", "appearance": []},
    })
    excluded = {reg_module._tag_key("1boy")}
    prompt = reg_module._rewrite_json_caption_for_prompt(src, excluded)
    assert "1girl" in prompt
    assert "1boy" not in prompt
    on_disk = _read(src)
    assert "1boy" not in on_disk["tags"]["count"]
    assert "1girl" in on_disk["tags"]["count"]


# ---------------------------------------------------------------------------
# nl 自然语言保留
# ---------------------------------------------------------------------------

def test_nl_preserved_at_prompt_tail(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "tags": {
            "appearance": ["brown hair"],
            "nl": "She is wearing a school uniform.",
        },
    })
    prompt = reg_module._rewrite_json_caption_for_prompt(src, set())
    assert prompt.endswith("She is wearing a school uniform.")
    assert "brown hair" in prompt


# ---------------------------------------------------------------------------
# 复用：clear_reg_dir 来自 reg_builder
# ---------------------------------------------------------------------------

def test_clear_reg_dir_is_reused_from_reg_builder(reg_module) -> None:
    from studio.services.reg.builder import clear_reg_dir as upstream
    assert reg_module.clear_reg_dir is upstream
    # 同时确认本地不再定义私有复制
    assert not hasattr(reg_module, "_clear_reg_dir")


# ---------------------------------------------------------------------------
# txt 路径不退化：单 .txt caption 仍按 space-form 写回，trigger 由 .txt 内容决定
# ---------------------------------------------------------------------------

def test_txt_caption_path_unchanged(reg_module, tmp_path: Path) -> None:
    src = tmp_path / "cap.txt"
    src.write_text("brown_hair, blue eyes, 1girl", encoding="utf-8")
    excluded = {reg_module._tag_key("1girl")}
    prompt = reg_module._rewrite_caption_for_prompt(src, excluded)
    assert "brown hair" in prompt  # underscore → space 归一
    assert "blue eyes" in prompt
    assert "1girl" not in prompt
    on_disk = src.read_text(encoding="utf-8")
    assert on_disk == prompt
