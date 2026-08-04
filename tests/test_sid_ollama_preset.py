from __future__ import annotations

from studio.infrastructure.llm_presets import builtin_llm_presets
from studio.infrastructure.secrets import LLMMessage
from studio.services.tagging.llm import _apply_tags


def test_ollama_and_sid_presets_are_builtin() -> None:
    presets = {item["id"]: item for item in builtin_llm_presets()}
    assert presets["joycaption_ollama"]["base_url"] == "http://localhost:11434/v1"
    assert presets["sid_subject_json"]["output_format"] == "json"
    assert presets["sid_subject_json"]["model"] == "llama-joycaption-beta-one-hf-llava"


def test_llm_request_placeholders_include_class_word() -> None:
    messages = [
        LLMMessage(role="user", type="text", content="class={{class_word}} tags={{tags}}")
    ]
    rendered = _apply_tags(messages, "blue eyes", "1girl")
    assert rendered[0].content == "class=1girl tags=blue eyes"
    assert messages[0].content == "class={{class_word}} tags={{tags}}"
