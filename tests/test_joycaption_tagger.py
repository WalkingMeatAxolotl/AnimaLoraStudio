"""JoyCaption backward-compat wrapper over LLM tagger preset.

JoyCaption 已合并到 LLM tagger 的 builtin preset；wrapper 仅为 `get_tagger("joycaption")`
旧调用方兜底。下面测试核对 wrapper 强制切到 joycaption preset 后调 LLMTagger 的行为。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from studio import secrets
from studio.services import joycaption_tagger, llm_tagger


@pytest.fixture
def isolated_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    # 改写 joycaption preset 的 endpoint+生成参数
    secrets.update(
        {
            "llm_tagger": {
                "presets": [
                    {
                        "id": "joycaption",
                        "base_url": "http://x/v1",
                        "model": "m",
                        "messages": [
                            {"type": "text", "role": "system", "content": "hi"},
                            {"type": "image"},
                        ],
                        "max_retries": 1,
                    }
                ]
            }
        }
    )
    return tmp_path


def _png(path: Path) -> Path:
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    return path


def _ok_response(content: str = "tag1, tag2"):
    r = MagicMock()
    r.ok = True
    r.status_code = 200
    r.text = ""
    r.raise_for_status = MagicMock()
    r.json = MagicMock(
        return_value={"choices": [{"message": {"content": content}}]}
    )
    return r


def test_is_available_ok(isolated_secrets) -> None:
    sess = MagicMock()
    t = joycaption_tagger.JoyCaptionTagger(session=sess)
    ok, msg = t.is_available()
    assert ok is True
    assert "m" in msg
    sess.get.assert_not_called()


def test_is_available_requires_model(isolated_secrets) -> None:
    secrets.update(
        {"llm_tagger": {"presets": [{"id": "joycaption", "model": ""}]}}
    )
    t = joycaption_tagger.JoyCaptionTagger(session=MagicMock())
    ok, msg = t.is_available()
    assert ok is False
    assert "model" in msg


def test_is_available_no_base_url(isolated_secrets) -> None:
    secrets.update(
        {"llm_tagger": {"presets": [{"id": "joycaption", "base_url": ""}]}}
    )
    t = joycaption_tagger.JoyCaptionTagger(session=MagicMock())
    ok, msg = t.is_available()
    assert ok is False
    assert "base_url" in msg


def test_tag_emits_natural_caption(isolated_secrets, tmp_path: Path) -> None:
    sess = MagicMock()
    sess.post.return_value = _ok_response("a sunny day")
    t = joycaption_tagger.JoyCaptionTagger(session=sess)
    img = _png(tmp_path / "1.png")
    [r] = list(t.tag([img]))
    # joycaption preset 默认 output_format=text → 整段返回直接是 caption
    assert r["tags"] == ["a sunny day"]
    args, kwargs = sess.post.call_args
    assert args[0] == "http://x/v1/chat/completions"
    body = kwargs["json"]
    assert body["model"] == "m"
    assert body["messages"][0]["content"] == "hi"
    # messages[1] 是 image item，铺开成 user/[image_url]
    user_content = body["messages"][1]["content"]
    assert user_content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_tag_retries_then_fails(isolated_secrets, tmp_path: Path, monkeypatch) -> None:
    """所有重试都失败 → 单图返回 error，但循环继续。"""
    secrets.update(
        {
            "llm_tagger": {
                "presets": [{"id": "joycaption", "max_retries": 2, "timeout": 1}]
            }
        }
    )
    sess = MagicMock()
    bad = MagicMock()
    bad.status_code = 500
    bad.text = "boom"
    sess.post.return_value = bad
    monkeypatch.setattr(llm_tagger.time, "sleep", lambda _: None)
    t = joycaption_tagger.JoyCaptionTagger(session=sess)
    img = _png(tmp_path / "1.png")
    [r] = list(t.tag([img]))
    assert r["tags"] == []
    assert "失败" in r["error"]
    assert sess.post.call_count == 2
