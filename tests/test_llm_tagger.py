"""LLM tagger: OpenAI-compatible Chat Completions + Responses payloads."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from studio import secrets
from studio.services import llm_tagger


@pytest.fixture
def isolated_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    secrets.update(
        {
            "llm_tagger": {
                "current_preset": "style_json",
                "presets": [
                    {
                        "id": "style_json",
                        "base_url": "http://x/v1",
                        "api_key": "k",
                        "model": "vision",
                        "endpoint": "chat_completions",
                        "max_retries": 1,
                    }
                ],
            }
        }
    )
    return tmp_path


def _png(path: Path) -> Path:
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    return path


def _chat_response(content: str):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


def _responses_response(content: str):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": content},
                ]
            }
        ]
    }
    return r


def _sse_response(*payloads: dict):
    r = MagicMock()
    r.status_code = 200
    r.headers = {"content-type": "text/event-stream"}
    r.text = "".join(
        f"data: {json.dumps(payload)}\n\n"
        for payload in payloads
    ) + "data: [DONE]\n\n"
    r.json.side_effect = AssertionError("SSE responses must not use resp.json()")
    return r


class _SlowSession:
    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.lock = threading.Lock()

    def post(self, *args, **kwargs):
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return _chat_response('{"tags":["ink"]}')
        finally:
            with self.lock:
                self.active -= 1


def test_is_available_requires_model(isolated_secrets) -> None:
    secrets.update(
        {"llm_tagger": {"presets": [{"id": "style_json", "model": ""}]}}
    )
    ok, msg = llm_tagger.LLMTagger(session=MagicMock()).is_available()
    assert ok is False
    assert "model" in msg


def test_chat_completions_tag_normalizes_json(isolated_secrets, tmp_path: Path) -> None:
    sess = MagicMock()
    sess.post.return_value = _chat_response(
        '{"count":"1girl","appearance":["long hair"],"tags":["watercolor"],'
        '"environment":["blue background"],"nl":"Soft style."}'
    )
    tagger = llm_tagger.LLMTagger(session=sess)
    img = _png(tmp_path / "1.png")

    [result] = list(tagger.tag([img]))

    assert result["tags"] == ["1girl", "long hair", "watercolor", "blue background"]
    assert result["caption"] == (
        "1girl, long hair, watercolor, blue background. Soft style."
    )
    assert result["caption_json"]["tags"]["appearance"] == ["long hair"]
    args, kwargs = sess.post.call_args
    assert args[0] == "http://x/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer k"
    body = kwargs["json"]
    assert body["model"] == "vision"
    assert "anime style LoRA" in body["messages"][0]["content"]
    assert body["stream"] is False
    assert kwargs["timeout"] == (10, 60)
    # messages[1] 是 image item，被铺开成 user/[image_url]
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_tag_emits_start_and_done_progress(isolated_secrets, tmp_path: Path) -> None:
    sess = MagicMock()
    sess.post.return_value = _chat_response('{"tags":["ink"]}')
    tagger = llm_tagger.LLMTagger(session=sess)
    img = _png(tmp_path / "1.png")
    progress: list[tuple[int, int]] = []

    list(tagger.tag([img], on_progress=lambda d, t: progress.append((d, t))))

    assert progress == [(0, 1), (1, 1)]


def test_chat_completions_tag_accepts_sse_delta_stream(
    isolated_secrets, tmp_path: Path
) -> None:
    sess = MagicMock()
    sess.post.return_value = _sse_response(
        {"choices": [{"delta": {"content": '{"tags":["ink"'}}]},
        {"choices": [{"delta": {"content": "]}"}}]},
    )
    tagger = llm_tagger.LLMTagger(session=sess)
    img = _png(tmp_path / "1.png")

    [result] = list(tagger.tag([img]))

    assert result["tags"] == ["ink"]
    assert result["caption"] == "ink"


def test_tag_uses_configured_concurrency(
    isolated_secrets, tmp_path: Path
) -> None:
    secrets.update(
        {"llm_tagger": {"presets": [{"id": "style_json", "concurrency": 3}]}}
    )
    sess = _SlowSession()
    tagger = llm_tagger.LLMTagger(session=sess)
    imgs = [_png(tmp_path / f"{i}.png") for i in range(4)]
    progress: list[tuple[int, int]] = []

    results = list(tagger.tag(imgs, on_progress=lambda d, t: progress.append((d, t))))

    assert len(results) == 4
    assert all(r["tags"] == ["ink"] for r in results)
    assert sess.calls == 4
    assert sess.max_active >= 2
    assert progress[0] == (0, 4)
    assert progress[-1] == (4, 4)


def test_minute_limit_applies_only_with_concurrency(
    isolated_secrets, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[tuple[float, int]] = []

    class _FakeLimiter:
        def __init__(
            self,
            requests_per_second: float,
            *,
            max_requests_per_minute: int = 0,
        ) -> None:
            waits.append((requests_per_second, max_requests_per_minute))

        def wait(self) -> None:
            pass

    monkeypatch.setattr(llm_tagger, "_RequestRateLimiter", _FakeLimiter)
    secrets.update(
        {
            "llm_tagger": {
                "presets": [
                    {
                        "id": "style_json",
                        "concurrency": 3,
                        "max_requests_per_minute": 12,
                    }
                ]
            }
        }
    )
    sess = _SlowSession(delay=0.01)
    tagger = llm_tagger.LLMTagger(session=sess)
    imgs = [_png(tmp_path / f"concurrent-{i}.png") for i in range(2)]

    list(tagger.tag(imgs))

    assert waits == [(0.0, 12)]

    waits.clear()
    secrets.update(
        {
            "llm_tagger": {
                "presets": [
                    {
                        "id": "style_json",
                        "concurrency": 1,
                        "max_requests_per_minute": 12,
                    }
                ]
            }
        }
    )
    tagger = llm_tagger.LLMTagger(session=_SlowSession(delay=0.01))
    list(tagger.tag([_png(tmp_path / "serial.png")]))

    assert waits == [(0.0, 0)]


def test_rate_limiter_uses_rolling_minute_window(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1000.0
    slept: list[float] = []

    def fake_monotonic() -> float:
        return now

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        slept.append(seconds)
        now += seconds

    monkeypatch.setattr(llm_tagger.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(llm_tagger.time, "sleep", fake_sleep)
    limiter = llm_tagger._RequestRateLimiter(
        0.0,
        max_requests_per_minute=2,
    )

    limiter.wait()
    limiter.wait()
    limiter.wait()

    assert slept == [1.0] * 60


def test_uses_editable_prompt_preset(isolated_secrets, tmp_path: Path) -> None:
    secrets.update(
        {
            "llm_tagger": {
                "current_preset": "my_style",
                "presets": [
                    {
                        "id": "my_style",
                        "label": "My Style",
                        "messages": [
                            {"type": "text", "role": "system", "content": "MY PROMPT"},
                            {"type": "image"},
                        ],
                        "base_url": "http://x/v1",
                        "model": "vision",
                        "max_retries": 1,
                    }
                ],
            }
        }
    )
    sess = MagicMock()
    sess.post.return_value = _chat_response('{"tags":["ink"]}')
    tagger = llm_tagger.LLMTagger(session=sess)
    img = _png(tmp_path / "1.png")

    list(tagger.tag([img]))

    body = sess.post.call_args.kwargs["json"]
    assert body["messages"][0]["content"] == "MY PROMPT"


def test_responses_endpoint_payload(isolated_secrets, tmp_path: Path) -> None:
    secrets.update(
        {"llm_tagger": {"presets": [{"id": "style_json", "endpoint": "responses"}]}}
    )
    sess = MagicMock()
    sess.post.return_value = _responses_response('{"tags":["ink","limited palette"]}')
    tagger = llm_tagger.LLMTagger(session=sess)
    img = _png(tmp_path / "1.png")

    [result] = list(tagger.tag([img]))

    assert result["tags"] == ["ink", "limited palette"]
    args, kwargs = sess.post.call_args
    assert args[0] == "http://x/v1/responses"
    body = kwargs["json"]
    assert body["instructions"]
    assert body["stream"] is False
    # builtin style_json 只有 system message，无 user → input content 仅 input_image
    image_part = next(c for c in body["input"][0]["content"] if c["type"] == "input_image")
    assert image_part["image_url"].startswith("data:image/jpeg;base64,")


def test_fetch_openai_compatible_models() -> None:
    sess = MagicMock()
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "data": [
            {"id": "vision-b"},
            {"id": "vision-a"},
            {"id": "vision-a"},
            {"name": "vision-c"},
        ]
    }
    sess.get.return_value = r

    items = llm_tagger.fetch_openai_compatible_models(
        "http://x/v1",
        "secret",
        timeout=9,
        session=sess,
    )

    assert items == ["vision-a", "vision-b", "vision-c"]
    args, kwargs = sess.get.call_args
    assert args[0] == "http://x/v1/models"
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["timeout"] == 9


def test_text_connectivity_uses_chat_shape() -> None:
    sess = MagicMock()
    r = MagicMock()
    r.status_code = 200
    r.text = '{"choices":[{"message":{"content":"ok"}}]}'
    r.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    sess.post.return_value = r

    result = llm_tagger.test_openai_compatible_connection(
        "http://x/v1",
        "secret",
        "text-model",
        endpoint="chat_completions",
        timeout=11,
        max_tokens=64,
        session=sess,
    )

    assert result["ok"] is True
    assert result["endpoint_url"] == "http://x/v1/chat/completions"
    assert result["response_preview"] == "ok"
    args, kwargs = sess.post.call_args
    assert args[0] == "http://x/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["json"]["max_tokens"] == 512
    assert kwargs["json"]["stream"] is False
    assert kwargs["json"]["messages"][1]["role"] == "user"


def test_text_connectivity_accepts_chat_sse_delta_stream() -> None:
    sess = MagicMock()
    sess.post.return_value = _sse_response(
        {"choices": [{"delta": {"content": "hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
    )

    result = llm_tagger.test_openai_compatible_connection(
        "http://x/v1",
        "secret",
        "text-model",
        endpoint="chat_completions",
        session=sess,
    )

    assert result["ok"] is True
    assert result["status_code"] == 200
    assert result["response_preview"] == "hello"


def test_text_connectivity_reports_sse_error_payload() -> None:
    sess = MagicMock()
    sess.post.return_value = _sse_response(
        {
            "error": {
                "type": "upstream_error",
                "code": 403,
                "message": "model route is forbidden",
            }
        },
    )

    result = llm_tagger.test_openai_compatible_connection(
        "http://x/v1",
        "secret",
        "bad-model",
        endpoint="chat_completions",
        session=sess,
    )

    assert result["ok"] is False
    assert result["status_code"] == 200
    assert "upstream_error" in result["error"]
    assert "model route is forbidden" in result["error"]


def test_text_connectivity_uses_responses_shape() -> None:
    sess = MagicMock()
    r = MagicMock()
    r.status_code = 200
    r.text = '{"output_text":"ok"}'
    r.json.return_value = {"output_text": "ok"}
    sess.post.return_value = r

    result = llm_tagger.test_openai_compatible_connection(
        "http://x/v1",
        "",
        "text-model",
        endpoint="responses",
        session=sess,
    )

    assert result["ok"] is True
    args, kwargs = sess.post.call_args
    assert args[0] == "http://x/v1/responses"
    assert kwargs["json"]["instructions"]
    assert kwargs["json"]["stream"] is False
    assert isinstance(kwargs["json"]["input"], str)


def test_responses_payload_uses_instructions(isolated_secrets, tmp_path: Path) -> None:
    secrets.update(
        {"llm_tagger": {"presets": [{"id": "style_json", "endpoint": "responses"}]}}
    )
    sess = MagicMock()
    sess.post.return_value = _responses_response('{"tags":["ink"]}')
    tagger = llm_tagger.LLMTagger(session=sess)
    img = _png(tmp_path / "1.png")

    list(tagger.tag([img]))

    body = sess.post.call_args.kwargs["json"]
    assert "instructions" in body
    assert body["input"][0]["role"] == "user"


def test_text_preset_returns_natural_caption(isolated_secrets, tmp_path: Path) -> None:
    # 切换到 joycaption preset 并把 endpoint 重新指向测试 URL
    secrets.update(
        {
            "llm_tagger": {
                "current_preset": "joycaption",
                "presets": [
                    {
                        "id": "joycaption",
                        "base_url": "http://x/v1",
                        "model": "vision",
                        "max_retries": 1,
                    }
                ],
            }
        }
    )
    sess = MagicMock()
    sess.post.return_value = _chat_response("a calm natural caption")
    tagger = llm_tagger.LLMTagger(session=sess)
    img = _png(tmp_path / "1.png")

    [result] = list(tagger.tag([img]))

    assert result["tags"] == ["a calm natural caption"]
    assert result["caption"] == "a calm natural caption"
    assert "caption_json" not in result


def test_image_data_url_respects_payload_cap(tmp_path: Path) -> None:
    img = tmp_path / "large.png"
    Image.effect_noise((1024, 1024), 80).convert("RGB").save(img)

    data_url = llm_tagger.LLMTagger._image_to_data_url(
        img,
        max_side=1024,
        quality=95,
        max_image_mb=0.25,
    )

    encoded = data_url.split(",", 1)[1].encode("ascii")
    assert len(encoded) <= int(0.25 * 1024 * 1024)
