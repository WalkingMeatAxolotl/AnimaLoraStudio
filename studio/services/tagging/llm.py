"""OpenAI-compatible vision LLM tagger.

Supports both Chat Completions and Responses style endpoints. The model is
asked to return structured JSON, then the worker can persist either local JSON
caption files or rendered TXT captions.
"""
from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from PIL import Image, ImageOps

from ... import secrets
from .caption_format import (
    caption_json_to_tags,
    caption_json_to_text,
    normalize_caption_json,
)
from .base import ProgressFn, TagResult


logger = logging.getLogger(__name__)


_CONNECTIVITY_SYSTEM_PROMPT = (
    "You are a diagnostic assistant. Answer exactly what the user asks, "
    "without mentioning policies, hidden reasoning, or implementation details."
)

_CONNECTIVITY_USER_PROMPT = """Connectivity test for an OpenAI-compatible endpoint.

Please return JSON only, no Markdown:
{
  "ok": true,
  "summary": "one sentence saying the endpoint can answer a non-trivial request",
  "items": [
    "base URL routing works",
    "model selection works",
    "authentication works",
    "the service can generate a moderately long answer"
  ],
  "note": "short note"
}

To make this a real generation test rather than a tiny ping, include 8 to 12
short English words in the summary and keep every item as a complete phrase.
"""


class _RequestRateLimiter:
    def __init__(
        self,
        requests_per_second: float,
        *,
        max_requests_per_minute: int = 0,
    ) -> None:
        self._interval = 0.0
        if requests_per_second > 0:
            self._interval = 1.0 / requests_per_second
        self._max_per_minute = max(0, int(max_requests_per_minute or 0))
        self._window_seconds = 60.0
        self._lock = threading.Lock()
        self._next_at = 0.0
        self._request_times: list[float] = []

    def wait(self) -> None:
        if self._interval <= 0 and self._max_per_minute <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                sleep_for = max(0.0, self._next_at - now)
                if self._max_per_minute > 0:
                    cutoff = now - self._window_seconds
                    self._request_times = [
                        ts for ts in self._request_times if ts > cutoff
                    ]
                    if len(self._request_times) >= self._max_per_minute:
                        sleep_for = max(
                            sleep_for,
                            self._request_times[0] + self._window_seconds - now,
                        )
                if sleep_for <= 0:
                    base = max(now, self._next_at)
                    if self._interval > 0:
                        self._next_at = base + self._interval
                    if self._max_per_minute > 0:
                        self._request_times.append(now)
                    return
            time.sleep(min(sleep_for, 1.0))


def _openai_compatible_endpoint(base_url: str, *, kind: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("LLM base_url 为空")

    parsed = urlparse(base)
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/models"):
        if path.endswith(suffix):
            root = base[: -len(suffix)]
            return f"{root}/{kind}"
    if path.endswith("/v1"):
        return f"{base}/{kind}"
    if path:
        return f"{base}/v1/{kind}"
    return f"{base}/v1/{kind}"


def _response_text(resp: requests.Response) -> str:
    text = getattr(resp, "text", "")
    return text if isinstance(text, str) else ""


def _response_content_type(resp: requests.Response) -> str:
    headers = getattr(resp, "headers", {}) or {}
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return ""
    value = getter("content-type")
    if not isinstance(value, str):
        value = getter("Content-Type")
    return value.lower() if isinstance(value, str) else ""


def _is_sse_response(resp: requests.Response, raw: str) -> bool:
    return "text/event-stream" in _response_content_type(resp) or raw.lstrip().startswith("data:")


def _iter_sse_payloads(raw: str) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []

    def flush() -> Iterator[dict[str, Any]]:
        if not data_lines:
            return
        data = "\n".join(data_lines).strip()
        data_lines.clear()
        if not data or data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM SSE 返回不是合法 JSON: {data[:200]}") from exc
        if isinstance(payload, dict):
            yield payload

    for line in raw.splitlines():
        line = line.rstrip("\r")
        if not line:
            yield from flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data = line[5:]
            if data.startswith(" "):
                data = data[1:]
            data_lines.append(data)
    yield from flush()


def _format_llm_error(error: Any) -> str:
    if isinstance(error, dict):
        parts = [
            error.get("type"),
            error.get("code") or error.get("status") or error.get("status_code"),
            error.get("message") or error.get("detail") or error.get("error"),
        ]
        text = " ".join(str(part) for part in parts if part not in (None, ""))
        return text or json.dumps(error, ensure_ascii=False)
    return str(error or "").strip()


def _extract_openai_error(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if error:
        return _format_llm_error(error)
    response = payload.get("response")
    if isinstance(response, dict):
        nested = _extract_openai_error(response)
        if nested:
            return nested
    last_error = payload.get("last_error")
    if last_error:
        return _format_llm_error(last_error)
    event_type = str(payload.get("type") or "")
    if event_type == "error" or event_type.endswith(".error"):
        return _format_llm_error(
            payload.get("message") or payload.get("detail") or payload
        )
    if str(payload.get("status") or "").lower() in {"failed", "error"}:
        return _format_llm_error(payload.get("message") or payload)
    return ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def _extract_openai_stream_delta(payload: dict[str, Any]) -> str:
    event_type = str(payload.get("type") or "")
    if event_type in {"response.output_text.delta", "response.refusal.delta"}:
        return str(payload.get("delta") or "")
    choices = payload.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    delta = choice.get("delta") or {}
    if isinstance(delta, dict):
        return _content_to_text(delta.get("content") or delta.get("text"))
    return ""


def _extract_openai_final_text(payload: dict[str, Any], *, endpoint: str) -> str:
    response = payload.get("response")
    if isinstance(response, dict):
        try:
            return LLMTagger._extract_responses_text(response)
        except Exception:  # noqa: BLE001
            return ""
    try:
        if endpoint == "responses" or "output_text" in payload or "output" in payload:
            return LLMTagger._extract_responses_text(payload)
        if "choices" in payload:
            return LLMTagger._extract_chat_text(payload)
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _extract_sse_response_text(raw: str, *, endpoint: str) -> str:
    parts: list[str] = []
    final_text = ""
    seen_payload = False
    for payload in _iter_sse_payloads(raw):
        seen_payload = True
        error = _extract_openai_error(payload)
        if error:
            raise RuntimeError(f"LLM SSE error: {error}")
        delta = _extract_openai_stream_delta(payload)
        if delta:
            parts.append(delta)
        final = _extract_openai_final_text(payload, endpoint=endpoint)
        if final:
            final_text = final
    if not seen_payload:
        raise RuntimeError("LLM SSE 返回为空")
    return ("".join(parts) if parts else final_text).strip()


def _extract_llm_response_text(resp: requests.Response, *, endpoint: str) -> str:
    raw = _response_text(resp)
    if _is_sse_response(resp, raw):
        return _extract_sse_response_text(raw, endpoint=endpoint)
    payload = resp.json()
    error = _extract_openai_error(payload)
    if error:
        raise RuntimeError(f"LLM error: {error}")
    return (
        LLMTagger._extract_responses_text(payload)
        if endpoint == "responses"
        else LLMTagger._extract_chat_text(payload)
    )


def fetch_openai_compatible_models(
    base_url: str,
    api_key: str = "",
    *,
    timeout: int = 30,
    session: Optional[requests.Session] = None,
) -> list[str]:
    endpoint = _openai_compatible_endpoint(base_url, kind="models")
    headers = {"Accept": "application/json"}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    sess = session or requests.Session()
    resp = sess.get(endpoint, headers=headers, timeout=max(5, int(timeout)))
    if resp.status_code >= 400:
        raise RuntimeError(f"模型列表读取失败 (HTTP {resp.status_code}): {resp.text[:300]}")
    payload = resp.json()
    raw_items = payload.get("data") or payload.get("models") or []
    items: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        model_id = str(raw.get("id") or raw.get("name") or "").strip()
        key = model_id.lower()
        if not model_id or key in seen:
            continue
        seen.add(key)
        items.append(model_id)
    items.sort(key=str.lower)
    return items


def test_openai_compatible_connection(
    base_url: str,
    api_key: str,
    model: str,
    *,
    endpoint: str = "chat_completions",
    timeout: int = 60,
    max_tokens: int = 700,
    temperature: float = 0.2,
    session: Optional[requests.Session] = None,
) -> dict[str, Any]:
    """Run a text-only connectivity test without persisting any settings."""
    endpoint_kind = "responses" if endpoint == "responses" else "chat/completions"
    endpoint_url = _openai_compatible_endpoint(base_url, kind=endpoint_kind)
    token_budget = max(512, int(max_tokens or 700))
    if endpoint == "responses":
        body = {
            "model": model,
            "instructions": _CONNECTIVITY_SYSTEM_PROMPT,
            "input": _CONNECTIVITY_USER_PROMPT,
            "temperature": temperature,
            "max_output_tokens": token_budget,
            "stream": False,
        }
    else:
        body = {
            "model": model,
            "temperature": temperature,
            "max_tokens": token_budget,
            "stream": False,
            "messages": [
                {"role": "system", "content": _CONNECTIVITY_SYSTEM_PROMPT},
                {"role": "user", "content": _CONNECTIVITY_USER_PROMPT},
            ],
        }
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    sess = session or requests.Session()
    started = time.monotonic()
    result: dict[str, Any] = {
        "ok": False,
        "endpoint": endpoint,
        "endpoint_url": endpoint_url,
        "model": model,
        "elapsed_ms": 0,
        "status_code": None,
        "response_preview": "",
        "error": "",
        "request_shape": "responses_text" if endpoint == "responses" else "chat_completions_text",
    }
    try:
        resp = sess.post(
            endpoint_url,
            headers=headers,
            json=body,
            timeout=(10, max(5, int(timeout or 60))),
        )
        result["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        result["status_code"] = resp.status_code
        raw_preview = _response_text(resp)[:1000]
        result["response_preview"] = raw_preview
        if resp.status_code >= 400:
            result["error"] = f"HTTP {resp.status_code}: {raw_preview[:500]}"
            return result
        text = _extract_llm_response_text(resp, endpoint=endpoint)
        result["response_preview"] = text[:1000]
        result["ok"] = bool(text.strip())
        if not result["ok"]:
            result["error"] = "LLM 返回空内容"
        return result
    except Exception as exc:  # noqa: BLE001
        result["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        result["error"] = str(exc)
        return result


class LLMTagger:
    name = "llm"
    requires_service = True

    def __init__(
        self,
        overrides: dict | None = None,
        *,
        session: Optional[requests.Session] = None,
    ) -> None:
        # overrides 可以包含两类键：
        #   - "current_preset"：切换 active preset id
        #   - 任意 LLMPresetConfig 字段（base_url / model / endpoint / temperature / ...）
        # `api_key` 出于安全考虑不允许从 overrides 传（避免泄漏到 task 日志）；要改用
        # Settings 持久化或显式 secrets.update()。
        self._overrides = {
            k: v
            for k, v in (overrides or {}).items()
            if v is not None and k != "api_key"
        }
        self._external_session = session is not None
        self._session = session or requests.Session()

    def _cfg(self) -> "secrets.LLMPresetConfig":
        """返回最终生效的 active preset（已 apply overrides）。"""
        tagger_cfg = secrets.load().llm_tagger
        # 1) 决定 active preset
        preset_id = str(self._overrides.get("current_preset") or tagger_cfg.current_preset)
        active = next((p for p in tagger_cfg.presets if p.id == preset_id), None)
        if active is None:
            active = tagger_cfg.active
        # 2) apply 字段 overrides
        preset_dict = active.model_dump()
        for k, v in self._overrides.items():
            if k == "current_preset":
                continue
            if k in preset_dict:
                preset_dict[k] = v
        return secrets.LLMPresetConfig(**preset_dict)

    def is_available(self) -> tuple[bool, str]:
        cfg = self._cfg()
        if not cfg.base_url:
            return False, "未配置 base_url"
        if not cfg.model:
            return False, "未配置 model"
        return True, f"{cfg.endpoint} · {cfg.model}"

    def prepare(self) -> None:
        ok, msg = self.is_available()
        if not ok:
            raise RuntimeError(f"LLM tagger 不可用: {msg}")

    def tag(
        self,
        image_paths: list[Path],
        on_progress: ProgressFn = lambda d, t: None,
    ) -> Iterator[TagResult]:
        cfg = self._cfg()
        total = len(image_paths)
        on_progress(0, total)
        workers = min(max(1, int(cfg.concurrency or 1)), max(1, total))
        if workers <= 1 or total <= 1:
            done = 0
            rate_limiter = _RequestRateLimiter(
                cfg.requests_per_second,
                max_requests_per_minute=cfg.max_requests_per_minute,
            )
            for p in image_paths:
                yield self._tag_one(cfg, p, rate_limiter=rate_limiter)
                done += 1
                on_progress(done, total)
            return

        self._ensure_session_pool(workers)
        rate_limiter = _RequestRateLimiter(
            cfg.requests_per_second,
            max_requests_per_minute=cfg.max_requests_per_minute,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._tag_one, cfg, p, rate_limiter=rate_limiter): p
                for p in image_paths
            }
            done = 0
            for future in concurrent.futures.as_completed(futures):
                p = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"image": p, "tags": [], "error": str(exc)}
                yield result
                done += 1
                on_progress(done, total)

    def _tag_one(
        self,
        cfg: "secrets.LLMPresetConfig",
        image_path: Path,
        *,
        rate_limiter: _RequestRateLimiter,
    ) -> TagResult:
        try:
            data_url = self._image_to_data_url(
                image_path,
                max_side=cfg.max_side,
                quality=cfg.jpeg_quality,
                max_image_mb=cfg.max_image_mb,
            )
            content = self._call_with_retry(
                cfg,
                data_url,
                image_path,
                rate_limiter=rate_limiter,
            )
            if cfg.output_format == "text":
                text = content.strip()
                if not text:
                    raise RuntimeError("LLM 返回空内容")
                return {"image": image_path, "tags": [text], "caption": text}
            parsed = self._parse_json_text(content)
            caption_json = self._normalize_llm_payload(parsed)
            return {
                "image": image_path,
                "tags": caption_json_to_tags(caption_json),
                "caption": caption_json_to_text(caption_json),
                "caption_json": caption_json,
            }
        except Exception as exc:  # noqa: BLE001
            return {"image": image_path, "tags": [], "error": str(exc)}

    def _ensure_session_pool(self, workers: int) -> None:
        if self._external_session:
            return
        pool_size = max(1, int(workers))
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _headers(self, cfg: "secrets.LLMPresetConfig") -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        return headers

    def _call_with_retry(
        self,
        cfg: "secrets.LLMPresetConfig",
        data_url: str,
        image_path: Path,
        *,
        rate_limiter: _RequestRateLimiter | None = None,
    ) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, cfg.max_retries + 1):
            try:
                if cfg.endpoint == "responses":
                    endpoint = _openai_compatible_endpoint(cfg.base_url, kind="responses")
                    body = self._responses_payload(cfg, data_url, image_path)
                else:
                    endpoint = _openai_compatible_endpoint(
                        cfg.base_url, kind="chat/completions"
                    )
                    body = self._chat_payload(cfg, data_url, image_path)
                if rate_limiter is not None:
                    rate_limiter.wait()
                started = time.monotonic()
                logger.info(
                    "LLM tagger POST %s model=%s endpoint=%s image=%s timeout=%ss",
                    endpoint,
                    cfg.model,
                    cfg.endpoint,
                    image_path.name,
                    cfg.timeout,
                )
                resp = self._session.post(
                    endpoint,
                    headers=self._headers(cfg),
                    json=body,
                    timeout=(10, max(5, int(cfg.timeout))),
                )
                elapsed_ms = int((time.monotonic() - started) * 1000)
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {resp.status_code} after {elapsed_ms}ms at {endpoint}: {resp.text[:300]}"
                    )
                content = _extract_llm_response_text(resp, endpoint=cfg.endpoint)
                logger.info(
                    "LLM tagger OK %s model=%s image=%s elapsed=%sms",
                    endpoint,
                    cfg.model,
                    image_path.name,
                    elapsed_ms,
                )
                return content
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < cfg.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"LLM 调用失败（{cfg.max_retries} 次重试）: {last_exc}")

    def _chat_payload(
        self,
        cfg: "secrets.LLMPresetConfig",
        data_url: str,
        image_path: Path,
    ) -> dict[str, Any]:
        """按 preset.messages 顺序构造 chat-completions messages 数组。

        text item → 单条 message；image item → 单条 `{role: user, content: [image_url]}`。
        相邻同 role 不自动合并（OpenAI 完全 OK；Anthropic 兼容层会自动处理）。
        """
        messages: list[dict[str, Any]] = []
        for item in cfg.messages:
            if item.type == "image":
                messages.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": data_url}}],
                })
            else:
                messages.append({"role": item.role, "content": item.content})
        return {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "stream": False,
            "messages": messages,
        }

    def _responses_payload(
        self,
        cfg: "secrets.LLMPresetConfig",
        data_url: str,
        image_path: Path,
    ) -> dict[str, Any]:
        """Responses API 限制式适配：

        - 所有 type=text + role=system 的 messages 合并进 instructions（用 \\n\\n 拼接）
        - 取第一条 type=text + role=user 的 content 作为 user 文本（其他 user/assistant 忽略）
        - 图片附加到 user content 数组里

        OpenAI Responses 自身的 multi-turn input 支持不稳，第三方兼容也参差 ——
        所以这里采用「单 system + 单 user + image」的保守映射。UI 会在选 responses
        endpoint 时提示该限制。
        """
        system_texts: list[str] = []
        user_text = ""
        for item in cfg.messages:
            if item.type != "text":
                continue
            if item.role == "system":
                if item.content:
                    system_texts.append(item.content)
            elif item.role == "user" and not user_text:
                user_text = item.content

        user_content: list[dict[str, Any]] = []
        if user_text:
            user_content.append({"type": "input_text", "text": user_text})
        user_content.append({"type": "input_image", "image_url": data_url})

        return {
            "model": cfg.model,
            "instructions": "\n\n".join(system_texts),
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_tokens,
            "stream": False,
            "input": [{"role": "user", "content": user_content}],
        }

    @staticmethod
    def _extract_chat_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
        return str(content or "").strip()

    @staticmethod
    def _extract_responses_text(payload: dict[str, Any]) -> str:
        direct = payload.get("output_text")
        if direct:
            return str(direct).strip()
        parts: list[str] = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"}:
                    parts.append(str(content.get("text") or ""))
        if parts:
            return "".join(parts).strip()
        # Some compatible providers return Chat Completions shape from /responses.
        return LLMTagger._extract_chat_text(payload)

    @staticmethod
    def _parse_json_text(content: str) -> dict[str, Any]:
        text = str(content or "").strip()
        if not text:
            raise RuntimeError("LLM 返回空内容")
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if fenced:
            text = fenced.group(1).strip()
        else:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                text = match.group(0).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM 返回不是合法 JSON: {text[:200]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM JSON 顶层必须是 object")
        return parsed

    @staticmethod
    def _normalize_llm_payload(parsed: dict[str, Any]) -> dict[str, Any]:
        if isinstance(parsed.get("tags"), list) and not any(
            key in parsed for key in ("appearance", "environment", "ai_output", "fixed")
        ):
            return normalize_caption_json({"tags": parsed.get("tags"), "nl": parsed.get("nl", "")})
        return normalize_caption_json(parsed)

    @staticmethod
    def _image_to_data_url(
        image_path: Path,
        *,
        max_side: int,
        quality: int,
        max_image_mb: float = 5.0,
    ) -> str:
        path = Path(image_path)
        if not path.exists():
            raise RuntimeError(f"Image does not exist: {path}")
        max_bytes = max(1, int(float(max_image_mb or 5.0) * 1024 * 1024))
        side = max(64, int(max_side))
        initial_quality = max(1, min(100, int(quality)))
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img) or img
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            while True:
                work = img.copy()
                work.thumbnail((side, side))
                canvas = Image.new("RGB", work.size, (255, 255, 255))
                canvas.paste(work, mask=work.getchannel("A"))
                for q in (initial_quality, 75, 65, 55, 45, 35):
                    q = min(initial_quality, q)
                    buf = io.BytesIO()
                    canvas.save(buf, format="JPEG", quality=max(1, q), optimize=True)
                    data = buf.getvalue()
                    encoded_bytes = base64.b64encode(data)
                    if len(encoded_bytes) <= max_bytes:
                        encoded = encoded_bytes.decode("ascii")
                        return f"data:image/jpeg;base64,{encoded}"
                if side <= 256:
                    raise RuntimeError(
                        f"压缩后图片仍超过上限 {max_image_mb:g} MB，请调大 max_image_mb 或使用更小图片"
                    )
                side = max(256, int(side * 0.8))
                initial_quality = min(initial_quality, 75)
