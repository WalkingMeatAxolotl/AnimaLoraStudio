"""文本编码工具：Qwen 隐藏态 + T5 加权 tokenization + tag 权重解析。

抽自原 runtime/anima_train.py L777-1071（ADR 0003 PR-A）。

公开：
- encode_qwen — Qwen3 文本编码（带空字符串兜底）
- tokenize_t5_weighted — 参考 ComfyUI anima-kai，按 tag 切分 + 权重 + pad
- build_comfy_anima_conditioning_inputs — Generate Comfy parity 路径用的
  raw-Qwen + SDTokenizer-style T5 加权输入
- tokenize_t5_comfy_literal — 训练 caption 的 Comfy-style 字面 T5 tokenization
  （批量、不解析权重语法；caption_comfy_encoding=true 时由训练 loop 使用）

内部：
- _parse_weighted_tag / _build_qwen_text_from_prompt
"""

from __future__ import annotations

import logging

import torch


logger = logging.getLogger(__name__)


def encode_qwen(model, tokenizer, texts, device, max_length=512, preserve_empty_text: bool = False):
    """Qwen 文本编码。"""
    # Qwen3 tokenizer 对空字符串可能返回 0 tokens（会导致模型内部 reshape 失败）
    # ComfyUI 的 AnimaTokenizer 设置了 min_length=1，这里做同等兜底。
    if isinstance(texts, str):
        texts = [texts]
    if preserve_empty_text:
        texts = [("" if t is None else str(t)) for t in texts]
    else:
        texts = [(" " if (t is None or str(t).strip() == "") else str(t)) for t in texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    uses_comfy_masking = bool(getattr(model, "uses_comfy_clip_masking", False))

    # 仍可能出现空序列（极端 tokenizer 行为），强制塞 1 个 token
    if inputs["input_ids"].ndim == 2 and inputs["input_ids"].shape[1] == 0:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        bs = len(texts)
        inputs["input_ids"] = torch.full((bs, 1), int(pad_id), dtype=torch.long)
        mask_value = 0 if uses_comfy_masking else 1
        inputs["attention_mask"] = torch.full((bs, 1), mask_value, dtype=torch.long)
    inputs = inputs.to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden = outputs.hidden_states[-1]
    # Comfy CLIPTextEncode passes attention masks into Qwen but does not zero
    # masked hidden states before handing them to Anima's LLM adapter.
    if not uses_comfy_masking:
        mask = inputs["attention_mask"].unsqueeze(-1)
        hidden = hidden * mask

    return hidden, inputs["attention_mask"]


def _parse_weighted_tag(tag: str) -> tuple[str, float]:
    """
    解析单个 tag 的权重（参考指南"权重控制"）。
    支持：
    - (tag:1.5)
    - (tag) / ((tag))  => 1.1^n
    - [tag]            => 1/1.1
    """
    import re

    s = tag.strip()
    if not s:
        return "", 1.0

    # 显式 (xxx:1.23)
    m = re.fullmatch(r"\(\s*(.+?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*\)", s)
    if m:
        return m.group(1).strip(), float(m.group(2))

    # 统计外层 () / [] 深度
    w = 1.0
    while True:
        s2 = s.strip()
        if len(s2) >= 2 and s2[0] == "(" and s2[-1] == ")":
            s = s2[1:-1].strip()
            w *= 1.1
            continue
        if len(s2) >= 2 and s2[0] == "[" and s2[-1] == "]":
            s = s2[1:-1].strip()
            w /= 1.1
            continue
        break
    return s.strip(), float(w)


def _build_qwen_text_from_prompt(prompt: str) -> str:
    # Qwen 通道不传权重，只传"干净标签文本"（参考 ComfyUI anima-kai 的做法）
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    clean = []
    for p in parts:
        t, _w = _parse_weighted_tag(p)
        if t:
            clean.append(t)
    return ", ".join(clean)


_ESCAPED_OPEN_PAREN = "\x00"
_ESCAPED_CLOSE_PAREN = "\x01"


def _split_trailing_prompt_weight(text: str) -> tuple[str, float | None]:
    if ":" not in text:
        return text, None
    prompt_text, maybe_weight = text.rsplit(":", 1)
    if not prompt_text:
        return text, None
    try:
        return prompt_text, float(maybe_weight.strip())
    except ValueError:
        pass
    return text, None


def _escape_prompt_parentheses(text: str) -> str:
    out: list[str] = []
    idx = 0
    while idx < len(text):
        if text[idx] == "\\" and idx + 1 < len(text) and text[idx + 1] == "(":
            out.append(_ESCAPED_OPEN_PAREN)
            idx += 2
            continue
        if text[idx] == "\\" and idx + 1 < len(text) and text[idx + 1] == ")":
            out.append(_ESCAPED_CLOSE_PAREN)
            idx += 2
            continue
        out.append(text[idx])
        idx += 1
    return "".join(out)


def _unescape_prompt_parentheses(text: str) -> str:
    return text.replace(_ESCAPED_OPEN_PAREN, "(").replace(_ESCAPED_CLOSE_PAREN, ")")


def _parse_prompt_parentheses(text: str) -> list[str]:
    result: list[str] = []
    current_item = ""
    nesting_level = 0
    for char in text:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item += char
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def _token_weights_from_prompt_groups(text: str, current_weight: float) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for segment in _parse_prompt_parentheses(text):
        weight = current_weight
        if len(segment) >= 2 and segment[0] == "(" and segment[-1] == ")":
            inner = segment[1:-1]
            colon_idx = inner.rfind(":")
            weight *= 1.1
            if colon_idx > 0:
                inner_text, explicit_weight = _split_trailing_prompt_weight(inner)
                if explicit_weight is not None:
                    weight = explicit_weight
                    inner = inner_text
            out.extend(_token_weights_from_prompt_groups(inner, weight))
        else:
            out.append((segment, current_weight))
    return out


def _parse_comfy_weighted_prompt_segments(text: str, base_weight: float = 1.0) -> list[tuple[str, float]]:
    escaped_text = _escape_prompt_parentheses(text)
    return [
        (_unescape_prompt_parentheses(segment), weight)
        for segment, weight in _token_weights_from_prompt_groups(escaped_text, float(base_weight))
    ]


def _tokenizer_input_ids_without_eos(tokenizer, text: str, eos_id: int) -> list[int]:
    tokenized = tokenizer(text, add_special_tokens=False)
    ids = tokenized["input_ids"]
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    out = [int(tid) for tid in ids]
    while out and out[-1] == int(eos_id):
        out.pop()
    return out


def build_comfy_anima_conditioning_inputs(t5_tokenizer, prompt: str, max_length=512):
    """Build generate-time Anima text inputs using Comfy-compatible tokenization.

    Comfy's Anima tokenizer keeps Qwen text raw while using SDTokenizer-style
    weighted parsing for T5. This helper intentionally does not replace the
    legacy training tokenization helpers.
    """
    qwen_text = "" if prompt is None else str(prompt)

    eos_id = t5_tokenizer.eos_token_id if t5_tokenizer.eos_token_id is not None else 1
    ids: list[int] = []
    weights: list[float] = []
    for segment, weight in _parse_comfy_weighted_prompt_segments(qwen_text):
        if not segment:
            continue
        segment_ids = _tokenizer_input_ids_without_eos(t5_tokenizer, segment, int(eos_id))
        ids.extend(segment_ids)
        weights.extend([float(weight)] * len(segment_ids))

    ids.append(int(eos_id))
    weights.append(1.0)

    if max_length and len(ids) > int(max_length):
        keep = max(1, int(max_length))
        ids = ids[:keep]
        weights = weights[:keep]
        ids[-1] = int(eos_id)
        weights[-1] = 1.0

    t5_ids = torch.tensor([ids], dtype=torch.long)
    t5_weights = torch.tensor([weights], dtype=torch.float32)
    t5_attn = torch.ones_like(t5_ids, dtype=torch.long)
    return qwen_text, t5_ids, t5_attn, t5_weights


def tokenize_t5_comfy_literal(tokenizer, texts, max_length=512):
    """Comfy-style 字面 T5 tokenization（训练 caption 用，批量版）。

    与 build_comfy_anima_conditioning_inputs 的差异：caption 是数据不是 prompt，
    整段按字面文本分词——不做权重语法解析、不清洗。booru tag 的括号
    （`ganyu (genshin impact)`）保持字面字符，等价于 ComfyUI 用户推理时
    转义 `\\(...\\)` 后 T5 实际看到的 token 序列。

    返回与 tokenize_t5_weighted 相同的约定：input_ids / attention_mask(1=有效) /
    token_weights（有效位 1.0），padding 位权重 0.0（下游乘到 LLMAdapter 输出
    上等于把 padding cross 清零）。
    """
    if isinstance(texts, str):
        texts = [texts]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    seqs: list[list[int]] = []
    for text in texts:
        ids = _tokenizer_input_ids_without_eos(tokenizer, str(text), int(eos_id))
        ids.append(int(eos_id))
        if max_length and len(ids) > int(max_length):
            ids = ids[: int(max_length)]
            ids[-1] = int(eos_id)
        seqs.append(ids)

    max_len = max((len(s) for s in seqs), default=1)
    input_ids = torch.full((len(seqs), max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    token_w = torch.zeros((len(seqs), max_len), dtype=torch.float32)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, : len(s)] = 1
        token_w[i, : len(s)] = 1.0
    return input_ids, attention_mask, token_w


def apply_t5_token_weights(cross: torch.Tensor, token_weights: torch.Tensor | None) -> torch.Tensor:
    """Apply ComfyUI Anima t5xxl_weights to LLM adapter output.

    ComfyUI multiplies the processed cross-attention embeddings by T5 token
    weights before padding to 512 tokens. This helper keeps the behavior local
    to callers that already precompute ``cross`` via ``preprocess_text_embeds``.
    """
    if token_weights is None:
        return cross
    weights = token_weights.to(device=cross.device, dtype=cross.dtype)
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)
    if weights.ndim != 2:
        return cross

    if weights.shape[0] == 1 and cross.shape[0] != 1:
        weights = weights.expand(cross.shape[0], -1)
    if weights.shape[0] != cross.shape[0]:
        return cross
    if weights.shape[1] < cross.shape[1]:
        weights = torch.nn.functional.pad(weights, (0, cross.shape[1] - weights.shape[1]), value=1.0)
    elif weights.shape[1] > cross.shape[1]:
        weights = weights[:, : cross.shape[1]]
    return cross * weights.unsqueeze(-1)


def tokenize_t5_weighted(tokenizer, texts, max_length=512):
    """
    参考 ComfyUI 的 anima-kai：按逗号切分 tag，逐 tag 分词，并为每个 token 附带权重。
    返回：input_ids, attention_mask(1=有效), token_weights
    """
    if isinstance(texts, str):
        texts = [texts]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    all_ids = []
    all_w = []
    for text in texts:
        tags = [t.strip() for t in str(text).split(",") if t.strip()]
        ids = []
        ws = []
        for tag in tags:
            clean_tag, weight = _parse_weighted_tag(tag)
            if not clean_tag:
                continue
            tok = tokenizer(clean_tag, add_special_tokens=False)
            for tid in tok["input_ids"]:
                ids.append(int(tid))
                ws.append(float(weight))

        # 末尾补一个 eos（ComfyUI 也是最后加一个终止 token）
        ids.append(int(eos_id))
        ws.append(1.0)

        # 截断到 max_length（保留最后一个 eos）
        if max_length and len(ids) > max_length:
            ids = ids[: max_length - 1] + [int(eos_id)]
            ws = ws[: max_length - 1] + [1.0]

        all_ids.append(torch.tensor(ids, dtype=torch.long))
        all_w.append(torch.tensor(ws, dtype=torch.float32))

    # pad 到 batch 内最长
    max_len = max(x.numel() for x in all_ids) if all_ids else 1
    input_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
    token_w = torch.zeros((len(all_w), max_len), dtype=torch.float32)
    attention_mask = torch.zeros((len(all_ids), max_len), dtype=torch.long)

    for i, (ids, ws) in enumerate(zip(all_ids, all_w)):
        L = ids.numel()
        input_ids[i, :L] = ids
        token_w[i, :L] = ws
        attention_mask[i, :L] = 1

    return input_ids, attention_mask, token_w
