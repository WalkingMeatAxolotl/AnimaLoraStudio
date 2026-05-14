"""文本编码工具：Qwen 隐藏态 + T5 加权 tokenization + tag 权重解析。

抽自原 runtime/anima_train.py L777-1071（ADR 0003 PR-A）。

公开：
- encode_qwen — Qwen3 文本编码（带空字符串兜底）
- tokenize_t5_weighted — 参考 ComfyUI anima-kai，按 tag 切分 + 权重 + pad

内部：
- _parse_weighted_tag / _build_qwen_text_from_prompt
"""

from __future__ import annotations

import logging

import torch


logger = logging.getLogger(__name__)


def encode_qwen(model, tokenizer, texts, device, max_length=512):
    """Qwen 文本编码。"""
    # Qwen3 tokenizer 对空字符串可能返回 0 tokens（会导致模型内部 reshape 失败）
    # ComfyUI 的 AnimaTokenizer 设置了 min_length=1，这里做同等兜底。
    if isinstance(texts, str):
        texts = [texts]
    texts = [(" " if (t is None or str(t).strip() == "") else str(t)) for t in texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    # 仍可能出现空序列（极端 tokenizer 行为），强制塞 1 个 token
    if inputs["input_ids"].ndim == 2 and inputs["input_ids"].shape[1] == 0:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        bs = len(texts)
        inputs["input_ids"] = torch.full((bs, 1), int(pad_id), dtype=torch.long)
        inputs["attention_mask"] = torch.ones((bs, 1), dtype=torch.long)
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
    # 清零 padding 位置
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
