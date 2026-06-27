"""评估指标 registry —— 所有 eval 指标的单一真相。

给三处共用一份定义：① Settings 复选框列表（用户勾选启用哪些指标）；② eval 编排
（`eval_auto` 只跑「启用集合」对应的 runner）；③ 前端指标说明 / 展示。

一个 **runner** = 一个 metric job（同一次推理产出一个或多个指标）：
- `clip` runner 产出 clip_t + clip_i（共享 CLIP 编码）
- `dino` runner 产出 dino_i
- `ccip` runner 产出 ccip_i（角色身份）
- `tag`  runner 产出 tag_recall（prompt 跟随，复用 WD14）

runner 级门控：只要它的任一指标被启用就跑该 runner（同 runner 的指标共享推理，
单独关其中一个不省算力，只影响展示）。`models` 是该指标依赖的下载中心条目 key。
"""
from __future__ import annotations

from typing import Any, Iterable

# 顺序即前端展示顺序。default=True 的进默认启用集合（保持现有行为）。
METRICS: list[dict[str, Any]] = [
    {
        "key": "clip_t", "label": "CLIP-T", "runner": "clip",
        "models": ["clip"], "default": True,
        "desc": "生成图与 prompt 文本的 CLIP 相似度（prompt following）",
        "note": "CLIP 文本塔不认 booru tag，tag caption 下分低、噪声大",
    },
    {
        "key": "clip_i", "label": "CLIP-I", "runner": "clip",
        "models": ["clip"], "default": True,
        "desc": "生成图与参考图的 CLIP 图像相似度（整体语义）",
        "note": "自然图域、偏粗；非动漫适配",
    },
    {
        "key": "dino_i", "label": "DINO-I", "runner": "dino",
        "models": ["dino"], "default": True,
        "desc": "生成图与参考图的 DINOv2 特征相似度（主体/结构保真）",
        "note": "非动漫适配；small 版区分力弱",
    },
    {
        "key": "ccip_i", "label": "CCIP-I", "runner": "ccip",
        "models": ["ccip"], "default": False,
        "desc": "生成图被参考集判为同一动漫角色的比例（角色身份保真，动漫域）",
        "note": "仅单角色的角色 LoRA；弱发色/肤色",
    },
    {
        "key": "tag_recall", "label": "Tag-Recall", "runner": "tag",
        "models": ["wd14"], "default": False,
        "desc": "对生成图回标，prompt 里 booru tag 的召回率（动漫原生 prompt following）",
        "note": "仅 booru-tag caption 适用",
    },
]

_BY_KEY = {m["key"]: m for m in METRICS}
ALL_KEYS: list[str] = [m["key"] for m in METRICS]
DEFAULT_ENABLED: list[str] = [m["key"] for m in METRICS if m["default"]]


def metric(key: str) -> dict[str, Any] | None:
    return _BY_KEY.get(key)


def normalize_enabled(enabled: Iterable[str] | None) -> set[str]:
    """过滤出合法的指标 key；None / 空 → 默认集合（保持现有行为）。"""
    if not enabled:
        return set(DEFAULT_ENABLED)
    out = {k for k in enabled if k in _BY_KEY}
    return out or set(DEFAULT_ENABLED)


def enabled_runners(enabled: Iterable[str] | None) -> list[str]:
    """启用集合 → 需要跑的 runner（按 METRICS 顺序去重）。一个 runner 只要它的
    任一指标被启用就在列。"""
    active = normalize_enabled(enabled)
    out: list[str] = []
    for m in METRICS:
        if m["key"] in active and m["runner"] not in out:
            out.append(m["runner"])
    return out


def runner_metrics(runner: str) -> list[str]:
    return [m["key"] for m in METRICS if m["runner"] == runner]


def public_catalog() -> list[dict[str, Any]]:
    """给前端 Settings 复选框 + 指标说明用（不含内部 runner 字段以外的实现细节）。"""
    return [
        {
            "key": m["key"], "label": m["label"], "runner": m["runner"],
            "models": m["models"], "default": m["default"],
            "desc": m["desc"], "note": m["note"],
        }
        for m in METRICS
    ]
