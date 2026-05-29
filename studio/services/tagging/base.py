"""Tagger 抽象 + 工厂（PP4）。

每个 tagger 是一个独立类，按 `Tagger` 协议暴露相同接口：
    name / requires_service / is_available / prepare / tag

worker 拿到 name 后调 `get_tagger(name)` 取实例，跑 `prepare()` → 流式
`tag()` 拿结果。所有真实 IO（onnx 推理 / HTTP 调 vLLM）放在子类里，
本模块只定义协议和工厂，便于 mock 测试。
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Iterator, Protocol, TypedDict, runtime_checkable

ProgressFn = Callable[[int, int], None]  # (done, total)


class TagResult(TypedDict, total=False):
    image: Path
    tags: list[str]                # 排序好的（按概率降）
    caption: str                   # 可选：渲染后的完整 caption 文本
    caption_json: dict             # 可选：结构化 JSON caption
    raw_scores: dict[str, float]   # 可选：每 tag 的概率
    error: str                     # 失败时填


@runtime_checkable
class Tagger(Protocol):
    name: str
    requires_service: bool

    def is_available(self) -> tuple[bool, str]:
        """快速检查是否可跑。返回 (ok, 状态描述)。前端 status 条用。"""

    def prepare(self) -> None:
        """耗时初始化（如 WD14 加载 ONNX；JoyCaption 调 /v1/models）。
        worker 启动一次。"""

    def tag(
        self,
        image_paths: list[Path],
        on_progress: ProgressFn = lambda d, t: None,
    ) -> Iterator[TagResult]:
        """流式：每张图 yield 一次 TagResult；失败时 result 含 'error' 字段。"""


# tagger 名 → (模块路径, 类名, 是否吃 overrides 参数)
# 加新 tagger 在这里加一行；server.py 的 TagJobRequest 同步加 `<name>_overrides`
# 字段（如果支持本次任务覆盖）即可。worker 走通用 `params.get(f"{name}_overrides")`。
_TAGGER_SPEC: dict[str, tuple[str, str, bool]] = {
    "wd14": ("studio.services.tagging.wd14", "WD14Tagger", True),
    "cltagger": ("studio.services.tagging.cltagger", "CLTagger", True),
    "joycaption": ("studio.services.tagging.joycaption", "JoyCaptionTagger", False),
    "llm": ("studio.services.tagging.llm", "LLMTagger", True),
}


def get_tagger(name: str, overrides: dict | None = None) -> Tagger:
    """工厂：按 `_TAGGER_SPEC` 懒加载实现并实例化。

    `overrides` 仅本地 ONNX tagger 当前消费 —— 本次打标参数覆盖，不影响全局
    secrets.json；不接 overrides 的 tagger（如 joycaption）会忽略。
    """
    spec = _TAGGER_SPEC.get(name)
    if spec is None:
        raise ValueError(f"unknown tagger: {name!r}")
    module_path, cls_name, takes_overrides = spec
    klass = getattr(importlib.import_module(module_path), cls_name)
    return klass(overrides=overrides) if takes_overrides else klass()


VALID_TAGGER_NAMES: tuple[str, ...] = tuple(_TAGGER_SPEC.keys())
