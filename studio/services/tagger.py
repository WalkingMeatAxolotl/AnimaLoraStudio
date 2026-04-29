"""Tagger 抽象 + 工厂（PP4）。

每个 tagger 是一个独立类，按 `Tagger` 协议暴露相同接口：
    name / requires_service / is_available / prepare / tag

worker 拿到 name 后调 `get_tagger(name)` 取实例，跑 `prepare()` → 流式
`tag()` 拿结果。所有真实 IO（onnx 推理 / HTTP 调 vLLM）放在子类里，
本模块只定义协议和工厂，便于 mock 测试。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Protocol, TypedDict, runtime_checkable

ProgressFn = Callable[[int, int], None]  # (done, total)


class TagResult(TypedDict, total=False):
    image: Path
    tags: list[str]                # 排序好的（按概率降）
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


def get_tagger(name: str, overrides: dict | None = None) -> Tagger:
    """工厂：name = 'wd14' | 'joycaption'。

    `overrides` 仅 wd14 当前消费 —— 本次打标参数覆盖（threshold / model_id /
    blacklist_tags 等）；不影响全局 secrets.json。joycaption 暂时忽略。
    """
    if name == "wd14":
        from .wd14_tagger import WD14Tagger
        return WD14Tagger(overrides=overrides)
    if name == "joycaption":
        from .joycaption_tagger import JoyCaptionTagger
        return JoyCaptionTagger()
    raise ValueError(f"unknown tagger: {name!r}")


VALID_TAGGER_NAMES: tuple[str, ...] = ("wd14", "joycaption")
