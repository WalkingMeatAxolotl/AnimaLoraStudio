"""JoyCaption backward-compat shim.

JoyCaption 已合并为 LLM tagger 的 builtin preset。本 wrapper 仅为旧调用方
(`get_tagger("joycaption")`) 兜底 —— 新代码请直接 `get_tagger("llm",
overrides={"current_preset": "joycaption"})` 或不传 overrides 由全局
`current_preset` 决定。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import requests

from .llm import LLMTagger
from .base import ProgressFn, TagResult


class JoyCaptionTagger:
    name = "joycaption"
    requires_service = True

    def __init__(self, *, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()

    def _llm(self) -> LLMTagger:
        # 强制切到 joycaption preset（其字段由 builtin defaults + 用户在 Settings
        # 里改过的覆盖共同决定）。
        return LLMTagger(overrides={"current_preset": "joycaption"}, session=self._session)

    def is_available(self) -> tuple[bool, str]:
        return self._llm().is_available()

    def prepare(self) -> None:
        self._llm().prepare()

    def tag(
        self,
        image_paths: list[Path],
        on_progress: ProgressFn = lambda d, t: None,
    ) -> Iterator[TagResult]:
        yield from self._llm().tag(image_paths, on_progress=on_progress)
