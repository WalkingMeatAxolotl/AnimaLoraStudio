"""用户可见 INFO 叙事行的 i18n 字典（tmp/log-text-audit Q1 拍板）。

分层口径：
- **INFO 叙事行（用户可见面：run.log / daemon ring / 下载卡 / CLI 终端）**
  走本模块 ``msg(msg_id, **kwargs)``，按 UI 语言输出中/英文；
- WARNING / ERROR / DEBUG 排障行与 studio.log 面统一英文，不进字典；
- traceback 与第三方库输出原样。

语言解析顺序：``ANIMA_UI_LANG`` env（supervisor / daemon spawn 时从
``secrets.system.ui_language`` 注入）→ 兜底读 secrets（手跑 / CLI 场景）→ zh。

字典按域拆三个数据文件（``_messages_{train,worker,server}.py``），避免多人/
多刀并行编辑冲突；msg_id 命名 ``<domain>.<event>``（snake_case），全仓唯一，
重复定义在 import 时报错。新增用户可见 INFO 行必须进字典——这是 Q1 口径的
长期成本，勿绕过。
"""
from __future__ import annotations

import os
from typing import Optional

UI_LANG_ENV = "ANIMA_UI_LANG"
_SUPPORTED = ("zh", "en")
_FALLBACK = "zh"

from ._messages_train import MESSAGES as _TRAIN  # noqa: E402
from ._messages_worker import MESSAGES as _WORKER  # noqa: E402
from ._messages_server import MESSAGES as _SERVER  # noqa: E402

MESSAGES: dict[str, dict[str, str]] = {}
for _part in (_TRAIN, _WORKER, _SERVER):
    _dup = MESSAGES.keys() & _part.keys()
    if _dup:
        raise RuntimeError(f"duplicate log msg_id across domains: {sorted(_dup)}")
    MESSAGES.update(_part)

_secrets_lang: Optional[str] = None


def current_lang() -> str:
    """当前日志语言。env 优先；缺失时读一次 secrets 并缓存；再兜底 zh。"""
    lang = os.environ.get(UI_LANG_ENV, "").strip().lower()
    if lang in _SUPPORTED:
        return lang
    global _secrets_lang
    if _secrets_lang is None:
        try:
            from .secrets import load  # noqa: PLC0415 — 惰性避免 import 环

            _secrets_lang = str(load().system.ui_language)
        except Exception:
            _secrets_lang = _FALLBACK
    return _secrets_lang if _secrets_lang in _SUPPORTED else _FALLBACK


def msg(msg_id: str, **kwargs: object) -> str:
    """按当前语言渲染一条用户可见 INFO 文案。

    缺 key / 占位符不匹配时不抛——日志路径绝不能因文案问题崩（T7 精神）：
    缺 key 返回 ``msg_id + 参数``（在日志里显眼可 grep，等于 fail-loud）。
    """
    entry = MESSAGES.get(msg_id)
    if entry is None:
        return f"{msg_id} {kwargs}" if kwargs else msg_id
    template = entry.get(current_lang()) or entry.get("zh") or entry.get("en") or msg_id
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def _reset_lang_cache_for_tests() -> None:
    global _secrets_lang
    _secrets_lang = None
