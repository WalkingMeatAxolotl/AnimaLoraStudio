"""Re-export shim — PR-3 真实模块 studio.services.tagging.base。

本文件用 sys.modules 别名让 `studio.services.tagger` 直接指向
真实子模块对象。任何 monkeypatch / 属性访问（含私有 _xxx 和 import 进来
的依赖模块）都直接落到真实模块，与未拆分前行为一致。

新代码请直接 `from studio.services.tagging.base import X`。
"""
import sys as _sys

from .tagging import base as _real

_sys.modules[__name__] = _real
