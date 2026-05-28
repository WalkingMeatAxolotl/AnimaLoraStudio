"""Re-export shim — PR-3 真实模块 studio.services.inference.cache。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到真实模块。
新代码请直接 `from studio.services.inference.cache import X`。
"""
import sys as _sys

from .inference import cache as _real

_sys.modules[__name__] = _real
