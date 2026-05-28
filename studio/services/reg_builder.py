"""Re-export shim — PR-3 真实模块 studio.services.reg.builder。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到真实模块。
新代码请直接 `from studio.services.reg.builder import X`。
"""
import sys as _sys

from .reg import builder as _real

_sys.modules[__name__] = _real
