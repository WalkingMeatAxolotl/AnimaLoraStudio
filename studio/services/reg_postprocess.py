"""Re-export shim — PR-3 真实模块 studio.services.reg.postprocess。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到真实模块。
新代码请直接 `from studio.services.reg.postprocess import X`。
"""
import sys as _sys

from .reg import postprocess as _real

_sys.modules[__name__] = _real
