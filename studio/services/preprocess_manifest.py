"""Re-export shim — PR-3 真实模块 studio.services.preprocess.manifest。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到真实模块。
新代码请直接 `from studio.services.preprocess.manifest import X`。
"""
import sys as _sys

from .preprocess import manifest as _real

_sys.modules[__name__] = _real
