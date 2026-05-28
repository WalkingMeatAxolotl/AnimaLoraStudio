"""Re-export shim — PR-3 真实模块 studio.services.runtime.pending_install。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到真实模块。
新代码请直接 `from studio.services.runtime.pending_install import X`。
"""
import sys as _sys

from .runtime import pending_install as _real

_sys.modules[__name__] = _real
