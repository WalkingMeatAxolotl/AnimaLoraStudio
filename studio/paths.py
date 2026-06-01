"""Re-export shim — PR-7 真实模块 studio.infrastructure.paths。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到真实模块
（同 PR-3 services/ shim 模式）。新代码请直接
`from studio.infrastructure.paths import X`。
"""
import sys as _sys

from .infrastructure import paths as _real

_sys.modules[__name__] = _real
