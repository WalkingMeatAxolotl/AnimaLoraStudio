"""Re-export shim — PR-3 真实模块 studio.services.dataset.browse。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到真实模块。
新代码请直接 `from studio.services.dataset.browse import X`。
"""
import sys as _sys

from .services.dataset import browse as _real

_sys.modules[__name__] = _real
