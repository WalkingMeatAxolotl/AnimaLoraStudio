"""Re-export shim — PR-3 真实模块 studio.services.presets.io。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到真实模块。
新代码请直接 `from studio.services.presets.io import X`。
"""
import sys as _sys

from .services.presets import io as _real

_sys.modules[__name__] = _real
