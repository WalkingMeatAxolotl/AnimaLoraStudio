"""Re-export shim — PR-3 真实模块 studio.services.data_io.train_io。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到真实模块。
新代码请直接 `from studio.services.data_io.train_io import X`。
"""
import sys as _sys

from .data_io import train_io as _real

_sys.modules[__name__] = _real
