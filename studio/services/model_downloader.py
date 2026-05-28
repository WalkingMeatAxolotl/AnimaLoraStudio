"""Re-export shim — PR-3 真实模块 studio.services.models.downloader。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到真实模块。
新代码请直接 `from studio.services.models.downloader import X`。
"""
import sys as _sys

from .models import downloader as _real

_sys.modules[__name__] = _real
