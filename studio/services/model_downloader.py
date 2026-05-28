"""Re-export shim — PR-3.8 真实模块 studio.services.models（4-way 拆后改为别名整个包）。

sys.modules 别名让旧路径的 monkeypatch / 私有访问透明转发到 models 包入口；
原 `from studio.services.model_downloader import X` 经由包 __init__.py re-export 仍工作。
新代码请直接 `from studio.services.models import X`。
"""
import sys as _sys

from . import models as _real

_sys.modules[__name__] = _real
