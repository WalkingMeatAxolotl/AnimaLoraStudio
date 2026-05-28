"""Re-export shim — PR-7 真实包 studio.infrastructure.migrations。

注意：本 shim 是 .py 文件但代理的是 _package_，sys.modules 别名让
`from studio.migrations import X` / `from studio.migrations._v2_projects import Y`
都透明转发到真实包（同 PR-3 services/ 子包同名覆盖模式的镜像）。
"""
import sys as _sys

from .infrastructure import migrations as _real

_sys.modules[__name__] = _real
