"""Re-export shim — PR-7 真实模块 studio.infrastructure.log_tail。"""
import sys as _sys

from .infrastructure import log_tail as _real

_sys.modules[__name__] = _real
