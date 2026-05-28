"""Re-export shim — PR-7 真实模块 studio.infrastructure.event_bus。"""
import sys as _sys

from .infrastructure import event_bus as _real

_sys.modules[__name__] = _real
