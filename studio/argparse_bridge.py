"""Re-export shim — PR-7 真实模块 studio.infrastructure.argparse_bridge。"""
import sys as _sys

from .infrastructure import argparse_bridge as _real

_sys.modules[__name__] = _real
