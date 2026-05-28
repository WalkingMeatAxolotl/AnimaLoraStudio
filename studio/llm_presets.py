"""Re-export shim — PR-7 真实模块 studio.infrastructure.llm_presets。"""
import sys as _sys

from .infrastructure import llm_presets as _real

_sys.modules[__name__] = _real
