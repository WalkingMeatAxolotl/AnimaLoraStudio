"""Re-export shim — PR-7 真实模块 studio.infrastructure.db。

188 行单文件未做 connection/tasks/settings 3-way 拆（planning 原方案）—
理由同 secrets.py：现状不影响读 / 修改，3-way split 收益有限，独立 PR 更稳。

migrations/ 子包也搬到 studio/infrastructure/migrations/，`db.init_db()` 内
的 `from .migrations import apply_all` 相对引用透明 work。
"""
import sys as _sys

from .infrastructure import db as _real

_sys.modules[__name__] = _real
