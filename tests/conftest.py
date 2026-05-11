"""测试公共配置：保证 `import studio.*` / `import train_monitor` / `import anima_*` 能找到。

`train_monitor` 和 `anima_*`（train / generate / daemon / reg_ai）都在 `runtime/`，
没改成包导入（仍是裸脚本风格），所以要把 `runtime/` 注入 sys.path。"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "runtime"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)
