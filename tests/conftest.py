"""测试公共配置：保证 `import studio.*` / `import train_monitor` 能找到。

`train_monitor` 在 PR-1 后搬到 `tools/`，没改成包导入（仍是裸脚本风格），所以
要把 `tools/` 也注入 sys.path；`anima_train` 同理搬到 `scripts/`，但测试里
用的是 importlib spec 直接指文件路径，不靠 sys.path。"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tools"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)
