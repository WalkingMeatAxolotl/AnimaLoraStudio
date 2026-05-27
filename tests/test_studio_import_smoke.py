"""PR-1 安全网 — studio/ 下每个 .py 都能 import。

重构期间最易出的事故是循环 import / 搬目录后 import 路径忘改。
本测试用 parametrize 给每个模块独立 case，定位精确。

排除：
- __pycache__/
- studio/web/（前端构建产物 + node_modules）
- 任何 .py 在 web/node_modules/ 下
- studio/__main__.py（无 `if __name__ == "__main__"` 守护，import 时直接执行 main()）

已知 import-time 副作用（属现状，PR-5/PR-7 才改）：
- studio.server: ensure_dirs() + db.init_db()
- studio.services.onnxruntime_setup: DLL preload
本测试接受这些副作用，只验证 import 不抛异常。
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

STUDIO_ROOT = Path(__file__).parent.parent / "studio"


def _enumerate_modules() -> list[str]:
    modules: list[str] = []
    for py in sorted(STUDIO_ROOT.rglob("*.py")):
        rel = py.relative_to(STUDIO_ROOT.parent)
        parts = rel.with_suffix("").parts
        # 跳过缓存
        if "__pycache__" in parts:
            continue
        # 跳过前端目录下混进来的 .py（如 web/node_modules/flatted/python/flatted.py）
        if "web" in parts:
            continue
        # 跳过 __main__：无 if __name__ 守护，import 即执行 main() 并 SystemExit
        if parts[-1] == "__main__":
            continue
        # 包内 __init__ 用包名表示
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(parts))
    return modules


MODULES = _enumerate_modules()


def test_module_list_not_empty() -> None:
    assert len(MODULES) > 50, f"枚举到的 studio/ 模块数只有 {len(MODULES)}，可能扫描出错"


@pytest.mark.parametrize("module_name", MODULES, ids=MODULES)
def test_import_module(module_name: str) -> None:
    importlib.import_module(module_name)
