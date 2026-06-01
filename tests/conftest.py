"""测试公共配置：保证 `import studio.*` / `import train_monitor` / `import anima_*` 能找到。

`train_monitor` 和 `anima_*`（train / generate / daemon / reg_ai）都在 `runtime/`，
没改成包导入（仍是裸脚本风格），所以要把 `runtime/` 注入 sys.path。

PR-1 C4 加 _isolate_studio_logging session fixture：让 api/lifespan、cli.main、
workers/_base.worker_main 在被测试触发时不真装 setup_logging（防污染 caplog
+ 防写真 studio_data/logs/）。详 fixture docstring。"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "runtime"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


@pytest.fixture(scope="session", autouse=True)
def _isolate_studio_logging(tmp_path_factory: pytest.TempPathFactory):
    """PR-1 C4 — 测试期间业务代码 setup_logging 全部 noop（保 caplog 干净）。

    业务入口（api/lifespan / cli.main / workers/_base.worker_main）会在自身
    启动时调 setup_logging。测试触发任何一处（比如 TestClient(app) 跑 lifespan）
    会装真 file handler 写 repo studio_data/logs/，污染。设 env 让 setup_logging
    顶部 early return。

    测 setup_logging 自身的 tests/test_logging_setup.py 用 monkeypatch.delenv
    单独解除该 env。
    """
    os.environ["ANIMA_LOGGING_NO_BOOTSTRAP"] = "1"
    # 同时设 ANIMA_LOG_DIR 兜底（万一某测试自己显式调 setup_logging 不传 log_dir）
    os.environ["ANIMA_LOG_DIR"] = str(tmp_path_factory.mktemp("studio_logs"))
    yield
