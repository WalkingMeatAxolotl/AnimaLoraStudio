"""AnimaStudio FastAPI server — PR-8 final shim.

历史上 server.py 一文件 4657 行装全 router / lifespan / middleware / 工具
helper。PR-5..PR-6.5 把全部 routes + BaseModel + helper 搬到 `studio/api/`
子包；PR-8 final shim 只剩 30+ 行：

- `app` —— FastAPI 实例（真实在 `studio/api/app.py`），所有 router 通过那里
  `include_router` 注册。外部代码用 `from studio.server import app` 拿到的就是
  同一对象。
- `main` —— uvicorn 启动入口（真实在 `studio/api/main.py`）。
- `HTTPException` —— `pytest.raises(server.HTTPException)` 兼容。
- 6 个 path 常量 —— 测试 fixture 仍 `monkeypatch.setattr(server, X)` 期望存在；
  保 re-import 不破坏老 fixture（新位置的 patch 也已加，详 PR-5/6 PR 描述）。
- SPA mount —— 依赖运行时 `WEB_DIST.exists()` 检查，留在这里因为搬到
  `api/app.py` 会让检查时机变成模块 import-time（lifespan 之前），行为漂。

入口：
    uvicorn studio.server:app                # 老 cli.py / dev 模式
    python -m studio                         # cli.py 默认 run
    python -m studio.server [--host ...]     # 直接启 server（main()）
"""
from __future__ import annotations

# pytest.raises(server.HTTPException) 兼容
from fastapi import HTTPException  # noqa: F401

from . import db  # noqa: F401  test fixtures: monkeypatch.setattr(server.db, "STUDIO_DB", X)
from .api.app import app
from .api.main import main
from .api.static import SPAStaticFiles
from .paths import (
    LOGS_DIR,          # noqa: F401  test fixture monkeypatch path
    OUTPUT_DIR,        # noqa: F401
    REPO_ROOT,         # noqa: F401
    STUDIO_DB,         # noqa: F401
    USER_PRESETS_DIR,  # noqa: F401
    WEB_DIST,
)


# SPA mount — see module docstring for why this stays in server.py
if WEB_DIST.exists():
    app.mount(
        "/studio",
        SPAStaticFiles(directory=str(WEB_DIST), html=True),
        name="studio",
    )


if __name__ == "__main__":
    main()
