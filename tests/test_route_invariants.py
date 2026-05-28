"""PR-1 安全网 — route 数量与 decorator 计数的粗粒度不变量。

snapshot 是精细网（任何字符变化都触发），本文件是粗粒度二道防线：
- 数量落在合理区间
- server.py + api/routers/*.py 的 @app.<verb> / @router.<verb> 装饰器总数 == APIRoute 数

PR-5 起从 server.py 抽 router 到 api/routers/，本测试同步扫描两处装饰器
之和。新加 router 文件（PR-6 继续抽）后扫描自动覆盖。
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.routing import APIRoute

from studio.server import app

STUDIO_DIR = Path(__file__).parent.parent / "studio"
SERVER_PY = STUDIO_DIR / "server.py"
API_ROUTERS_DIR = STUDIO_DIR / "api" / "routers"


def test_route_count_in_sane_range() -> None:
    n = len(app.routes)
    assert 100 <= n <= 250, f"len(app.routes) = {n}，超出合理区间 [100, 250]"


def test_decorator_count_matches_api_routes() -> None:
    # server.py 里 `@app.<verb>(...)` 装饰器
    src = SERVER_PY.read_text(encoding="utf-8")
    app_decorator_count = len(
        re.findall(
            r"^@app\.(get|post|put|delete|patch|api_route)\b",
            src,
            flags=re.MULTILINE,
        )
    )
    # api/routers/*.py 里 `@router.<verb>(...)` 装饰器
    router_decorator_count = 0
    if API_ROUTERS_DIR.is_dir():
        for py in sorted(API_ROUTERS_DIR.glob("*.py")):
            if py.name == "__init__.py":
                continue
            router_src = py.read_text(encoding="utf-8")
            router_decorator_count += len(
                re.findall(
                    r"^@router\.(get|post|put|delete|patch|api_route)\b",
                    router_src,
                    flags=re.MULTILINE,
                )
            )

    decorator_total = app_decorator_count + router_decorator_count
    api_route_count = sum(1 for r in app.routes if isinstance(r, APIRoute))
    assert decorator_total == api_route_count, (
        f"装饰器总数 {decorator_total}（server.py {app_decorator_count} + "
        f"api/routers/ {router_decorator_count}）≠ app.routes 里 APIRoute 实例 "
        f"{api_route_count} —— 差额可能来自漏 include_router 或 router 注册时丢了一个"
    )
