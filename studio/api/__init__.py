"""HTTP/API 层 — PR-5 起从 studio/server.py 抽出。

子模块：
    app.py        FastAPI 实例 + middleware + lifespan 绑定
    lifespan.py   启动 / 关闭钩子（ensure_dirs / db.init_db / supervisor 启停 / SSE）
    middleware.py _SelectiveGZipMiddleware（按路径前缀跳过 gzip）
    errors.py     4 个 HTTPException helper（safe_join_or_400 / 校验 / data export 路径）
    responses.py  共享响应常量（EMPTY_STATE）
    static.py     SPAStaticFiles（react-router 兜底回 index.html）
    main.py       `main()` uvicorn 启动入口
"""
