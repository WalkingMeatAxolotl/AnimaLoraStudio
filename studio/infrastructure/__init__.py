"""底层基础设施 — PR-7 起从 studio/ 顶层抽出。

子模块：
    paths.py           路径常量 + safe_join / validate_path_component
    event_bus.py       进程内 SSE 总线
    log_tail.py        per-task 日志增量读 + monitor state 轮询
    argparse_bridge.py pydantic 模型 → argparse 参数派生
    llm_presets.py     studio/llm_presets/*.json 出厂预设加载

后续 PR：
    secrets/  PR-7 commit 2  secrets.py 763 → models/store/migrations 3 文件
    db/       PR-7 commit 3  db.py 188 + studio/migrations/ → infrastructure/db/
"""
