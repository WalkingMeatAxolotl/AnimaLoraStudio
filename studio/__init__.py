"""AnimaStudio - 训练监控、配置编辑与任务队列守护进程。

`__version__` 是全仓库的版本号唯一来源（single source of truth）：
- FastAPI app（server.py）通过它注入 `app.version` + `/api/health` 暴露
- 前端 Sidebar 通过 `/api/health` 拉取，不再硬编码
- `studio/web/package.json` 的 version 字段需手动同步保持一致
- 每次 release 改这里 + 在 CHANGELOG.md 加一段 + 同步 package.json

版本规则：MAJOR.MINOR.PATCH（语义版本，但 0.x 阶段 MINOR 即视为破坏性升级）。
"""
__version__ = "0.6.0"
