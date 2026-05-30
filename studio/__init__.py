"""AnimaStudio — 训练监控、配置编辑与任务队列守护进程。

## 顶层架构（0.11.0 重构后，详 ADR-0008）

studio/ 4 层（依赖方向：自上而下，**不允许反向**）：

```
    api/             HTTP 表面（FastAPI app + 27 router + schemas + deps）
      ↓
    services/        业务服务（11 子包：tagging / booru / reg / inference /
                    models / preprocess / projects / dataset / presets /
                    runtime / data_io）
      ↓
    domain/          pydantic 模型（TrainingConfig 643 行 + LoRA / XY / Generate
                    / RegAi + migrations）
      ↓
    infrastructure/  路径常量 / 数据库 / event bus / secrets / 日志 / argparse
                    桥接 / migrations
```

`supervisor/`（任务调度守护线程）和 `workers/`（4 个子进程入口）跨层使用，
不归 4 层之一。

## 入口文件

- `server.py` — 51 行 shim，re-export `app` / `main`，给老 `from studio.server`
  路径兼容（真实实现在 api/app.py / api/main.py）
- `cli.py` — `python -m studio` launcher（build / run / dev / test 子命令）
- `__main__.py` — `python -m studio` 入口

## __version__

全仓库版本号唯一来源（single source of truth）：
- FastAPI app 通过它注入 `app.version` + `/api/health` 暴露
- 前端 Sidebar 通过 `/api/health` 拉取，不再硬编码
- `studio/web/package.json` 的 version 字段需手动同步保持一致
- 每次 release 改这里 + 在 CHANGELOG.md 加一段 + 同步 package.json

版本规则：MAJOR.MINOR.PATCH（语义版本，但 0.x 阶段 MINOR 即视为破坏性升级）。
"""
__version__ = "0.10.3"
