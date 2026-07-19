# AnimaStudio

训练流水线（抓图 → 筛选 → 打标 → 正则 → 训练 → 出图测试）的 Web 工作台。后端 FastAPI + SQLite，前端 React + Vite。

## 目录结构（ADR 0008 四层架构）

```
studio/
├── api/               # HTTP 表面：FastAPI app + routers + schemas + deps + exception_handlers
├── services/          # 业务服务子包：tagging / booru / reg / inference / models（含 families/ 族资产 registry）/
│                      #   preprocess / projects / dataset / presets / runtime / data_io
├── domain/            # pydantic 模型：TrainingConfig（含 model_family / 能力门控 / config_rules）/
│                      #   LoRA / XY / Generate / RegAi / family_switch + migrations
├── infrastructure/    # 路径 / 数据库 / event bus / secrets / 日志 / argparse 桥接 / migrations
├── supervisor/        # 任务调度守护线程
├── workers/           # 后台子进程入口（download / tag / reg_build / preprocess）
├── server.py          # 兼容 shim，re-export `app` / `main`（真实入口在 api/app.py / api/main.py）
└── web/               # React + Vite 前端源码
    ├── src/
    └── dist/          # npm run build 产物（后端挂在根路径 /，ADR 0012）
```

`schema.py` / `secrets.py` / `paths.py` 等根级文件是重构前 API 的兼容 shim，真身分别在
`domain/training.py`、`infrastructure/secrets.py`、`services/models/paths.py`。

运行时数据写到仓库根目录下的 `studio_data/`（SQLite + 用户 preset + 任务档案），已加入 `.gitignore`。

## 启动

### 跨平台启动器（推荐）

`python -m studio` 是统一入口，子进程由 Python 管理（Windows / macOS / Linux 都用一样的命令）：

```bash
python -m studio              # 默认 = run
python -m studio run          # 构建前端（如缺）+ 起后端
python -m studio dev          # 前后端开发模式（5173 + 8765 --reload，并行）
python -m studio build        # 仅构建前端
python -m studio test         # 跑 pytest + vitest
```

dev 模式会同时起 Vite 和 uvicorn 两个子进程，Ctrl+C 会一起干掉（Windows 用 `CTRL_BREAK_EVENT`，POSIX 用进程组 SIGTERM）。

### Windows 快捷脚本

`studio.bat` 调同一份 Python 启动器，双击即可。

### 直接调后端

```bash
python -m studio.server --host 0.0.0.0 --port 8765 [--reload]
```

### 前端

开发模式（热重载）：

```bash
cd studio/web
npm install            # 首次
npm run dev            # → http://127.0.0.1:5173/（/api、/samples 反代到后端）
```

生产构建（产物给后端挂在根路径 `/`，ADR 0012）：

```bash
cd studio/web
npm run build          # 输出到 studio/web/dist/
# 后端不用重启，刷新浏览器即可（服务端启动时检测 dist/ 是否存在）
```

## 前端页面

- **项目**（`/`）— 项目列表；进入项目后侧栏切 Stepper（下载 / 筛选 / 预处理 / 打标 / 标签编辑 / 正则集 / 训练），训练配置含模型族（Anima / Krea 2）切换
- **队列**（`/queue`）— 全类型任务统一台账：取消 / 重试 / 删除 / 暂停恢复 / 定时；分区 + 过滤分页；SSE 实时刷新；任务详情含日志 / 监控 / 输出
- **预设**（`/tools/presets`）— 全局训练预设池（自动保存），与 version 配置双向 fork
- **测试**（`/tools/generate`）— 单图 / XY 矩阵 / 常驻推理 daemon；按族选底模 / TE，fp8 / 显存策略
- **监控**（`/tools/monitor`）— 训练实时 loss / lr / 采样图
- **设置**（`/tools/settings`）— 按 tab 分区；模型下载中心（按族分区 + variant 单选）在「训练」tab

跨步骤架构（数据模型 / SQLite / SSE / secrets / Tagger 抽象 / Preset 池）见
[docs/architecture/studio-pipeline.md](../docs/architecture/studio-pipeline.md)。
