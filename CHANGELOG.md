# Changelog

仓库版本号唯一来源是 `studio/__init__.py` 的 `__version__`。FastAPI（`/api/health`）和前端 Sidebar 都从它派生。`studio/web/package.json` 的 `version` 字段需手动同步保持一致。

每次 release 改 `__version__` + 同步 `package.json` + 在本文件加一段。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本规则按语义化版本（0.x 阶段 MINOR 视为破坏性升级）。

---

## [0.5.0] — 2026-05-09

累计 49 commits / 132 files (+17k / -1.6k)。集中在 4 块：测试出图、先验生成、Setup 重写、Settings 拆分 + 新 tagger（CLTagger）。

### 新增

- **测试出图（Generate）**
  - 侧栏「测试」入口；`/api/generate` + `tools/anima_generate.py`（#19）
  - 推理 daemon（常驻 GPU，避免每次重载）+ XY 矩阵评测（参数扫）（#22）
  - `inference_core` 抽出，修多 LoRA 加载 P0 bug（#19）
  - SSE 改共享一条 EventSource，解 outputs/刷页面挂死
  - favicon 随机轮换（noal_*.png）
- **先验生成（无 LoRA）**
  - Step 4 加「先验生成」tab + explainer
  - `/api/projects/.../reg/generate-prior` + `tools/anima_reg_ai.py`
  - `RegMeta.generation_method` 区分手工 / AI 生成
- **Setup & 环境**
  - `studio.bat` 纯 ASCII 守护（cp936 cmd.exe 不再炸）+ 单测兜底
  - bootstrap：Windows 优先 `py -3`，Linux 迭代版本检查
  - venv stale check + `--reinstall` flag（环境救命）
  - 首装 GPU-aware torch；CPU-only 误装大警告
  - defer torch reinstall 到 launcher 进程，解 Windows 锁文件 + 自愈僵尸目录
  - Settings 加 PyTorch section，一键重装为 CUDA 版
  - `studio.sh --mirror` flag + HF 镜像端点可配置（Settings UI toggle）
  - ONNX CUDA 错误推理期自动降 CPU；系统 CUDA 时跳过 torch wheel preload
- **Attention Backend（#21）**
  - `attention_backend` 单字段替代 `xformers` / `flash_attn` 双 bool
  - `/api/xformers/{status,install}` + Settings xformers 卡片
  - 加速三选一下拉
  - flash_attn 一键装 wheel + 模型层 fast path + CLI 入口
  - `detect_env` 改用 torch ABI 拿 cuda_tag，不依赖 nvidia-smi
- **Tagger**
  - 新 CLTagger（外部贡献，#14）
  - 抽 `OnnxTaggerBase`，CLTagger 自动获得 PP10 线程池
  - tagger registry + 统一 `<name>_overrides` 持久化键
- **版本控制**
  - 版本号集中到 `studio/__init__.py:__version__`，FastAPI / Sidebar 都从这派生
  - 新建本 `CHANGELOG.md`
- **文档结构重构**
  - 拆 `docs/` 为三块：`user-guide/`（用户向）、`architecture/`（开发者向）、`adr/`（决策记录）
  - 新建 `docs/README.md` 总入口 + `docs/adr/README.md` 含 ADR 模板
  - 三篇互斥方案文档合并为 [ADR 0001 — LoKr 走 lycoris-lora 而不切 sd-scripts](docs/adr/0001-lokr-via-lycoris-lora.md)
  - 删除已落地的 11 篇 PP 阶段 plan（`studio-pipeline/PP0–PP10`），保留 overview 改写为 `architecture/studio-pipeline.md`
  - 删除过期的 `trainer-optimization-analysis.md`（2025-02 快照，建议项已落地）
  - `docs/_local/` 进 `.gitignore` 收个人草稿

### 变更

- **Settings 页**
  - 拆 4 个 tab：数据集 / 打标 / 训练 / 页面
  - ONNX Runtime 拆独立 section
  - WD14 / CLTagger 改 anima 主模型样式（radio + 行内下载）
  - 字段对齐 + 2K 屏留白修复
- 训练脚本搬到 `scripts/` + `tools/`，淘汰 `monitor_smooth.html`
- `LoraEntry` 抽到 `schema.py`（收尾 PR-9）
- 隐藏「监控与进度」组，`no_progress` 默认改 True

### 修复

- patch lycoris-lora 3.4.0 `LokrModule.get_weight` rank_dropout device bug
- stale 检测 mtime 改回并联，本地未 commit 编辑也触发重建
- 折叠态干掉单独的「导出训练集」按钮，避免误触
- 修补 PR #14 遗留的 UX 与测试漏洞

### 子 PR（已合到 dev）

- #14 CLTagger 支持（外部贡献）
- #19 PR-17 borrowed（Generate Phase 1）
- #20 PR-17 part 2（reg / inference_core 收尾）
- #21 attention_backend 整合
- #22 测试页面重设计 Phase 2（XY / daemon / 评测）

---

## [0.1.0] — 初始版本

`__version__` 字段诞生时的占位版本号（FastAPI app version 与 package.json 同此）；当时 Sidebar 显示的是手写的 `0.4`，未与代码版本号对齐。本次 0.5.0 起统一治理。
